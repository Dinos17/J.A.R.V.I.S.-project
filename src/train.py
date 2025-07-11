"""
J.A.R.V.I.S. Training Pipeline
Implements streaming training for massive datasets with memory efficiency.
"""

import os
import json
import logging
import torch
import wandb
from pathlib import Path
from typing import Dict, List, Optional, Iterator
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
    Trainer, DataCollatorForLanguageModeling, EarlyStoppingCallback
)
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from tqdm import tqdm
import gc
import glob

# Import our data processing modules
from data.preprocess import StreamingDataProcessor, DatasetManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for J.A.R.V.I.S. training."""
    
    # Model configuration
    model_name: str = "gpt2"
    max_length: int = 256  # Reduced from 512 for speed
    batch_size: int = 1  # Reduced for CPU training
    gradient_accumulation_steps: int = 16  # Increased to maintain effective batch size
    
    # Training parameters
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 50  # Reduced
    weight_decay: float = 0.01
    
    # Memory optimization
    gradient_checkpointing: bool = False  # Disabled for speed
    fp16: bool = False  # Disabled for CPU
    dataloader_pin_memory: bool = False
    
    # LoRA configuration for efficient fine-tuning
    use_lora: bool = True
    lora_r: int = 8  # Reduced from 16 for speed
    lora_alpha: int = 16  # Reduced from 32
    lora_dropout: float = 0.1
    
    # Checkpointing
    save_steps: int = 1000  # Less frequent saving
    eval_steps: int = 1000
    save_total_limit: int = 2  # Keep fewer checkpoints
    
    # Data processing
    chunk_size: int = 1000
    max_memory_usage: int = 1500  # MB
    
    # Output paths
    output_dir: str = "models/JARVIS"
    checkpoint_dir: str = "data/checkpoints"

class StreamingDataset:
    """Handles streaming dataset loading for memory efficiency."""
    
    def __init__(self, data_glob: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_files = [Path(f) for f in glob.glob(data_glob)]
        self.current_chunk_idx = 0
        self.current_data = None
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_data is None or len(self.current_data) == 0:
            if self.current_chunk_idx >= len(self.chunk_files):
                raise StopIteration
            
            # Load next chunk
            chunk_file = self.chunk_files[self.current_chunk_idx]
            self.current_data = self._load_chunk(chunk_file)
            self.current_chunk_idx += 1
        
        return self.current_data.pop(0)
    
    def _load_chunk(self, chunk_file: Path) -> List[Dict]:
        """Load a chunk of data from file."""
        data = []
        with open(chunk_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    # Tokenize the text
                    tokenized = self._tokenize_text(item['text'])
                    if tokenized:
                        data.append(tokenized)
                except (json.JSONDecodeError, KeyError):
                    continue
        
        return data
    
    def _tokenize_text(self, text: str) -> Optional[Dict]:
        """Tokenize text with proper conversation formatting."""
        try:
            # Ensure proper conversation format
            if not text.strip().endswith("Assistant:"):
                text = text.strip() + "\nAssistant:"
            
            # Add system prompt for better conversation understanding
            if not text.startswith("J.A.R.VIS is an AI assistant"):
                text = "J.A.R.VIS is an AI assistant. Be helpful, friendly, and concise.\n\n" + text
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': encoding['input_ids'].squeeze().clone()
            }
        except Exception as e:
            logger.warning(f"Tokenization error: {e}")
            return None

class JARVISTrainer:
    """Main trainer class for J.A.R.V.I.S."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Initialize components
        self._setup_directories()
        self._setup_logging()
    
    def _setup_directories(self):
        """Create necessary directories."""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging and wandb."""
        try:
            wandb.init(
                project="jarvis-training",
                name="jarvis-pretraining",
                config=vars(self.config)
            )
        except Exception as e:
            logger.warning(f"Wandb initialization failed: {e}")
    
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer."""
        logger.info(f"Loading model: {self.config.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )

        # Apply LoRA if enabled
        if self.config.use_lora:
            self._apply_lora()
        
        # Temporarily disable gradient checkpointing to avoid gradient issues
        # if self.config.gradient_checkpointing:
        #     self.model.gradient_checkpointing_enable()
        
        # Ensure model is in training mode
        self.model.train()
        
        # Double-check that LoRA parameters have gradients enabled
        for name, param in self.model.named_parameters():
            if "lora" in name:
                param.requires_grad_(True)
        
        logger.info(f"Model loaded successfully. Parameters: {self.model.num_parameters():,}")

    def _apply_lora(self):
        """Apply LoRA configuration for efficient fine-tuning."""
        # GPT-2 specific target modules
        target_modules = ["c_attn", "c_proj", "c_fc"]
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Freeze base model parameters
        for name, param in self.model.named_parameters():
            if "lora" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # Print trainable parameters
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        logger.info(f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.4f}")
        
        # Ensure model is in training mode
        self.model.train()
    
    def prepare_training_data(self, dataset_globs) -> Dataset:
        """Prepare training data with streaming capabilities."""
        if isinstance(dataset_globs, str):
            dataset_globs = [dataset_globs]
        logger.info(f"Preparing training data from: {dataset_globs}")
        all_data_list = []
        for data_glob in dataset_globs:
            logger.info(f"Loading data from: {data_glob}")
            streaming_dataset = StreamingDataset(
                data_glob,
                self.tokenizer,
                self.config.max_length
            )
            data_list = []
            # Reduce samples for faster training
            max_samples_per_dataset = 500  # Reduced from 1000
            for item in tqdm(streaming_dataset, desc=f"Loading {data_glob}"):
                data_list.append(item)
                if len(data_list) >= max_samples_per_dataset:
                    break
            all_data_list.extend(data_list)
            logger.info(f"Loaded {len(data_list)} samples from {data_glob}")
        dataset = Dataset.from_list(all_data_list)
        logger.info(f"Combined dataset prepared with {len(dataset)} total samples")
        return dataset
    
    def setup_trainer(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """Setup the trainer with proper configuration."""

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_steps=5,  # More frequent logging for monitoring
            save_steps=self.config.save_steps,
            save_strategy="steps",
            save_total_limit=self.config.save_total_limit,
            fp16=False,  # Disable fp16 to avoid issues
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="wandb" if wandb.run else None,
            run_name="jarvis-training",
            ddp_find_unused_parameters=False,
            gradient_checkpointing=False,  # Disable to avoid gradient issues
            dataloader_num_workers=0,  # Disable multiprocessing
            # CPU optimizations
            dataloader_prefetch_factor=None,
            torch_compile=False,  # Disable for CPU
            optim="adamw_torch",  # Use PyTorch optimizer
            lr_scheduler_type="linear",
            # Reduce memory usage
            max_grad_norm=1.0,
            # Faster training settings
            group_by_length=False,
            length_column_name="length",
            ignore_data_skip=False,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Ensure the model is ready for training
        self.model.train()
        for param in self.model.parameters():
            if hasattr(param, 'requires_grad') and param.requires_grad:
                param.requires_grad_(True)

        # Callbacks
        callbacks = []
        if eval_dataset:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )
    
    def train(self, dataset_globs, eval_dataset_path: Optional[str] = None):
        """Main training function."""
        logger.info("Starting J.A.R.V.I.S. training pipeline")
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Check if model has trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model has {trainable_params:,} trainable parameters")
        
        if trainable_params == 0:
            logger.error("No trainable parameters found! Check LoRA configuration.")
            raise ValueError("Model has no trainable parameters")
        
        # Prepare training data
        train_dataset = self.prepare_training_data(dataset_globs)
        
        # Prepare evaluation data if provided
        eval_dataset = None
        if eval_dataset_path:
            eval_dataset = self.prepare_training_data(eval_dataset_path)
        
        # Setup trainer
        self.setup_trainer(train_dataset, eval_dataset)
        
        # Start training
        logger.info("Beginning training...")
        try:
            # Verify model is ready
            logger.info(f"Model device: {next(self.model.parameters()).device}")
            logger.info(f"Model training mode: {self.model.training}")
            
            # Check a few parameters to ensure gradients are enabled
            trainable_count = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    trainable_count += 1
                    if trainable_count <= 3:  # Log first 3 trainable parameters
                        logger.info(f"Trainable parameter: {name}, requires_grad: {param.requires_grad}")
            
            logger.info(f"Total trainable parameters: {trainable_count}")
            
            train_result = self.trainer.train()
            
            # Save final model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            # Also save the base model weights for inference compatibility
            if hasattr(self.model, 'base_model'):
                self.model.base_model.save_pretrained(self.config.output_dir)
            else:
                self.model.save_pretrained(self.config.output_dir)
            
            # Log training results
            logger.info(f"Training completed. Loss: {train_result.training_loss}")
            
            # Save training metrics
            metrics = train_result.metrics
            with open(f"{self.config.output_dir}/training_metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def resume_training(self, checkpoint_path: str):
        """Resume training from a checkpoint."""
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Load checkpoint
        self.trainer = Trainer.from_pretrained(checkpoint_path)
        
        # Continue training
        self.trainer.train(resume_from_checkpoint=checkpoint_path)

def main():
    """Main training function."""
    
    # Configuration
    config = TrainingConfig()
    
    # Initialize trainer
    trainer = JARVISTrainer(config)
    
    # Training paths (replace with actual paths)
    pretraining_data_path = "data/processed/redpajama"
    finetuning_data_path = "data/processed/sharegpt52k"
    
    # Check if data exists
    if not os.path.exists(pretraining_data_path):
        logger.error(f"Pretraining data not found: {pretraining_data_path}")
        logger.info("Please run the data preprocessing pipeline first.")
        return
    
    # Start training
    try:
        # Phase 1: Pretraining on massive text datasets
        logger.info("Phase 1: Pretraining on RedPajama/C4")
        dataset_globs = [
            "data/processed/redpajama_chunk_*.jsonl",
            "data/processed/c4_chunk_*.jsonl",
            "data/processed/dialogstudio_chunk_*.jsonl"
        ]
        trainer.train(dataset_globs)
        
        # Phase 2: Fine-tuning on conversational data
        if os.path.exists(finetuning_data_path):
            logger.info("Phase 2: Fine-tuning on conversational data")
            config.model_name = config.output_dir  # Load pretrained model
            config.learning_rate = 1e-5  # Lower learning rate for fine-tuning
            config.num_epochs = 1  # Fewer epochs for fine-tuning
            
            trainer = JARVISTrainer(config)
            trainer.train(finetuning_data_path)
        
        logger.info("J.A.R.V.I.S. training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 