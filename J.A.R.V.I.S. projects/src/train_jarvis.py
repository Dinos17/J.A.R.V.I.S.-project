#!/usr/bin/env python3
"""
J.A.R.V.I.S. Training Pipeline - Fixed Version
Properly handles conversation format and separates pretraining/finetuning phases.
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

# FAST MODE for quick training/testing
FAST_MODE = False  # Set to True for fastest possible training

# Maximize CPU usage for fastest training
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
torch.set_num_threads(8)
np.set_printoptions(precision=3, suppress=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for J.A.R.V.I.S. training."""
    
    # Model configuration
    model_name: str = "gpt2"
    max_length: int = 256
    batch_size: int = 16  # Increased for speed
    gradient_accumulation_steps: int = 16
    
    # Training parameters
    learning_rate: float = 5e-5
    num_epochs: int = 1  # Only 1 epoch for fastest run
    warmup_steps: int = 50
    weight_decay: float = 0.01
    
    # Memory optimization
    gradient_checkpointing: bool = False
    fp16: bool = False
    dataloader_pin_memory: bool = False
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    # Checkpointing
    save_steps: int = 50000  # Less frequent for speed
    eval_steps: int = 50000  # Less frequent for speed
    save_total_limit: int = 2
    
    # Data processing
    chunk_size: int = 1000
    max_memory_usage: int = 1500
    
    # Output paths
    output_dir: str = "models/JARVIS"
    checkpoint_dir: str = "data/checkpoints"

class ConversationDataset:
    """Handles conversation dataset loading with proper formatting."""
    
    def __init__(self, data_glob: str, tokenizer, max_length: int = 256):
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
        """Load a chunk of conversation data from file."""
        data = []
        with open(chunk_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    # Tokenize the conversation text
                    tokenized = self._tokenize_conversation(item['text'])
                    if tokenized:
                        data.append(tokenized)
                except (json.JSONDecodeError, KeyError):
                    continue
        
        return data
    
    def _tokenize_conversation(self, text: str) -> Optional[Dict]:
        """Tokenize conversation text with proper formatting."""
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
    """Main trainer class for J.A.R.VIS."""
    
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
                name="jarvis-conversation-training",
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
    
    def prepare_conversation_data(self, dataset_globs) -> Dataset:
        """Prepare conversation training data."""
        if isinstance(dataset_globs, str):
            dataset_globs = [dataset_globs]
        logger.info(f"Preparing conversation data from: {dataset_globs}")
        all_data_list = []
        for data_glob in dataset_globs:
            logger.info(f"Loading conversation data from: {data_glob}")
            conversation_dataset = ConversationDataset(
                data_glob,
                self.tokenizer,
                self.config.max_length
            )
            data_list = []
            # max_samples_per_dataset = 5000  # Remove or comment this out
            for item in tqdm(conversation_dataset, desc=f"Loading {data_glob}"):
                data_list.append(item)
                # if len(data_list) >= max_samples_per_dataset:
                #     break
            all_data_list.extend(data_list)
            logger.info(f"Loaded {len(data_list)} conversation samples from {data_glob}")
        dataset = Dataset.from_list(all_data_list)
        logger.info(f"Combined conversation dataset prepared with {len(dataset)} total samples")
        return dataset
    
    def setup_trainer(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """Setup the trainer with proper configuration."""

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,  # Use config batch size
            per_device_eval_batch_size=self.config.batch_size,   # Use config batch size
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_steps=5,
            save_steps=self.config.save_steps,  # Use config save_steps
            save_strategy="steps",
            save_total_limit=self.config.save_total_limit,
            fp16=False,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,  # Fully disable W&B logging
            run_name="jarvis-conversation-training",
            ddp_find_unused_parameters=False,
            gradient_checkpointing=False,
            dataloader_num_workers=8,  # Increased for speed
            dataloader_prefetch_factor=None,
            torch_compile=False,
            optim="adamw_torch",
            lr_scheduler_type="linear",
            max_grad_norm=1.0,
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
    
    def train_conversations(self, dataset_globs):
        """Train specifically on conversation data."""
        logger.info("Starting J.A.R.VIS conversation training")
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Check if model has trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model has {trainable_params:,} trainable parameters")
        
        if trainable_params == 0:
            logger.error("No trainable parameters found! Check LoRA configuration.")
            raise ValueError("Model has no trainable parameters")
        
        # Prepare conversation training data
        train_dataset = self.prepare_conversation_data(dataset_globs)
        
        # Setup trainer
        self.setup_trainer(train_dataset)
        
        # Start training
        logger.info("Beginning conversation training...")
        try:
            train_result = self.trainer.train()
            
            # Save final model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            # Log training results
            logger.info(f"Conversation training completed. Loss: {train_result.training_loss}")
            
            # Save training metrics
            metrics = train_result.metrics
            with open(f"{self.config.output_dir}/conversation_training_metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            
        except Exception as e:
            logger.error(f"Conversation training failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

def main():
    """Main training function for conversation training."""
    
    # Configuration for conversation training
    config = TrainingConfig()
    config.learning_rate = 3e-5  # Slightly lower for conversation training
    config.num_epochs = 2  # Fewer epochs for conversation training

    # FAST MODE overrides for speed
    if FAST_MODE:
        config.num_epochs = 1
        config.batch_size = 2
        config.max_length = 128
        max_samples_per_dataset = 100
        print("[JARVIS] FAST_MODE enabled: Using 1 epoch, batch size 2, max_length 128, 100 samples per dataset.")
    else:
        max_samples_per_dataset = None
        if not torch.cuda.is_available():
            print("[JARVIS] WARNING: No GPU detected. For fastest results, enable FAST_MODE or use a smaller dataset.")

    # Initialize trainer
    trainer = JARVISTrainer(config)
    
    # Conversation training paths (use all datasets for broader training)
    conversation_data_paths = [
        "data/processed/redpajama_chunk_*.jsonl",
        "data/processed/c4_chunk_*.jsonl",
        "data/processed/sharegpt52k_chunk_*.jsonl",
        "data/processed/dialogstudio_chunk_*.jsonl"
    ]
    
    # Check if conversation data exists
    data_exists = False
    for path_pattern in conversation_data_paths:
        if glob.glob(path_pattern):
            data_exists = True
            break
    
    if not data_exists:
        logger.error("Conversation training data not found!")
        logger.info("Please run the data preprocessing pipeline first to generate conversation data.")
        return
    
    # Patch the trainer to use max_samples_per_dataset in prepare_conversation_data
    orig_prepare = trainer.prepare_conversation_data
    def patched_prepare_conversation_data(dataset_globs):
        if max_samples_per_dataset is None:
            return orig_prepare(dataset_globs)
        if isinstance(dataset_globs, str):
            dataset_globs = [dataset_globs]
        logger.info(f"Preparing conversation data from: {dataset_globs}")
        all_data_list = []
        for data_glob in dataset_globs:
            logger.info(f"Loading conversation data from: {data_glob}")
            conversation_dataset = ConversationDataset(
                data_glob,
                trainer.tokenizer,
                config.max_length
            )
            data_list = []
            for item in tqdm(conversation_dataset, desc=f"Loading {data_glob}"):
                data_list.append(item)
                if len(data_list) >= max_samples_per_dataset:
                    break
            all_data_list.extend(data_list)
            logger.info(f"Loaded {len(data_list)} conversation samples from {data_glob}")
        dataset = Dataset.from_list(all_data_list)
        logger.info(f"Combined conversation dataset prepared with {len(dataset)} total samples")
        return dataset
    trainer.prepare_conversation_data = patched_prepare_conversation_data

    # Start conversation training
    try:
        logger.info("Training J.A.R.VIS on all datasets (fast mode)...")
        trainer.train_conversations(conversation_data_paths)
        logger.info("J.A.R.VIS conversation training completed successfully!")
        
    except Exception as e:
        logger.error(f"Conversation training failed: {e}")
        raise

if __name__ == "__main__":
    main()

# DATASET LINKS (for reference)
# 1️⃣ RedPajama-Data-1T — https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T
# 2️⃣ C4 — https://huggingface.co/datasets/allenai/c4
# 3️⃣ ShareGPT52K — https://huggingface.co/datasets/RyokoAI/ShareGPT52K
# 4️⃣ DialogStudio — https://huggingface.co/datasets/Salesforce/dialogstudio 