#!/usr/bin/env python3
"""
J.A.R.V.I.S. Advanced Training Pipeline
Unified training and data processing with multiple strategies.
"""

import os
import sys
import json
import logging
import hashlib
import tempfile
import shutil
import glob
import gc
import platform
import itertools
from pathlib import Path
from typing import List, Dict, Optional, Any, Iterator, Union
from dataclasses import dataclass
import warnings
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM
)
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback, EarlyStoppingCallback
from transformers.data.data_collator import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
import datasets
from datasets import Dataset as HFDataset, load_dataset
import wandb
from tqdm import tqdm
import numpy as np
import requests
import psutil
import torch_directml

# Disable multiprocessing on Windows to avoid handle pointer errors
if platform.system() == "Windows":
    torch.multiprocessing.set_start_method('spawn', force=True)

# MAXIMUM SPEED OPTIMIZATIONS
os.environ["OMP_NUM_THREADS"] = "16"  # Increased from 8 to 16
os.environ["MKL_NUM_THREADS"] = "16"  # Increased from 8 to 16
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Enable parallel tokenization
torch.set_num_threads(16)  # Increased from 8 to 16
np.set_printoptions(precision=3, suppress=True)

# Disable warnings for speed
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Advanced configuration for J.A.R.V.I.S. training."""
    # Model
    model_name: str = "gpt2"
    batch_size: int = 8
    max_length: int = 64
    # Training
    learning_rate_pretraining: float = 3e-5
    learning_rate_finetuning: float = 5e-6
    gradient_accumulation_steps: int = 4
    max_steps: int = 500000
    num_epochs: int = 1
    warmup_steps: int = 50
    weight_decay: float = 0.01
    # Memory
    chunk_size: int = 500
    max_memory_usage: int = 1200
    pin_memory: bool = True
    # LoRA
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_bias: str = "none"
    lora_target_modules: tuple = ("c_attn", "c_proj", "c_fc")
    # Output
    output_dir: str = "models/JARVIS"
    checkpoint_dir: str = "data/checkpoints"
    logs_dir: str = "logs"
    # Checkpointing
    save_steps: int = 50000
    eval_steps: int = 50000
    save_total_limit: int = 1
    # Data
    max_samples_per_dataset: int = None
    sample_skip_rate: int = 50
    # Mixed precision
    fp16: bool = True
    gradient_checkpointing: bool = True
    dataloader_pin_memory: bool = True

class AdvancedDataset(Dataset):
    """Advanced dataset class with deduplication and caching."""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class StreamingDataset(IterableDataset):
    """Streaming dataset for memory-efficient processing."""
    
    def __init__(self, data_iterator, tokenizer, max_length=256):
        self.data_iterator = data_iterator
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __iter__(self):
        for sample in self.data_iterator:
            yield sample

class DeduplicationProcessor:
    """Handles dataset deduplication and statistics tracking."""
    
    def __init__(self):
        self.seen_samples = set()
        self.stats = {
            'total_samples': 0,
            'unique_samples': 0,
            'duplicate_samples': 0
        }
    
    def create_sample_hash(self, sample: Dict) -> str:
        """Create a hash of the sample content for deduplication."""
        text = sample.get('text', '')
        if not text:
            text = sample.get('content', '')
        if not text:
            text = str(sample)
        return hashlib.md5(text.encode()).hexdigest()
    
    def process_sample(self, sample: Dict) -> Optional[Dict]:
        """Process a sample and return it if unique."""
        self.stats['total_samples'] += 1
        sample_hash = self.create_sample_hash(sample)
        
        if sample_hash not in self.seen_samples:
            self.seen_samples.add(sample_hash)
            self.stats['unique_samples'] += 1
            return sample
        else:
            self.stats['duplicate_samples'] += 1
            return None
    
    def get_stats(self) -> Dict:
        """Get current deduplication statistics."""
        return self.stats.copy()

class ConversationFormatter:
    """Handles conversation formatting for different datasets."""
    
    @staticmethod
    def format_sharegpt52k(sample: Dict) -> Optional[str]:
        """Format ShareGPT52K conversation data."""
        conversations = sample.get('conversations', None)
        
        # Handle string conversations
        if isinstance(conversations, str):
            try:
                conversations = json.loads(conversations)
            except Exception:
                return None
        
        if not isinstance(conversations, list):
            return None
        
        text_parts = []
        for conv in conversations:
            if isinstance(conv, dict):
                role = conv.get('from', '')
                content = conv.get('value', '')
                if role and content:
                    text_parts.append(f"{role}: {content}")
        
        if not text_parts:
            return None
        
        text = "\n".join(text_parts)
        if len(text.strip()) < 10:
            return None
        
        # Add system prompt
        if not text.startswith("J.A.R.VIS is an AI assistant"):
            text = "J.A.R.VIS is an AI assistant. Be helpful, friendly, and concise.\n\n" + text
        
        return text
    
    @staticmethod
    def format_dialogstudio(sample: Dict) -> Optional[str]:
        """Format DialogStudio conversation data."""
        dialog = sample.get('log', None)
        if not dialog or not isinstance(dialog, list):
            return None
        
        text_parts = []
        for turn in dialog:
            if isinstance(turn, dict):
                user = turn.get('user utterance', '').strip()
                system = turn.get('system response', '').strip()
                if user:
                    text_parts.append(f"User: {user}")
                if system:
                    text_parts.append(f"Assistant: {system}")
        
        if not text_parts:
            return None
        
        text = "\n".join(text_parts)
        if len(text.strip()) < 10:
            return None
        
        # Add system prompt
        if not text.startswith("J.A.R.VIS is an AI assistant"):
            text = "J.A.R.VIS is an AI assistant. Be helpful, friendly, and concise.\n\n" + text
        
        return text
    
    @staticmethod
    def format_plain_text(sample: Dict) -> Optional[str]:
        """Format plain text data."""
        text = sample.get('text', '')
        if not text or len(text.strip()) < 10:
            return None
        
        # Add system prompt for consistency
        if not text.startswith("J.A.R.VIS is an AI assistant"):
            text = "J.A.R.VIS is an AI assistant. Be helpful, friendly, and concise.\n\n" + text
        
        return text

class ProgressBarCallback(TrainerCallback):
    """Custom progress bar callback with detailed metrics."""
    
    def __init__(self, total_steps: Optional[int] = None):
        super().__init__()
        self.pbar = tqdm(total=total_steps, desc="Training Progress", position=0, leave=True, dynamic_ncols=True)
        self.last_step = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        steps = state.global_step - self.last_step
        if steps > 0:
            self.pbar.update(steps)
            self.last_step = state.global_step
            
            # Get latest loss
            loss = state.log_history[-1]['loss'] if state.log_history and 'loss' in state.log_history[-1] else 'N/A'
            eta = self.pbar.format_dict.get('remaining', 'N/A')
            
            self.pbar.set_postfix({
                "Step": state.global_step,
                "Loss": f"{loss:.4f}" if isinstance(loss, float) else loss,
                "ETA": eta
            })
    
    def on_train_end(self, args, state, control, **kwargs):
        self.pbar.close()

# Device selection for training

def get_training_device():
    import torch
    if torch.cuda.is_available():
        print('[JARVIS] CUDA GPU detected. Training will use CUDA.')
        return torch.device('cuda')
    else:
        print('[JARVIS WARNING] No CUDA GPU detected. Training will use CPU and may be very slow.')
        return torch.device('cpu')

class AdvancedJARVISTrainer:
    """Advanced J.A.R.V.I.S. trainer with integrated data processing."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = get_training_device()
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.deduplicator = DeduplicationProcessor()
        self.formatter = ConversationFormatter()
        
        # Setup directories
        self._setup_directories()
        self._setup_logging()
    
    def _setup_directories(self):
        """Create necessary directories, skipping logs_dir to avoid log file creation."""
        directories = [
            self.config.output_dir,
            self.config.checkpoint_dir
            # self.config.logs_dir  # Do not create logs_dir
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging and wandb."""
        try:
            wandb.init(
                project="jarvis-advanced-training",
                name=f"jarvis-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=vars(self.config)
            )
        except Exception as e:
            logger.warning(f"Wandb initialization failed: {e}")
    
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer with device optimizations."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer with fast settings
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=True,  # Use fast tokenizer
            model_max_length=self.config.max_length
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # CUDA/CPU optimizations
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.85)
            logger.info(f"CUDA GPU detected: {torch.cuda.get_device_name()}")
            logger.info(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        # Load model with device-optimized settings
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.fp16 and torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
        )
        self.model = self.model.to(self.device)
        
        # Apply LoRA if enabled
        if self.config.use_lora:
            self._apply_lora()
        
        # Ensure model is in training mode
        self.model.train()
        
        logger.info(f"Model loaded successfully. Parameters: {self.model.num_parameters():,}")
    
    def _apply_lora(self):
        """Apply LoRA configuration for efficient fine-tuning."""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias=self.config.lora_bias
        )
        
        if self.model is not None:
            self.model = get_peft_model(self.model, lora_config)  # type: ignore
            
            # Freeze base model parameters, unfreeze LoRA params
            for name, param in self.model.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            # Print trainable parameters
            trainable_params = 0
            all_param = 0
            print("[JARVIS DEBUG] Trainable parameters:")
            for name, param in self.model.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                    print(f"[JARVIS DEBUG] {name} is trainable, shape: {param.shape}")
            logger.info(f"Trainable params: {trainable_params:,} || All params: {all_param:,} || Trainable%: {100 * trainable_params / all_param:.4f}")
    
    def tokenize_text(self, text: str) -> Optional[Dict]:
        """Tokenize text with proper formatting."""
        if self.tokenizer is None:
            logger.error("Tokenizer not set!")
            return None
        
        try:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.config.max_length,
                return_tensors='pt'
            )
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            # For language modeling, labels should be input_ids (not detached, just a clone)
            labels = encoding['input_ids'].clone()
            # Do NOT set requires_grad on labels
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': labels.squeeze()
            }
        except Exception as e:
            logger.warning(f"Tokenization error: {e}")
            return None
    
    def load_dataset_streaming(self, dataset_config: Dict) -> Iterator[Dict]:
        """Load dataset with streaming for memory efficiency, robustly skipping corrupt samples."""
        try:
            dataset_name = dataset_config['name']
            dataset_id = dataset_config['dataset_id']
            config_name = dataset_config.get('config_name')
            split = dataset_config.get('split', 'train')
            trust_remote_code = dataset_config.get('trust_remote_code', False)
            data_files = dataset_config.get('data_files')
            max_samples = dataset_config.get('max_samples')
            sample_skip_rate = dataset_config.get('sample_skip_rate', self.config.sample_skip_rate)

            logger.info(f"Loading streaming dataset: {dataset_name}")

            # Use local cleaned file for ShareGPT52K if specified
            if dataset_name == "ShareGPT52K" and dataset_id == "json" and data_files:
                dataset = load_dataset('json', data_files=data_files, split='train', streaming=True)
            else:
                if config_name:
                    dataset = load_dataset(dataset_id, config_name, split=split, streaming=True, trust_remote_code=trust_remote_code)
                else:
                    dataset = load_dataset(dataset_id, split=split, streaming=True, trust_remote_code=trust_remote_code)

            sample_count = 0
            skip_rate = sample_skip_rate

            for i, sample in enumerate(dataset):  # type: ignore
                # Skip samples for speed (process every Nth sample)
                if i % skip_rate != 0:
                    continue
                try:
                    # Format based on dataset type
                    if dataset_name == "ShareGPT52K":
                        formatted_text = self.formatter.format_sharegpt52k(sample)  # type: ignore
                    elif dataset_name == "DialogStudio":
                        formatted_text = self.formatter.format_dialogstudio(sample)  # type: ignore
                    else:
                        formatted_text = self.formatter.format_plain_text(sample)  # type: ignore

                    if formatted_text:
                        # Deduplicate
                        processed_sample = self.deduplicator.process_sample({'text': formatted_text})
                        if processed_sample:
                            # Tokenize
                            tokenized = self.tokenize_text(formatted_text)
                            if tokenized:
                                yield tokenized
                                sample_count += 1
                except Exception as e:
                    logger.warning(f"Skipping corrupt or invalid sample: {e}")
                    continue

                # Log progress every 1000 samples
                if sample_count % 1000 == 0 and sample_count > 0:
                    stats = self.deduplicator.get_stats()
                    logger.info(f"{dataset_name}: Processed {sample_count} samples (skipping {skip_rate-1} out of {skip_rate}), {stats['unique_samples']} unique")

                # Stop if max_samples is reached
                if max_samples is not None and sample_count >= max_samples:
                    logger.info(f"Reached max_samples limit for {dataset_name}: {max_samples}")
                    break

        except Exception as e:
            logger.error(f"Error loading dataset {dataset_config['name']}: {e}")
    
    def download_train_save_delete(self, dataset_configs: List[Dict]):
        """Download → Train → Save → Delete pipeline for each dataset."""
        logger.info("Starting FAST Download → Train → Save → Delete pipeline...")
        
        # Load model and tokenizer once
        self.load_model_and_tokenizer()
        
        for i, config in enumerate(dataset_configs):
            if not config.get('enabled', True):
                continue

            dataset_name = config['name']
            max_samples = config.get('max_samples')
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing dataset {i+1}/{len(dataset_configs)}: {dataset_name}")
            logger.info(f"{'='*60}")

            # Improved log message for sample limit
            if max_samples:
                logger.info(f"Step 1: Downloading {dataset_name} (up to {max_samples} samples)...")
            else:
                logger.info(f"Step 1: Downloading {dataset_name} (ENTIRE DATASET)...")

            try:
                tokenized_data = []
                
                for sample in self.load_dataset_streaming(config):
                    tokenized_data.append(sample)
                    
                    if len(tokenized_data) % 2000 == 0:  # Less frequent logging for speed
                        logger.info(f"Collected {len(tokenized_data)} samples from {dataset_name}")
                
                if not tokenized_data:
                    logger.warning(f"No valid data collected from {dataset_name}, skipping...")
                    continue
                
                logger.info(f"Downloaded {len(tokenized_data)} samples from {dataset_name} (ENTIRE DATASET)")
                
                # Step 2: Train on the data
                logger.info(f"Step 2: Training on {dataset_name}...")
                self._train_on_dataset(tokenized_data, dataset_name, i)
                
                # Step 3: Save model checkpoint
                logger.info(f"Step 3: Saving checkpoint for {dataset_name}...")
                checkpoint_path = f"{self.config.output_dir}/checkpoint_{dataset_name}"
                if self.model is not None:
                    self.model.save_pretrained(checkpoint_path)
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(checkpoint_path)
                
                # Step 4: Clear data from memory (delete)
                logger.info(f"Step 4: Clearing {dataset_name} data from memory...")
                del tokenized_data
                gc.collect()
                
                logger.info(f"Completed processing {dataset_name}")
                
            except Exception as e:
                logger.error(f"Error processing {dataset_name}: {e}")
                continue
        
        logger.info("FAST Download → Train → Save → Delete pipeline completed!")
    
    def _train_on_dataset(self, tokenized_data: List[Dict], dataset_name: str, dataset_idx: int):
        """Train on a single dataset."""
        # Create dataset
        dataset = AdvancedDataset(tokenized_data, self.tokenizer, self.config.max_length)
        
        # Setup training arguments optimized for AMD Radeon RX 580
        training_args = TrainingArguments(
            output_dir=f"{self.config.output_dir}/temp_training_{dataset_name}",
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate_finetuning,
            max_steps=getattr(self.config, 'max_steps', 500000),
            logging_steps=200,  # More frequent logging for AMD monitoring
            save_steps=self.config.save_steps,
            eval_steps=None,
            save_strategy="steps",
            load_best_model_at_end=False,
            remove_unused_columns=False,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            dataloader_num_workers=0,  # Set to 0 for Windows/CPU compatibility
            fp16=self.config.fp16,  # Enable for AMD GPU
            report_to="wandb",
            logging_dir=None,
            run_name=f"jarvis-{dataset_name}-training",
            ddp_find_unused_parameters=False,
            gradient_checkpointing=self.config.gradient_checkpointing,  # Enable for 8GB VRAM
            torch_compile=False,  # Disabled for Windows compatibility
            optim="adamw_torch",
            lr_scheduler_type="linear",
            max_grad_norm=1.0,
            group_by_length=False,
            length_column_name="length",
            ignore_data_skip=False,
            warmup_steps=self.config.warmup_steps,  # Add warmup for AMD stability
        )
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,  # type: ignore
            mlm=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            callbacks=[ProgressBarCallback()]
        )
        
        # Fix label_names warning
        trainer.label_names = []
        
        # Train
        trainer.train()
        
        logger.info(f"Training completed on {dataset_name}")
    
    def train_streaming(self, dataset_configs: List[Dict]):
        """Train using streaming datasets for memory efficiency."""
        logger.info("Starting streaming training...")
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Create combined streaming dataset
        def combined_iterator():
            for config in dataset_configs:
                if config.get('enabled', True):
                    yield from self.load_dataset_streaming(config)
        
        train_dataset = StreamingDataset(combined_iterator(), self.tokenizer, self.config.max_length)
        
        # Setup training arguments optimized for AMD Radeon RX 580
        training_args = TrainingArguments(
            output_dir=f"{self.config.output_dir}/streaming_checkpoints",
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate_finetuning,
            max_steps=999999,  # Train until stopped
            logging_steps=200,  # More frequent logging for AMD monitoring
            save_steps=self.config.save_steps,
            eval_steps=None,
            save_strategy="steps",
            load_best_model_at_end=False,
            remove_unused_columns=False,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            dataloader_num_workers=0,  # Set to 0 for Windows/CPU compatibility
            fp16=self.config.fp16,  # Enable for AMD GPU
            report_to="wandb",
            logging_dir=f"{self.config.logs_dir}/streaming",
            run_name="jarvis-streaming-training",
            ddp_find_unused_parameters=False,
            gradient_checkpointing=self.config.gradient_checkpointing,  # Enable for 8GB VRAM
            torch_compile=False,  # Disabled for Windows compatibility
            optim="adamw_torch",
            lr_scheduler_type="linear",
            max_grad_norm=1.0,
            group_by_length=False,
            length_column_name="length",
            ignore_data_skip=False,
            warmup_steps=self.config.warmup_steps,  # Add warmup for AMD stability
        )
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,  # type: ignore
            mlm=False,
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            callbacks=[ProgressBarCallback()]
        )
        
        # Fix label_names warning
        self.trainer.label_names = []
        
        # Start training
        logger.info("Starting streaming training - press Ctrl+C to stop when satisfied")
        try:
            self.trainer.train()
        except KeyboardInterrupt:
            logger.info("Training stopped by user!")
        
        # Save final model
        logger.info("Saving final model...")
        self.trainer.save_model(f"{self.config.output_dir}/streaming_final")
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(f"{self.config.output_dir}/streaming_final")
        
        logger.info("Streaming training completed!")

def check_system_resources(min_ram_gb=8, min_vram_gb=6, min_cpu_cores=2):
    """Check if system resources are sufficient for training, including AMD GPU via DirectML."""
    errors = []
    warnings_list = []
    # Check RAM
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    if ram_gb < min_ram_gb:
        errors.append(f"Insufficient RAM: {ram_gb:.1f}GB detected, {min_ram_gb}GB required.")
    # Check CPU cores
    cpu_cores = psutil.cpu_count(logical=False)
    if cpu_cores is None:
        warnings_list.append("Could not determine number of physical CPU cores.")
    elif cpu_cores < min_cpu_cores:
        errors.append(f"Insufficient CPU cores: {cpu_cores} detected, {min_cpu_cores} required.")
    # Check VRAM (if GPU available)
    vram_gb = None
    try:
        import torch
        import torch_directml
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if vram_gb < min_vram_gb:
                warnings_list.append(f"Low GPU VRAM: {vram_gb:.1f}GB detected, {min_vram_gb}GB recommended.")
        else:
            # Check for DirectML (AMD GPU)
            dml_device = torch_directml.device()
            # Try to allocate a small tensor to test if DirectML is available
            try:
                test_tensor = torch.empty((1, 1), device=dml_device)
                # VRAM detection for DirectML is not straightforward; skip VRAM check
                warnings_list.append("AMD GPU detected via DirectML. Training will use AMD GPU.")
            except Exception:
                warnings_list.append("No compatible GPU detected (CUDA or DirectML). Training will use CPU and may be very slow.")
    except Exception as e:
        warnings_list.append(f"Could not check GPU VRAM: {e}")
    return errors, warnings_list

def main():
    """Main training function."""
    import argparse
    parser = argparse.ArgumentParser(description="J.A.R.VIS Advanced Training Pipeline")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--mode", type=str, default="download-train-save-delete", 
                       choices=["download-train-save-delete", "streaming"], 
                       help="Training mode")
    args = parser.parse_args()
    # Load config
    with open(args.config, 'r') as f:
        config_data = json.load(f)
    # Helper for safe get
    def get(section, key, default=None):
        return config_data.get(section, {}).get(key, default)
    # Build TrainingConfig from config file
    config = TrainingConfig(
        model_name=get('model', 'model_name', 'gpt2'),
        batch_size=get('model', 'batch_size', 8),
        max_length=get('model', 'max_length', 64),
        learning_rate_pretraining=get('model', 'learning_rate_pretraining', 3e-5),
        learning_rate_finetuning=get('model', 'learning_rate_finetuning', 5e-6),
        gradient_accumulation_steps=get('training', 'gradient_accumulation_steps', 4),
        max_steps=get('training', 'max_steps', 500000),
        num_epochs=get('training', 'num_epochs', 1),
        warmup_steps=get('training', 'warmup_steps', 50),
        weight_decay=get('training', 'weight_decay', 0.01),
        chunk_size=get('memory', 'chunk_size', 500),
        max_memory_usage=get('memory', 'max_memory_usage', 1200),
        pin_memory=get('memory', 'pin_memory', True),
        use_lora=get('model', 'lora_enabled', False),
        lora_r=get('lora', 'r', 8),
        lora_alpha=get('lora', 'alpha', 16),
        lora_dropout=get('model', 'lora_dropout', 0.1),
        lora_bias=get('lora', 'bias', 'none'),
        lora_target_modules=tuple(get('lora', 'target_modules', ["c_attn", "c_proj", "c_fc"])),
        output_dir=get('output', 'base_dir', 'models/JARVIS'),
        checkpoint_dir=get('output', 'checkpoint_dir', 'data/checkpoints'),
        logs_dir=get('output', 'logs_dir', 'logs'),
        save_steps=get('training', 'save_steps', 50000),
        eval_steps=get('training', 'eval_steps', 50000),
        save_total_limit=get('training', 'save_total_limit', 1),
        max_samples_per_dataset=None,  # handled per-dataset
        sample_skip_rate=get('training', 'sample_skip_rate', 50),
        fp16=get('training', 'fp16', True),
        gradient_checkpointing=get('training', 'gradient_checkpointing', True),
        dataloader_pin_memory=get('memory', 'pin_memory', True)
    )
    # Dataset configurations with fixes
    dataset_configs = [
        {
            "name": "ShareGPT52K",
            "dataset_id": config_data['datasets']['finetuning']['sharegpt52k'].get('dataset_id', "RyokoAI/ShareGPT52K"),
            "enabled": config_data['datasets']['finetuning']['sharegpt52k']['enabled'],
            "split": "train",
            "max_samples": config_data['datasets']['finetuning']['sharegpt52k'].get('max_samples'),
            "data_files": config_data['datasets']['finetuning']['sharegpt52k'].get('data_files'),
            "conversation_format": config_data['datasets']['finetuning']['sharegpt52k'].get('conversation_format')
        },
        {
            "name": "DialogStudio",
            "dataset_id": "Salesforce/dialogstudio",
            "config_name": "MULTIWOZ2_2",
            "enabled": config_data['datasets']['finetuning']['dialogstudio']['enabled'],
            "split": "train",
            "trust_remote_code": True,
            "max_samples": config_data['datasets']['finetuning']['dialogstudio'].get('max_samples'),
            "conversation_format": config_data['datasets']['finetuning']['dialogstudio'].get('conversation_format')
        },
        {
            "name": "C4",
            "dataset_id": "allenai/c4",
            "config_name": "en",
            "enabled": config_data['datasets']['pretraining']['c4']['enabled'],
            "split": "train",
            "max_samples": config_data['datasets']['pretraining']['c4'].get('max_samples'),
            "filter_language": config_data['datasets']['pretraining']['c4'].get('filter_language')
        },
        {
            "name": "RedPajama",
            "dataset_id": "togethercomputer/RedPajama-Data-1T",
            "config_name": "common_crawl",
            "enabled": config_data['datasets']['pretraining']['redpajama']['enabled'],
            "split": "train",
            "trust_remote_code": True,
            "max_samples": config_data['datasets']['pretraining']['redpajama'].get('max_samples'),
            "filter_language": config_data['datasets']['pretraining']['redpajama'].get('filter_language')
        }
    ]
    
    # System resource check
    resource_errors, resource_warnings = check_system_resources(min_ram_gb=8, min_vram_gb=6, min_cpu_cores=2)
    for warn in resource_warnings:
        print(f"[JARVIS Resource Warning] {warn}")
        logging.warning(warn)
    if resource_errors:
        for err in resource_errors:
            print(f"[JARVIS Resource Check] {err}")
            logging.error(err)
        print("[JARVIS] Please upgrade your hardware or lower batch size/sequence length in the config and try again.")
        sys.exit(1)

    # Initialize trainer
    trainer = AdvancedJARVISTrainer(config)
    
    # Run training based on mode
    if args.mode == "download-train-save-delete":
        trainer.download_train_save_delete(dataset_configs)
    else:
        trainer.train_streaming(dataset_configs)
    
    logger.info("1.5-DAY Training pipeline completed!")

if __name__ == "__main__":
    main() 