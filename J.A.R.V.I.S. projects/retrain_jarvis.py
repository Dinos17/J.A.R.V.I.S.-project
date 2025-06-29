#!/usr/bin/env python3
"""
J.A.R.V.I.S. Retrain Script
Retrains the model with fixed conversation format for better responses.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.train_jarvis import JARVISTrainer, TrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Retrain J.A.R.VIS with conversation format fixes."""
    print("🔄 Retraining J.A.R.VIS with conversation format fixes...")
    print("This will improve response quality and coherence.")
    print()
    
    # Configuration optimized for conversation training
    config = TrainingConfig()
    config.learning_rate = 2e-5  # Lower learning rate for better stability
    config.num_epochs = 1  # Only 1 epoch for speed
    config.max_length = 512  # Longer context for better conversations
    config.batch_size = 2  # Increase batch size if RAM allows
    config.gradient_accumulation_steps = 8  # Effective batch size of 8
    
    # FAST TRAINING SETTINGS FOR SPEED
    config.lora_r = 4      # Lower LoRA rank for speed
    config.lora_alpha = 8  # Lower LoRA alpha for speed
    config.lora_dropout = 0.05  # Lower dropout for stability
    config.save_steps = 10000  # Save less frequently
    config.eval_steps = 10000  # Evaluate less frequently
    
    # Initialize trainer
    trainer = JARVISTrainer(config)
    
    # Conversation training paths - focus on high-quality conversation data
    conversation_data_paths = [
        "data/processed/sharegpt52k_chunk_*.jsonl",
        "data/processed/dialogstudio_chunk_*.jsonl"
    ]
    
    # Check if conversation data exists
    data_exists = False
    for path_pattern in conversation_data_paths:
        if list(Path("data/processed").glob(path_pattern.replace("data/processed/", ""))):
            data_exists = True
            break
    
    if not data_exists:
        print("❌ Conversation training data not found!")
        print("Please run the data preprocessing first:")
        print("python start_jarvis.py --mode download")
        return
    
    # Start retraining
    try:
        print("🚀 Starting conversation training...")
        trainer.train_conversations(conversation_data_paths)
        print("✅ J.A.R.VIS retraining completed successfully!")
        print("🎉 The model should now produce much better responses!")
        
    except Exception as e:
        print(f"❌ Retraining failed: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return

if __name__ == "__main__":
    main() 