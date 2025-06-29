#!/usr/bin/env python3
"""
J.A.R.V.I.S. Dataset Download Script
Downloads and processes all required datasets from Hugging Face.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.preprocess import StreamingDataProcessor, DatasetManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Download and process all datasets."""
    print("🚀 Starting J.A.R.V.I.S. Dataset Download and Processing...")
    print("This will download datasets from Hugging Face and process them for training.")
    print()
    
    # Create necessary directories
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    # Initialize processor and manager
    processor = StreamingDataProcessor()
    dataset_manager = DatasetManager()
    
    try:
        print("📥 Downloading and processing datasets...")
        print()
        
        # Download all datasets
        dataset_manager.download_all_datasets(processor)
        
        print()
        print("✅ All datasets downloaded and processed successfully!")
        print("🎉 You can now run training with: python src/train_jarvis.py --phase all_data")
        
    except Exception as e:
        print(f"❌ Dataset download failed: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return

if __name__ == "__main__":
    main() 