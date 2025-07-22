#!/usr/bin/env python3
"""
J.A.R.V.I.S. - Just A Rather Very Intelligent System
Main entry point for the AI training pipeline.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def print_banner():
    """Print J.A.R.V.I.S. banner."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║    🤖 J.A.R.V.I.S. - Just A Rather Very Intelligent System  ║
    ║                                                              ║
    ║    AI Training Pipeline for Limited Hardware Resources       ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'torch', 'transformers', 'datasets', 'peft', 'accelerate',
        'streamlit', 'plotly', 'psutil'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed!")
    return True

def setup_environment():
    """Setup the environment for J.A.R.V.I.S."""
    print("🔧 Setting up J.A.R.V.I.S. environment...")
    
    # Create necessary directories
    directories = [
        "data/pretraining",
        "data/finetuning", 
        "data/processed",
        "data/checkpoints",
        "models/JARVIS",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    print("✅ Environment setup complete!")

def start_web_interface():
    """Start the Streamlit web interface."""
    print("🌐 Starting J.A.R.V.I.S. Web Interface...")
    print("📱 Open your browser to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/web_interface.py", "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Web interface stopped.")

def start_training():
    """Start the training pipeline."""
    print("🚀 Starting J.A.R.V.I.S. Training Pipeline...")
    
    try:
        from src.core.train import AdvancedJARVISTrainer, TrainingConfig
        # Minimal config for demonstration; in production, load from config file
        config = TrainingConfig()
        # Minimal dataset_configs; in production, load from config file
        dataset_configs = [
            {
                "name": "ShareGPT52K",
                "dataset_id": "RyokoAI/ShareGPT52K",
                "enabled": True,
                "split": "train"
            }
        ]
        trainer = AdvancedJARVISTrainer(config)
        trainer.download_train_save_delete(dataset_configs)
    except KeyboardInterrupt:
        print("\n🛑 Training stopped.")
    except Exception as e:
        print(f"❌ Training failed: {e}")

def start_chat():
    """Start interactive chat with J.A.R.V.I.S."""
    print("💬 Starting J.A.R.V.I.S. Chat Interface...")
    
    try:
        subprocess.run([sys.executable, "src/infer.py", "--interactive"])
    except KeyboardInterrupt:
        print("\n🛑 Chat stopped.")

def show_status():
    """Show training and system status."""
    print("📊 J.A.R.V.I.S. Status Report")
    print("=" * 50)
    
    # Check model status
    model_paths = {
        "Pretrained": "models/JARVIS/pretrained",
        "Fine-tuned": "models/JARVIS/finetuned"
    }
    
    for model_name, model_path in model_paths.items():
        if os.path.exists(model_path):
            print(f"✅ {model_name} model: Found")
            
            # Get model size
            total_size = 0
            file_count = 0
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
                    file_count += 1
            
            print(f"   Size: {total_size / (1024**3):.2f} GB")
            print(f"   Files: {file_count}")
        else:
            print(f"❌ {model_name} model: Not found")
    
    # Check system resources
    import psutil
    memory = psutil.virtual_memory()
    print(f"\n💾 Memory: {memory.total / (1024**3):.1f} GB total, {memory.percent:.1f}% used")
    print(f"🖥️  CPU: {psutil.cpu_count()} cores, {psutil.cpu_percent():.1f}% usage")
    
    # Check GPU
    import torch
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("🎮 GPU: Not available")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S. AI Training System")
    parser.add_argument("--mode", choices=["web", "train", "chat", "status", "setup"], 
                       default="web", help="Mode to run J.A.R.V.I.S. in")
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies")
    parser.add_argument("--no-banner", action="store_true", help="Skip banner display")
    
    args = parser.parse_args()
    
    if not args.no_banner:
        print_banner()
    
    # Check dependencies if requested
    if args.check_deps:
        if not check_dependencies():
            sys.exit(1)
        return
    
    # Check dependencies before running
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first.")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Setup environment if needed
    if args.mode == "setup":
        setup_environment()
        return
    
    # Run selected mode
    if args.mode == "web":
        start_web_interface()
    elif args.mode == "train":
        start_training()
    elif args.mode == "chat":
        start_chat()
    elif args.mode == "status":
        show_status()

if __name__ == "__main__":
    main() 