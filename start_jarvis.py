#!/usr/bin/env python3
"""
J.A.R.V.I.S. Startup Script
Provides easy access to all J.A.R.V.I.S. functionality.
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
        'bitsandbytes', 'streamlit', 'plotly', 'psutil'
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
        "data/pretraining/redpajama",
        "data/pretraining/c4",
        "data/finetuning/sharegpt52k",
        "data/finetuning/dialogstudio",
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
        subprocess.run([sys.executable, "src/train_jarvis.py"])
    except KeyboardInterrupt:
        print("\n🛑 Training stopped.")

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
    
    try:
        subprocess.run([sys.executable, "src/train_jarvis.py", "--status"])
    except Exception as e:
        print(f"❌ Error getting status: {e}")

def start_dataset_download():
    """Start the dataset download process."""
    print("📥 Starting J.A.R.V.I.S. Dataset Download...")
    print("This will download datasets from Hugging Face:")
    print("- RedPajama-Data-1T (100K samples)")
    print("- C4 (50K samples)")
    print("- ShareGPT52K (10K samples)")
    print("- DialogStudio (5K samples)")
    print()
    
    try:
        subprocess.run([sys.executable, "download_datasets.py"])
    except KeyboardInterrupt:
        print("\n🛑 Dataset download stopped.")

def start_retraining():
    """Start the retraining pipeline with conversation format fixes."""
    print("🔄 Starting J.A.R.V.I.S. Retraining with Conversation Fixes...")
    print("This will improve response quality and coherence.")
    
    try:
        subprocess.run([sys.executable, "retrain_jarvis.py"])
    except KeyboardInterrupt:
        print("\n🛑 Retraining stopped.")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S. Startup Script")
    parser.add_argument("--mode", choices=["web", "train", "chat", "status", "setup", "download", "retrain"], 
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
    elif args.mode == "download":
        start_dataset_download()
    elif args.mode == "retrain":
        start_retraining()

if __name__ == "__main__":
    main() 