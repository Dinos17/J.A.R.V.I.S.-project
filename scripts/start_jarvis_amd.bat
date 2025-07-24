@echo off
echo ========================================
echo J.A.R.V.I.S. AMD Radeon RX 580 Optimized
echo ========================================
echo.
echo System Specifications:
echo - GPU: AMD Radeon RX 580 (8GB VRAM)
echo - CPU: Intel Core i3-8100 @ 3.60GHz
echo - RAM: 16GB DDR4
echo - OS: Windows 10 64-bit
echo.
echo Optimizations Applied:
echo - Batch Size: 8 (Effective: 32 with gradient accumulation)
echo - Sequence Length: 64 tokens
echo - FP16 Mixed Precision: Enabled
echo - Gradient Checkpointing: Enabled
echo - Sample Skip Rate: 50 (process every 50th sample)
echo - LoRA r=8, alpha=16 for optimal adaptation
echo.
echo Estimated Training Time: 8-16 hours
echo Estimated VRAM Usage: 6-7GB
echo.

REM Set AMD-specific environment variables
set OMP_NUM_THREADS=16
set MKL_NUM_THREADS=16
set TOKENIZERS_PARALLELISM=true
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo Starting J.A.R.V.I.S. training with AMD optimizations...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if required packages are installed
echo Checking dependencies...
python -c "import torch, transformers, peft, datasets, wandb" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install transformers peft datasets wandb tqdm numpy requests
)

echo.
echo Starting training with AMD-optimized configuration...
echo Using config: config_amd_rx580.json
echo.

REM Start training with AMD-optimized settings
python src/train.py --config config_amd_rx580.json --mode download-train-save-delete --skip-rate 50

echo.
echo Training completed!
echo Check the models/JARVIS directory for your trained model.
echo.
pause 