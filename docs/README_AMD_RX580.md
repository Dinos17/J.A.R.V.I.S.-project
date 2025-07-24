# J.A.R.V.I.S. AI Training - AMD Radeon RX 580 Optimized

## System Specifications
- **GPU**: AMD Radeon RX 580 (8GB VRAM)
- **CPU**: Intel Core i3-8100 @ 3.60GHz
- **RAM**: 16GB DDR4
- **OS**: Windows 10 64-bit

## Quick Start

### 1. Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft datasets wandb tqdm numpy requests psutil
```

### 2. Start Training (Recommended)
Double-click `start_jarvis_amd.bat` or run:
```bash
python src/train.py --config config_amd_rx580.json --mode download-train-save-delete
```

### 3. Monitor Progress
```bash
python monitor_amd_training.py
```

## AMD-Specific Optimizations

### Memory Management
- **Batch Size**: 8 (Effective: 32 with gradient accumulation)
- **Sequence Length**: 64 tokens
- **GPU Memory Usage**: 6-7GB (85% of 8GB VRAM)
- **Gradient Checkpointing**: Enabled for memory efficiency

### Performance Optimizations
- **FP16 Mixed Precision**: Enabled for AMD GPU
- **Flash Attention**: Enabled when available
- **Sample Skip Rate**: 50 (process every 50th sample)
- **LoRA Configuration**: r=8, alpha=16 for optimal adaptation

### CPU Optimizations
- **Thread Count**: 16 threads for Intel i3-8100
- **DataLoader Workers**: 4 workers optimized for quad-core CPU
- **Memory Pinning**: Enabled for faster data transfer

## Expected Performance

### Training Time
- **Per Dataset**: 2-4 hours
- **Total Training**: 8-16 hours
- **Sample Processing**: ~50 samples per second

### Resource Usage
- **GPU Memory**: 6-7GB VRAM
- **CPU Usage**: 60-80%
- **RAM Usage**: 10-12GB
- **GPU Utilization**: 85-95%

## Configuration Files

### `config_amd_rx580.json`
AMD-optimized configuration with:
- Memory-efficient settings for 8GB VRAM
- CPU-optimized parameters for Intel i3-8100
- Balanced speed/quality trade-offs

### `start_jarvis_amd.bat`
Windows batch script that:
- Sets AMD-specific environment variables
- Checks dependencies
- Starts training with optimal settings

## Monitoring and Troubleshooting

### Real-time Monitoring
```bash
# Monitor every 30 seconds
python monitor_amd_training.py

# Monitor every 60 seconds
python monitor_amd_training.py --interval 60

# Check status once
python monitor_amd_training.py --once
```

### Common Issues

#### Out of Memory (OOM)
If you encounter GPU memory errors:
1. Reduce batch size: `--batch-size 4`
2. Reduce sequence length: `--max-length 32`
3. Increase sample skip rate: `--skip-rate 100`

#### Slow Training
If training is too slow:
1. Increase sample skip rate: `--skip-rate 100`
2. Reduce sequence length: `--max-length 32`
3. Disable gradient checkpointing in config

#### High CPU Usage
If CPU usage is too high:
1. Reduce dataloader workers in config
2. Close other applications
3. Monitor with Task Manager

## Advanced Configuration

### Custom Batch Size
```bash
python src/train.py --config config_amd_rx580.json --batch-size 4
```

### Custom Sequence Length
```bash
python src/train.py --config config_amd_rx580.json --max-length 128
```

### Custom Sample Skip Rate
```bash
python src/train.py --config config_amd_rx580.json --skip-rate 25
```

## Training Modes

### 1. Download-Train-Save-Delete (Recommended)
Processes each dataset completely, trains, saves checkpoint, then moves to next dataset.
```bash
python src/train.py --config config_amd_rx580.json --mode download-train-save-delete
```

### 2. Streaming
Continuous training across all datasets (use Ctrl+C to stop).
```bash
python src/train.py --config config_amd_rx580.json --mode streaming
```

## Output Structure
```
models/JARVIS/
├── checkpoint_ShareGPT52K/
├── checkpoint_DialogStudio/
├── checkpoint_C4/
├── checkpoint_RedPajama/
└── final_model/

logs/
├── training_ShareGPT52K/
├── training_DialogStudio/
├── training_C4/
├── training_RedPajama/
└── amd_monitor.log
```

## Performance Tips

### Before Training
1. Close unnecessary applications
2. Disable Windows updates temporarily
3. Ensure adequate free disk space (50GB+)
4. Check GPU drivers are up to date

### During Training
1. Monitor with `monitor_amd_training.py`
2. Check GPU temperature (should be <85°C)
3. Monitor system stability
4. Don't interrupt training unless necessary

### After Training
1. Test the trained model
2. Backup checkpoints
3. Clean up temporary files
4. Restart system to clear memory

## Troubleshooting

### GPU Not Detected
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU info
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Memory Issues
```bash
# Check available memory
python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory / 1e9)"
```

### Slow Performance
1. Check if other applications are using GPU
2. Monitor CPU and RAM usage
3. Consider reducing batch size or sequence length
4. Check disk I/O performance

## Support

For AMD-specific issues:
1. Check AMD GPU drivers are latest
2. Ensure ROCm or CUDA compatibility
3. Monitor GPU temperature and fan speed
4. Consider underclocking if overheating

## License
This project is optimized for AMD Radeon RX 580 systems. Use at your own risk and monitor system resources during training. 