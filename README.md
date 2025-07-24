# 🤖 J.A.R.V.I.S. - Just A Rather Very Intelligent System

**AI Training Pipeline for Limited Hardware Resources**

## 🚀 Quick Start

### AMD Radeon RX 580 (Recommended)
```bash
# Double-click scripts/start_jarvis_amd.bat or run:
scripts/start_jarvis_amd.bat
```

### General Usage
```bash
# Install dependencies
pip install -r requirements.txt

# AMD-optimized training
python src/core/train.py --config configs/config_amd_rx580.json

# Monitor training progress
python tools/monitor_amd_training.py

# Start web interface
streamlit run src/core/web_interface.py

# Run inference
python src/core/infer.py
```

## 🎯 Features

- **🔄 Streaming Data Processing** - Handles massive datasets efficiently
- **💾 Memory Optimized** - Designed for 1 vCore, 2GB RAM
- **🎮 LoRA Fine-tuning** - Efficient parameter-efficient training
- **🌐 Web Interface** - Beautiful Streamlit dashboard
- **💬 Interactive Chat** - Real-time conversation with J.A.R.V.I.S.
- **📊 Progress Monitoring** - Real-time training metrics

## ⚙️ Configuration

### AMD Radeon RX 580 (Recommended)
Edit `configs/config_amd_rx580.json` for AMD-optimized settings:
- 8GB VRAM memory management
- AMD-specific optimizations
- Intel i3-8100 CPU tuning
- Balanced speed/quality trade-offs

### General Configuration
Edit `configs/config.json` for default settings:
- Hardware constraints
- Model parameters
- Dataset settings
- Training configuration

## 📁 Project Structure

```
J.A.R.V.I.S. project/
├── 📁 configs/                    # Configuration files
│   ├── config.json               # Default configuration
│   └── config_amd_rx580.json     # AMD-optimized configuration
├── 📁 src/                       # Source code
│   └── 📁 core/                 # Core functionality
│       ├── train.py             # Training pipeline
│       ├── infer.py             # Inference engine
│       ├── web_interface.py     # Web dashboard
│       └── main.py              # Main entry point
├── 📁 scripts/                   # Executable scripts
│   ├── start_jarvis_amd.bat     # AMD-optimized startup
│   └── start.bat                # Default startup
├── 📁 tools/                     # Utility tools
│   └── monitor_amd_training.py  # AMD training monitor
├── 📁 docs/                      # Documentation
│   └── README_AMD_RX580.md      # AMD-specific guide
├── models/                       # Trained models (generated)
├── logs/                         # Training logs (generated)
└── data/                         # Data and checkpoints (generated)
```

📖 **See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed structure documentation.**

## 🖥️ System Requirements

- **Minimum:** 1 vCore, 2GB RAM, 10GB storage
- **Recommended:** 4+ vCores, 8GB+ RAM, 50GB+ storage
- **OS:** Windows 10/11, Linux, macOS

## 🎮 Usage Modes

1. **AMD-Optimized Training** - Complete pipeline optimized for RX 580
2. **Web Interface** - Full dashboard with training control and chat
3. **Real-time Monitoring** - AMD-specific training monitor
4. **Inference Engine** - Run trained models for chat and generation
5. **System Status** - Check hardware and model status

## 📚 Datasets

- **Pretraining:** RedPajama, C4
- **Fine-tuning:** ShareGPT52K, DialogStudio

## 🔧 Troubleshooting

### AMD-Specific Issues
- **GPU Memory Issues:** Use `configs/config_amd_rx580.json` for 8GB VRAM optimization
- **Slow Training:** Monitor with `tools/monitor_amd_training.py`
- **High CPU Usage:** Check Intel i3-8100 optimization settings

### General Issues
- **Memory Issues:** Reduce batch size in configuration files
- **Slow Training:** Enable GPU if available
- **Import Errors:** Run `pip install -r requirements.txt`

📖 **See [docs/README_AMD_RX580.md](docs/README_AMD_RX580.md) for comprehensive AMD troubleshooting.**

---

**"Sometimes you gotta run before you can walk."** - Tony Stark 