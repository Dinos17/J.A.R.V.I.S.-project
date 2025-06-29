# 🤖 J.A.R.V.I.S. - AI Training Pipeline

**Just A Rather Very Intelligent System** - A comprehensive AI training pipeline designed for limited hardware resources (1 vCore, 2GB RAM) that trains a conversational AI model from scratch using streaming data processing.

## 🎯 Overview

J.A.R.V.I.S. is trained in two phases:
1. **Pretraining**: Massive text datasets (RedPajama-1T, C4) for general language understanding
2. **Fine-tuning**: Conversational datasets (ShareGPT52K, DialogStudio) for chat capabilities

## ✨ Key Features

- **🔄 Streaming Data Processing**: Handles massive datasets without loading them entirely into memory
- **💾 Memory Efficient**: Optimized for 1 vCore, 2GB RAM hardware constraints
- **📊 Incremental Training**: Supports checkpointing and resuming training
- **🎮 Web Interface**: Beautiful Streamlit dashboard for monitoring and control
- **💬 Interactive Chat**: Real-time conversation with trained J.A.R.V.I.S.
- **🔧 LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning
- **📈 Progress Monitoring**: Real-time training metrics and visualization

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd J.A.R.V.I.S.-projects

# Install dependencies
pip install -r requirements.txt

# Setup environment
python start_jarvis.py --mode setup
```

### 2. Start the Web Interface

```bash
python start_jarvis.py --mode web
```

Open your browser to `http://localhost:8501` to access the J.A.R.V.I.S. dashboard.

### 3. Start Training

```bash
# Complete pipeline (recommended)
python start_jarvis.py --mode train

# Or use the training script directly
python src/train_jarvis.py
```

### 4. Chat with J.A.R.V.I.S.

```bash
# After training is complete
python start_jarvis.py --mode chat
```

## 📁 Project Structure

```
J.A.R.V.I.S. projects/
├── src/
│   ├── data/
│   │   └── preprocess.py          # Data preprocessing and streaming
│   ├── train.py                   # Core training pipeline
│   ├── train_jarvis.py           # Complete training orchestration
│   ├── infer.py                   # Inference and chat interface
│   └── web_interface.py          # Streamlit web dashboard
├── data/
│   ├── pretraining/              # Raw pretraining datasets
│   ├── finetuning/               # Raw fine-tuning datasets
│   ├── processed/                # Processed streaming datasets
│   └── checkpoints/              # Training checkpoints
├── models/
│   └── J.A.R.V.I.S/             # Trained models
├── config.json                   # Training configuration
├── requirements.txt              # Python dependencies
├── start_jarvis.py              # Easy startup script
└── README.md                    # This file
```

## ⚙️ Configuration

The training pipeline is configured via `config.json`:

```json
{
  "hardware_constraints": {
    "max_memory_gb": 2,
    "max_cpu_cores": 1,
    "enable_gpu": true
  },
  "model": {
    "base_model": "microsoft/DialoGPT-medium",
    "batch_size": 2,
    "learning_rate_pretraining": 5e-5,
    "learning_rate_finetuning": 1e-5
  },
  "datasets": {
    "pretraining": {
      "redpajama": {"enabled": true, "max_samples": 100000},
      "c4": {"enabled": true, "max_samples": 50000}
    },
    "finetuning": {
      "sharegpt52k": {"enabled": true, "max_samples": 10000},
      "dialogstudio": {"enabled": true, "max_samples": 5000}
    }
  }
}
```

## 🎮 Usage Modes

### Web Interface (Recommended)
```bash
python start_jarvis.py --mode web
```
- **Dashboard**: System monitoring and training status
- **Training Control**: Start/stop training phases
- **Chat Interface**: Interactive conversation with J.A.R.V.I.S.
- **Model Info**: View model details and performance
- **Settings**: Configure training parameters

### Command Line Training
```bash
# Complete pipeline
python src/train_jarvis.py

# Specific phases
python src/train_jarvis.py --phase pretraining
python src/train_jarvis.py --phase finetuning

# Check status
python src/train_jarvis.py --status

# Resume from checkpoint
python src/train_jarvis.py --resume path/to/checkpoint
```

### Interactive Chat
```bash
python start_jarvis.py --mode chat
# or
python src/infer.py --interactive
```

## 📊 Training Process

### Phase 1: Pretraining
- **Datasets**: RedPajama-1T, C4
- **Purpose**: General language understanding
- **Duration**: ~3-5 hours (depending on hardware)
- **Output**: `models/J.A.R.V.I.S/pretrained/`

### Phase 2: Fine-tuning
- **Datasets**: ShareGPT52K, DialogStudio
- **Purpose**: Conversational capabilities
- **Duration**: ~1-2 hours
- **Output**: `models/J.A.R.V.I.S/finetuned/`

## 🔧 Hardware Requirements

### Minimum (Tested)
- **CPU**: 1 vCore
- **RAM**: 2GB
- **Storage**: 10GB free space
- **OS**: Windows 10/11, Linux, macOS

### Recommended
- **CPU**: 4+ vCores
- **RAM**: 8GB+
- **GPU**: NVIDIA GPU with 4GB+ VRAM
- **Storage**: 50GB+ free space

## 📈 Monitoring

### Web Dashboard
- Real-time system metrics
- Training progress visualization
- Memory usage charts
- Log monitoring

### Log Files
- `jarvis_training.log`: Training logs
- `logs/`: Detailed training metrics

### Checkpoints
- Automatic checkpointing every 250 steps
- Resume training from any checkpoint
- Model versioning and rollback

## 🛠️ Troubleshooting

### Common Issues

**1. Out of Memory Error**
```bash
# Reduce batch size in config.json
"batch_size": 1
# Increase gradient accumulation
"gradient_accumulation_steps": 16
```

**2. Slow Training**
```bash
# Enable GPU acceleration
# Reduce dataset size in config.json
"max_samples": 10000
```

**3. Missing Dependencies**
```bash
python start_jarvis.py --check-deps
pip install -r requirements.txt
```

### Performance Optimization

1. **Memory Management**:
   - Use smaller batch sizes
   - Enable gradient checkpointing
   - Use LoRA for efficient fine-tuning

2. **Data Processing**:
   - Reduce chunk sizes for streaming
   - Filter short texts
   - Use compression for datasets

3. **Training**:
   - Use mixed precision (fp16)
   - Enable gradient accumulation
   - Monitor memory usage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face**: Transformers library and model hub
- **Microsoft**: DialoGPT base model
- **RedPajama**: Open-source language model dataset
- **ShareGPT**: Conversational dataset
- **DialogStudio**: Multi-domain dialogue dataset

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `jarvis_training.log`
3. Open an issue on GitHub

---

**"Sometimes you gotta run before you can walk."** - Tony Stark

*J.A.R.V.I.S. is ready to assist you in creating your own AI!* 🤖✨ 