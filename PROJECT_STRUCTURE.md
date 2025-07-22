J.A.R.V.I.S. project/
├───ORGANIZATION_SUMMARY.md                       # 🗂️ High-level folder breakdown for contributors or users
├───PROJECT_STRUCTURE.md                          # 🧠 Technical architecture and design philosophy
├───README.md                                     # 📖 Entry point — explains setup, usage, and purpose
├───requirements.txt                              # 📦 Python dependencies for the entire project
│
├───configs
│   └───config_amd_rx580.json                     # ⚙️ Hardware-specific config for AMD RX 580 GPUs (batch size, precision, etc.)
├───data
│   ├───checkpoints                               # 💾 Saved model weights from training/evaluation runs
│   ├───processed                                 # 🧹 Cleaned and pre-tokenized datasets
│   └───raw                                       # 📥 Original input datasets (unprocessed)
├───docs
│   └───README_AMD_RX580.md                       # 📘 User manual for AMD RX580 support — driver/install/setup notes
├───models
│   └───JARVIS                                    # 🤖 Main model directory, housing fine-tuned versions
│       ├───c4                                    # 📚 Fine-tuning dataset based on Common Crawl (C4)
│       ├───dialogstudio                          # 🗨️ Dialogue-focused model training data
│       ├───redpajama                             # 🔴 Open pretraining dataset (RedPajama-compatible)
│       └───sharegpt52k                           # 💬 Final conversational fine-tune using ShareGPT
├───scripts
│   ├───start.bat                                 # ▶️ Default startup script (for CLI/local inference)
│   └───start_jarvis_amd.bat                      # 🚀 AMD RX580-optimized launch script (uses special config/hardware flags)
├───src
│   ├───__init__.py                               # 🧬 Marks the src directory as a Python package
│   └───core
│       ├───infer.py                              # 🧠 Handles inference logic — prompts in, responses out
│       ├───main.py                               # 🎯 Main entry script — ties everything together
│       ├───train.py                              # 📈 Training loop and data/model integration
│       ├───web_interface.py                      # 🌐 Starts API or web dashboard for interaction
│       ├───__init__.py                           # 📦 Package initializer
│       ├───inference                             # 🔍 Specialized inference methods or engines
│       ├───training                              # 🏋️ Training modules (dataloaders, loops, optimizers)
│       ├───utils                                 # 🧰 Utility scripts (logging, configuration parsing, metrics)
│       └───web                                   # 🌍 Web UI or backend logic (Flask, FastAPI, etc.)
├───tools
│   ├───clean_sharegpt52k.py                      # 🧼 Cleans and filters ShareGPT data before training
│   └───monitor_amd_training.py                   # 📊 Live training monitor — tuned for AMD RX580

``` 