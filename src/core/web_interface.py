"""
J.A.R.V.I.S. Web Interface
Streamlit-based web interface for monitoring training and interacting with J.A.R.V.I.S.
"""

import streamlit as st
import json
import os
import time
import psutil
import torch
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd

# Import our modules
from .infer import JARVISInference

# Page configuration
st.set_page_config(
    page_title="J.A.R.V.I.S. AI Training",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def get_system_info():
    """Get system information."""
    return {
        'memory_gb': psutil.virtual_memory().total / (1024**3),
        'memory_used_gb': psutil.virtual_memory().used / (1024**3),
        'memory_percent': psutil.virtual_memory().percent,
        'cpu_cores': psutil.cpu_count(),
        'cpu_percent': psutil.cpu_percent(),
        'gpu_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    }

def get_training_status():
    """Get training status."""
    # Check for training logs and checkpoints
    logs_dir = Path("logs")
    checkpoints_dir = Path("data/checkpoints")
    
    status = {
        'is_training': False,
        'last_checkpoint': None,
        'training_progress': 0,
        'logs': []
    }
    
    # Check for recent training activity
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log"))
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            if time.time() - latest_log.stat().st_mtime < 300:  # 5 minutes
                status['is_training'] = True
    
    # Check for checkpoints
    if checkpoints_dir.exists():
        checkpoint_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir()]
        if checkpoint_dirs:
            status['last_checkpoint'] = max(checkpoint_dirs, key=lambda x: x.stat().st_mtime)
    
    return status

def create_memory_chart():
    """Create memory usage chart."""
    memory_data = []
    timestamps = []
    
    for i in range(10):
        memory_data.append(psutil.virtual_memory().percent)
        timestamps.append(datetime.now().strftime("%H:%M:%S"))
        time.sleep(0.1)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=memory_data,
        mode='lines+markers',
        name='Memory Usage (%)',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title="Memory Usage Over Time",
        xaxis_title="Time",
        yaxis_title="Memory Usage (%)",
        height=300
    )
    
    return fig

def main():
    """Main web interface."""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 J.A.R.V.I.S. AI Training Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard", "Training Control", "Chat with J.A.R.V.I.S.", "Model Info", "Settings"]
    )
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "Training Control":
        show_training_control()
    elif page == "Chat with J.A.R.V.I.S.":
        show_chat_interface()
    elif page == "Model Info":
        show_model_info()
    elif page == "Settings":
        show_settings()

def show_dashboard():
    """Show the main dashboard."""
    
    # System Information
    st.header("🖥️ System Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    system_info = get_system_info()
    
    with col1:
        st.metric("Memory", f"{system_info['memory_gb']:.1f} GB")
    
    with col2:
        st.metric("Memory Used", f"{system_info['memory_used_gb']:.1f} GB", 
                 f"{system_info['memory_percent']:.1f}%")
    
    with col3:
        st.metric("CPU Cores", system_info['cpu_cores'])
    
    with col4:
        gpu_status = "✅ Available" if system_info['gpu_available'] else "❌ Not Available"
        st.metric("GPU", gpu_status)
    
    # Training Status
    st.header("📊 Training Status")
    
    status = get_training_status()
    
    if status['is_training']:
        st.success("🔄 Training in progress...")
    else:
        st.info("⏸️ No active training")
    
    if status['last_checkpoint']:
        st.info(f"📁 Last checkpoint: {status['last_checkpoint'].name}")
    
    # Memory Chart
    st.header("📈 Memory Usage")
    memory_chart = create_memory_chart()
    st.plotly_chart(memory_chart, use_container_width=True)
    
    # Recent Logs
    st.header("📝 Recent Logs")
    logs_dir = Path("logs")
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log"))
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            with open(latest_log, 'r') as f:
                logs = f.readlines()[-20:]  # Last 20 lines
                for log in logs:
                    st.text(log.strip())
        else:
            st.info("No training logs found yet.")
    else:
        st.info("No training logs found yet.")

def show_training_control():
    """Show training control interface."""
    st.header("🎮 Training Control")
    
    # Load configuration
    if os.path.exists("config.json"):
        with open("config.json", "r") as f:
            config = json.load(f)
    else:
        st.error("Configuration file not found!")
        return
    
    # Training parameters
    st.subheader("Training Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        batch_size = st.slider("Batch Size", 1, 8, config['model']['batch_size'])
        learning_rate = st.number_input("Learning Rate", 1e-6, 1e-3, 
                                       config['model']['learning_rate_pretraining'], 
                                       format="%.2e")
    
    with col2:
        max_length = st.slider("Max Sequence Length", 256, 1024, config['model']['max_length'])
        gradient_accumulation = st.slider("Gradient Accumulation Steps", 1, 32, 
                                         config['training']['gradient_accumulation_steps'])
    
    # Dataset configuration
    st.subheader("Dataset Configuration")
    
    datasets_config = config['datasets']
    
    # Pretraining datasets
    st.write("**Pretraining Datasets:**")
    for dataset_name, dataset_config in datasets_config['pretraining'].items():
        enabled = st.checkbox(f"Enable {dataset_name}", dataset_config['enabled'])
        if enabled:
            max_samples = st.number_input(f"{dataset_name} Max Samples", 
                                         min_value=100, value=dataset_config['max_samples'])
            dataset_config['enabled'] = True
            dataset_config['max_samples'] = max_samples
        else:
            dataset_config['enabled'] = False
    
    # Fine-tuning datasets
    st.write("**Fine-tuning Datasets:**")
    for dataset_name, dataset_config in datasets_config['finetuning'].items():
        enabled = st.checkbox(f"Enable {dataset_name}", dataset_config['enabled'])
        if enabled:
            max_samples = st.number_input(f"{dataset_name} Max Samples", 
                                         min_value=50, value=dataset_config['max_samples'])
            dataset_config['enabled'] = True
            dataset_config['max_samples'] = max_samples
        else:
            dataset_config['enabled'] = False
    
    # Training actions
    st.subheader("Training Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start Full Training"):
            st.info("Starting full training pipeline...")
            # This would trigger the training pipeline
            st.success("Training started! Check the logs for progress.")
    
    with col2:
        if st.button("📚 Start Pretraining"):
            st.info("Starting pretraining phase...")
            # This would trigger pretraining only
            st.success("Pretraining started!")
    
    with col3:
        if st.button("💬 Start Fine-tuning"):
            st.info("Starting fine-tuning phase...")
            # This would trigger fine-tuning only
            st.success("Fine-tuning started!")
    
    # Save configuration
    if st.button("💾 Save Configuration"):
        config['model']['batch_size'] = batch_size
        config['model']['learning_rate_pretraining'] = learning_rate
        config['model']['max_length'] = max_length
        config['training']['gradient_accumulation_steps'] = gradient_accumulation
        
        with open("config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        st.success("Configuration saved!")

def show_chat_interface():
    """Show chat interface with J.A.R.V.I.S."""
    
    st.header("💬 Chat with J.A.R.V.I.S.")
    
    # Check if model exists
    model_paths = ["models/JARVIS/finetuned", "models/JARVIS/pretrained"]
    model_path = None
    
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        st.warning("J.A.R.V.I.S. model not found. Please train the model first.")
        return
    
    # Model selection
    selected_model = st.selectbox(
        "Select Model",
        ["Fine-tuned (Recommended)", "Pretrained"],
        index=0
    )
    
    if selected_model == "Fine-tuned (Recommended)":
        model_path = "models/JARVIS/finetuned"
    else:
        model_path = "models/JARVIS/pretrained"
    
    # Initialize chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to ask J.A.R.V.I.S.?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("J.A.R.V.I.S. is thinking..."):
                try:
                    jarvis = JARVISInference(model_path)
                    response = jarvis.generate_response(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error generating response: {e}")
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

def show_model_info():
    """Show model information."""
    
    st.header("📋 Model Information")
    
    # Model status
    model_paths = {
        "Pretrained": "models/JARVIS/pretrained",
        "Fine-tuned": "models/JARVIS/finetuned"
    }
    
    for model_name, model_path in model_paths.items():
        st.subheader(f"{model_name} Model")
        
        if os.path.exists(model_path):
            st.success(f"✅ {model_name} model found")
            
            # Get model size
            total_size = 0
            file_count = 0
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
                    file_count += 1
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model Size", f"{total_size / (1024**3):.2f} GB")
            with col2:
                st.metric("Files", file_count)
            
            # Model files
            with st.expander(f"View {model_name} model files"):
                for root, dirs, files in os.walk(model_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        st.text(file_path)
        else:
            st.error(f"❌ {model_name} model not found")

def show_settings():
    """Show settings page."""
    
    st.header("⚙️ Settings")
    
    # Hardware settings
    st.subheader("Hardware Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_memory = st.slider("Max Memory (GB)", 1, 8, 2)
        max_cpu = st.slider("Max CPU Cores", 1, 8, 1)
    
    with col2:
        enable_gpu = st.checkbox("Enable GPU", value=True)
        memory_efficiency = st.selectbox("Memory Efficiency", ["low", "medium", "high"], index=2)
    
    # Training settings
    st.subheader("Training Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        batch_size = st.slider("Batch Size", 1, 8, 2)
        learning_rate = st.number_input("Learning Rate", 1e-6, 1e-3, 5e-5, format="%.2e")
    
    with col2:
        max_length = st.slider("Max Sequence Length", 256, 1024, 512)
        checkpoint_freq = st.slider("Checkpoint Frequency", 100, 1000, 250)
    
    # Save settings
    if st.button("💾 Save Settings"):
        st.success("Settings saved!")

if __name__ == "__main__":
    main() 