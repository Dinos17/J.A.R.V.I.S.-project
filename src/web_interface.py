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
from train_jarvis import JARVISTrainingPipeline
from infer import JARVISInference

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
    try:
        pipeline = JARVISTrainingPipeline()
        return pipeline.get_training_status()
    except Exception as e:
        st.error(f"Error getting training status: {e}")
        return None

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
    if status:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Training Phases")
            for phase_name, phase_info in status['phases'].items():
                status_icon = "✅" if phase_info['completed'] else "⏳"
                st.markdown(f"{status_icon} **{phase_name.title()}**: {phase_info['description']}")
        
        with col2:
            st.subheader("Dataset Status")
            for dataset_type, datasets in status['datasets'].items():
                st.markdown(f"**{dataset_type.title()}:**")
                for dataset_name, dataset_info in datasets.items():
                    file_count = dataset_info.get('files', 0)
                    st.markdown(f"  - {dataset_name}: {file_count} files")
    
    # Memory Chart
    st.header("📈 Memory Usage")
    memory_chart = create_memory_chart()
    st.plotly_chart(memory_chart, use_container_width=True)
    
    # Recent Logs
    st.header("📝 Recent Logs")
    if os.path.exists('jarvis_training.log'):
        with open('jarvis_training.log', 'r') as f:
            logs = f.readlines()[-20:]  # Last 20 lines
            for log in logs:
                st.text(log.strip())
    else:
        st.info("No training logs found yet.")

def show_training_control():
    """Show training control interface."""
    
    st.header("🎮 Training Control")
    
    # Initialize pipeline
    try:
        pipeline = JARVISTrainingPipeline()
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {e}")
        return
    
    # Training options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Start Training")
        
        training_mode = st.selectbox(
            "Training Mode",
            ["Complete Pipeline", "Pretraining Only", "Fine-tuning Only"]
        )
        
        if st.button("🚀 Start Training", type="primary"):
            with st.spinner("Starting training..."):
                try:
                    if training_mode == "Complete Pipeline":
                        success = pipeline.run_complete_pipeline()
                    elif training_mode == "Pretraining Only":
                        success = pipeline.train_phase('pretraining')
                    elif training_mode == "Fine-tuning Only":
                        success = pipeline.train_phase('finetuning')
                    
                    if success:
                        st.success("Training started successfully!")
                    else:
                        st.error("Training failed to start.")
                except Exception as e:
                    st.error(f"Training error: {e}")
    
    with col2:
        st.subheader("Training Status")
        
        if st.button("🔄 Refresh Status"):
            st.rerun()
        
        status = pipeline.get_training_status()
        if status:
            for phase_name, phase_info in status['phases'].items():
                status_text = "Completed" if phase_info['completed'] else "Not Started"
                status_color = "green" if phase_info['completed'] else "red"
                st.markdown(f"**{phase_name.title()}**: :{status_color}[{status_text}]")
    
    # Configuration
    st.subheader("⚙️ Configuration")
    
    if os.path.exists('config.json'):
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        # Display current configuration
        st.json(config)
        
        # Allow configuration editing
        if st.button("Edit Configuration"):
            st.text_area("Edit JSON Configuration", json.dumps(config, indent=2))
            if st.button("Save Configuration"):
                st.success("Configuration saved!")

def show_chat_interface():
    """Show chat interface with J.A.R.V.I.S."""
    
    st.header("💬 Chat with J.A.R.V.I.S.")
    
    # Check if model exists
    model_path = "models/JARVIS/finetuned"
    if not os.path.exists(model_path):
        st.warning("J.A.R.V.I.S. model not found. Please train the model first.")
        return
    
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