#!/usr/bin/env python3
"""
J.A.R.V.I.S. AMD Training Monitor
Monitors training progress and system resources for AMD Radeon RX 580.
"""

import os
import time
import psutil
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

try:
    import torch
    import GPUtil
except ImportError:
    print("Warning: torch or GPUtil not available. GPU monitoring disabled.")
    torch = None
    GPUtil = None

class AMDTrainingMonitor:
    """Monitor training progress and system resources for AMD GPU."""
    
    def __init__(self, log_dir="logs", config_file="config_amd_rx580.json"):
        self.log_dir = Path(log_dir)
        self.config_file = config_file
        self.start_time = datetime.now()
        self.training_logs = []
        
        # Setup logging
        self.log_dir.mkdir(exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "amd_monitor.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Load config
        self.config = self.load_config()
        
    def load_config(self):
        """Load AMD-specific configuration."""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file {self.config_file} not found. Using defaults.")
            return {
                "system_info": {"gpu": "AMD Radeon RX 580 (8GB VRAM)"},
                "performance_estimates": {"total_training_time": "8-16 hours"}
            }
    
    def get_system_info(self):
        """Get current system information."""
        info = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / (1024**3),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        }
        
        # GPU information (AMD specific)
        if torch and torch.cuda.is_available():
            try:
                gpu = torch.cuda.get_device_properties(0)
                info.update({
                    "gpu_name": gpu.name,
                    "gpu_memory_used_mb": torch.cuda.memory_allocated(0) / (1024**2),
                    "gpu_memory_total_mb": gpu.total_memory / (1024**2),
                    "gpu_memory_percent": (torch.cuda.memory_allocated(0) / gpu.total_memory) * 100,
                    "gpu_utilization": torch.cuda.utilization(0) if hasattr(torch.cuda, 'utilization') else "N/A"
                })
            except Exception as e:
                self.logger.warning(f"Could not get GPU info: {e}")
                info.update({
                    "gpu_name": "AMD GPU (info unavailable)",
                    "gpu_memory_used_mb": "N/A",
                    "gpu_memory_total_mb": "N/A",
                    "gpu_memory_percent": "N/A",
                    "gpu_utilization": "N/A"
                })
        else:
            info.update({
                "gpu_name": "No GPU detected",
                "gpu_memory_used_mb": "N/A",
                "gpu_memory_total_mb": "N/A",
                "gpu_memory_percent": "N/A",
                "gpu_utilization": "N/A"
            })
        
        return info
    
    def check_training_progress(self):
        """Check training progress from log files."""
        progress = {
            "training_logs": [],
            "latest_checkpoint": None,
            "estimated_completion": None
        }
        
        # Check for training logs
        log_files = list(self.log_dir.glob("*.log"))
        for log_file in log_files:
            if "training" in log_file.name.lower():
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            progress["training_logs"].append({
                                "file": log_file.name,
                                "last_line": lines[-1].strip(),
                                "line_count": len(lines)
                            })
                except Exception as e:
                    self.logger.warning(f"Could not read log file {log_file}: {e}")
        
        # Check for model checkpoints
        model_dir = Path("models/JARVIS")
        if model_dir.exists():
            checkpoints = list(model_dir.glob("**/checkpoint-*"))
            if checkpoints:
                latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
                progress["latest_checkpoint"] = {
                    "path": str(latest),
                    "modified": datetime.fromtimestamp(latest.stat().st_mtime).isoformat()
                }
        
        return progress
    
    def estimate_completion_time(self):
        """Estimate completion time based on current progress."""
        if not self.config.get("performance_estimates"):
            return "Unknown"
        
        elapsed = datetime.now() - self.start_time
        estimated_total = self.config["performance_estimates"].get("total_training_time", "8-16 hours")
        
        # Simple estimation (this could be improved with actual progress tracking)
        return f"Elapsed: {elapsed}, Estimated Total: {estimated_total}"
    
    def print_status(self):
        """Print current system status."""
        system_info = self.get_system_info()
        progress = self.check_training_progress()
        
        print("\n" + "="*60)
        print("J.A.R.V.I.S. AMD Training Monitor")
        print("="*60)
        print(f"Timestamp: {system_info['timestamp']}")
        print(f"Elapsed Time: {datetime.now() - self.start_time}")
        print()
        
        print("System Resources:")
        print(f"  CPU Usage: {system_info['cpu_percent']:.1f}%")
        print(f"  RAM Usage: {system_info['memory_used_gb']:.1f}GB / {system_info['memory_total_gb']:.1f}GB ({system_info['memory_percent']:.1f}%)")
        print(f"  GPU: {system_info['gpu_name']}")
        
        if system_info['gpu_memory_percent'] != "N/A":
            print(f"  GPU Memory: {system_info['gpu_memory_used_mb']:.1f}MB / {system_info['gpu_memory_total_mb']:.1f}MB ({system_info['gpu_memory_percent']:.1f}%)")
            if system_info['gpu_utilization'] != "N/A":
                print(f"  GPU Utilization: {system_info['gpu_utilization']}%")
        
        print()
        print("Training Progress:")
        if progress["latest_checkpoint"]:
            print(f"  Latest Checkpoint: {progress['latest_checkpoint']['path']}")
            print(f"  Checkpoint Time: {progress['latest_checkpoint']['modified']}")
        
        if progress["training_logs"]:
            print(f"  Active Log Files: {len(progress['training_logs'])}")
            for log in progress["training_logs"][-3:]:  # Show last 3 logs
                print(f"    {log['file']}: {log['line_count']} lines")
        
        print()
        print("Performance Estimates:")
        print(f"  {self.estimate_completion_time()}")
        print("="*60)
    
    def monitor_continuously(self, interval=30):
        """Monitor continuously with specified interval."""
        self.logger.info("Starting continuous monitoring...")
        print(f"Monitoring every {interval} seconds. Press Ctrl+C to stop.")
        
        try:
            while True:
                self.print_status()
                time.sleep(interval)
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user.")
            print("\nMonitoring stopped.")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S. AMD Training Monitor")
    parser.add_argument("--interval", type=int, default=30, help="Monitoring interval in seconds")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--config", type=str, default="config_amd_rx580.json", help="Config file")
    parser.add_argument("--once", action="store_true", help="Print status once and exit")
    
    args = parser.parse_args()
    
    monitor = AMDTrainingMonitor(args.log_dir, args.config)
    
    if args.once:
        monitor.print_status()
    else:
        monitor.monitor_continuously(args.interval)

if __name__ == "__main__":
    main() 