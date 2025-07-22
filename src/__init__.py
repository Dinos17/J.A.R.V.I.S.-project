"""
J.A.R.V.I.S. AI Training System
Advanced AI training pipeline optimized for AMD Radeon RX 580.
"""

__version__ = "2.0.0"
__author__ = "J.A.R.V.I.S. Development Team"
__description__ = "Advanced AI training pipeline with AMD optimizations"

# Import core modules
from .core.train import AdvancedJARVISTrainer, TrainingConfig
from .core.infer import JARVISInference

__all__ = [
    "AdvancedJARVISTrainer",
    "TrainingConfig", 
    "JARVISInference"
] 