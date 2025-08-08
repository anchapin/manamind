"""Training infrastructure for ManaMind AI agent.

This module contains:
- Self-play training loops
- Neural network training
- Distributed training support
- Training data management
"""

from manamind.training.self_play import SelfPlayTrainer
from manamind.training.neural_trainer import NeuralNetworkTrainer
from manamind.training.data_manager import TrainingDataManager

__all__ = [
    "SelfPlayTrainer",
    "NeuralNetworkTrainer",
    "TrainingDataManager",
]