"""Training infrastructure for ManaMind AI agent.

This module contains:
- Self-play training loops
- Neural network training
- Distributed training support
- Training data management

Note: ``TrainingDataManager`` is an interface stub. Every method raises
``NotImplementedError``. See issue #19.
"""

from manamind.training.data_manager import TrainingDataManager
from manamind.training.self_play import SelfPlayTrainer

__all__ = [
    "SelfPlayTrainer",
    "TrainingDataManager",
]
