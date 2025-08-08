"""ManaMind - AI agent for playing Magic: The Gathering at superhuman level.

This package contains the core components for training and deploying an AI agent
that can play Magic: The Gathering using deep reinforcement learning.
"""

__version__ = "0.1.0"
__author__ = "ManaMind Team"
__email__ = "team@manamind.ai"

from manamind.core.game_state import GameState
from manamind.core.action import Action
from manamind.models.policy_value_network import PolicyValueNetwork

__all__ = [
    "GameState",
    "Action", 
    "PolicyValueNetwork",
]
