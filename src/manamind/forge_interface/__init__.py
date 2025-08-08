"""Forge game engine interface for ManaMind.

This module provides the Python-Java bridge for communicating with the Forge
MTG engine. This is critical for Phase 1 training where the agent learns
to play against Forge's built-in AI.
"""

from manamind.forge_interface.forge_client import ForgeClient
# from manamind.forge_interface.game_runner import ForgeGameRunner
# from manamind.forge_interface.state_parser import ForgeStateParser

__all__ = [
    "ForgeClient",
    # "ForgeGameRunner",
    # "ForgeStateParser",
]
