"""Forge game engine interface for ManaMind.

This module provides the Python-Java bridge for communicating with the Forge
MTG engine. This is critical for Phase 1 training where the agent learns
to play against Forge's built-in AI.

``ForgeEnv`` drives a Forge seat over the JSON pipe in ``tools/forge-bridge``
(#74). ``ForgeClient``, ``ForgeGameRunner`` and ``ForgeStateParser`` are the
older py4j-era stubs; ``ForgeEnv`` replaces them.
"""

from manamind.forge_interface.forge_client import ForgeClient
from manamind.forge_interface.forge_env import (
    ForgeBridgeError,
    ForgeEnv,
    StepResult,
    attack_reply,
    block_reply,
    bridge_command,
    priority_reply,
)
from manamind.forge_interface.game_runner import (
    ForgeGameResult,
    ForgeGameRunner,
)
from manamind.forge_interface.state_parser import ForgeStateParser

__all__ = [
    "ForgeBridgeError",
    "ForgeClient",
    "ForgeEnv",
    "StepResult",
    "attack_reply",
    "block_reply",
    "bridge_command",
    "priority_reply",
    "ForgeGameResult",
    "ForgeGameRunner",
    "ForgeStateParser",
]
