"""Driving complete games through the Forge engine.

NOT YET IMPLEMENTED. This module defines the interface only; every method
raises :class:`NotImplementedError`. See issues #19 and #6.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from manamind.core.agent import Agent


@dataclass
class ForgeGameResult:
    """Outcome of a single game played through Forge."""

    game_id: str
    winner: Optional[int]
    history: List[Tuple[Any, Any, Any]] = field(default_factory=list)
    num_turns: int = 0


class ForgeGameRunner:
    """Runs games between two agents inside the Forge engine."""

    def __init__(self, forge_client: Any) -> None:
        self.forge_client = forge_client

    def play_game(
        self,
        agent1: Agent,
        agent2: Agent,
        deck1: Any = None,
        deck2: Any = None,
    ) -> Optional[ForgeGameResult]:
        """Play one complete game and return its result and history."""
        raise NotImplementedError("ForgeGameRunner is not implemented yet")

    def play_match(
        self, agent1: Agent, agent2: Agent, num_games: int
    ) -> List[ForgeGameResult]:
        """Play ``num_games`` games, alternating who is on the play."""
        raise NotImplementedError("ForgeGameRunner is not implemented yet")
