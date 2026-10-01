"""Model evaluation and benchmarking.

NOT YET IMPLEMENTED. This module defines the interface only; every method
raises :class:`NotImplementedError`. See issues #19 and #7.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from manamind.core.agent import Agent


@dataclass
class EvaluationResult:
    """Aggregate result of an evaluation run."""

    games_played: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    elo_estimate: Optional[float] = None


class Evaluator:
    """Measures agent strength against a fixed opponent or a gauntlet."""

    def evaluate_against(
        self, agent: Agent, opponent: Agent, num_games: int = 100
    ) -> EvaluationResult:
        """Play a match and report win rate."""
        raise NotImplementedError("Evaluator is not implemented yet")

    def evaluate_against_forge_ai(
        self, agent: Agent, difficulty: str = "default", num_games: int = 100
    ) -> EvaluationResult:
        """Benchmark against Forge's built-in AI at a given difficulty."""
        raise NotImplementedError("Evaluator is not implemented yet")

    def compare_checkpoints(
        self, checkpoint_a: Any, checkpoint_b: Any, num_games: int = 100
    ) -> Dict[str, Any]:
        """Head-to-head comparison of two model versions."""
        raise NotImplementedError("Evaluator is not implemented yet")
