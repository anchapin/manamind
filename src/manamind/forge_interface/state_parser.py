"""Translation between Forge game state and ManaMind's GameState.

NOT YET IMPLEMENTED. This module defines the interface only; every method
raises :class:`NotImplementedError`. See issues #19 and #6.
"""

from typing import Any, Dict

from manamind.core.action import Action
from manamind.core.game_state import GameState


class ForgeStateParser:
    """Converts between Forge's representation and ManaMind's."""

    def parse_game_state(self, forge_state: Dict[str, Any]) -> GameState:
        """Build a :class:`GameState` from Forge's serialized state."""
        raise NotImplementedError("ForgeStateParser is not implemented yet")

    def parse_legal_actions(self, forge_state: Dict[str, Any]) -> Any:
        """Extract the legal action list Forge is offering."""
        raise NotImplementedError("ForgeStateParser is not implemented yet")

    def serialize_action(self, action: Action) -> Dict[str, Any]:
        """Encode a ManaMind action in the form Forge expects."""
        raise NotImplementedError("ForgeStateParser is not implemented yet")
