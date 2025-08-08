"""Action representation and validation for Magic: The Gathering.

This module defines how actions (moves) in MTG are represented and validated.
Actions include playing lands, casting spells, activating abilities, etc.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import torch

from manamind.core.game_state import Card, GameState


class ActionType(Enum):
    """Extended taxonomy of MTG actions for comprehensive gameplay."""

    # Basic game actions
    PLAY_LAND = "play_land"
    CAST_SPELL = "cast_spell"
    ACTIVATE_ABILITY = "activate_ability"
    ACTIVATE_MANA_ABILITY = "activate_mana_ability"

    # Priority and timing
    PASS_PRIORITY = "pass_priority"
    HOLD_PRIORITY = "hold_priority"

    # Combat actions
    DECLARE_ATTACKERS = "declare_attackers"
    DECLARE_BLOCKERS = "declare_blockers"
    ASSIGN_COMBAT_DAMAGE = "assign_combat_damage"
    ORDER_BLOCKERS = "order_blockers"

    # Special actions
    MULLIGAN = "mulligan"
    KEEP_HAND = "keep_hand"
    CONCEDE = "concede"

    # Card-specific actions
    DISCARD = "discard"
    SACRIFICE = "sacrifice"
    DESTROY = "destroy"
    EXILE = "exile"

    # Targeting and choices
    CHOOSE_TARGET = "choose_target"
    CHOOSE_MODE = "choose_mode"
    CHOOSE_X_VALUE = "choose_x_value"
    ORDER_CARDS = "order_cards"

    # Mana actions
    TAP_FOR_MANA = "tap_for_mana"
    PAY_MANA = "pay_mana"


@dataclass
class Action:
    """Enhanced action representation supporting all MTG complexities.

    This is the fundamental unit of decision-making for the AI agent.
    Each action contains all the information needed to execute it.
    """

    # Core action data
    action_type: ActionType
    player_id: int
    timestamp: float = field(default_factory=time.time)

    # Card references
    card: Optional[Card] = None
    target_cards: List[Card] = field(default_factory=list)

    # Player/permanent targets
    target_players: List[int] = field(default_factory=list)
    target_permanents: List[int] = field(default_factory=list)

    # Legacy compatibility
    target: Optional[Any] = None

    # Mana payment
    mana_payment: Optional[Dict[str, int]] = None
    alternative_cost: Optional[str] = None

    # Choices and parameters
    x_value: Optional[int] = None
    modes_chosen: List[str] = field(default_factory=list)
    order_choices: List[int] = field(default_factory=list)
    additional_choices: Dict[str, Any] = field(default_factory=dict)

    # Combat-specific
    attackers: List[int] = field(default_factory=list)
    defenders: List[int] = field(default_factory=list)
    blockers: Dict[int, List[int]] = field(default_factory=dict)
    damage_assignment: Dict[int, Dict[int, int]] = field(default_factory=dict)

    # Neural network representation
    action_id: Optional[int] = None
    action_vector: Optional[torch.Tensor] = None

    def get_complexity_score(self) -> int:
        """Calculate action complexity for MCTS guidance."""
        score = 1  # Base complexity

        if self.target_cards:
            score += len(self.target_cards)
        if self.modes_chosen:
            score += len(self.modes_chosen) * 2
        if self.x_value:
            score += 3
        if self.blockers:
            score += sum(len(blockers) for blockers in self.blockers.values())

        return score

    def get_all_targets(self) -> List[Any]:
        """Get all targets referenced by this action."""
        targets = []
        if self.card:
            targets.append(self.card)
        if self.target:
            targets.append(self.target)
        targets.extend(self.target_cards)
        targets.extend(self.target_players)
        targets.extend(self.target_permanents)
        return targets

    def is_valid(self, game_state: GameState) -> bool:
        """Check if this action is legal in the given game state.

        Args:
            game_state: Current game state

        Returns:
            True if the action is legal, False otherwise
        """
        # Delegate to specific validators based on action type
        validator = ACTION_VALIDATORS.get(self.action_type)
        if validator:
            return validator.validate(self, game_state)
        return False

    def execute(self, game_state: GameState) -> GameState:
        """Execute this action and return the resulting game state.

        Args:
            game_state: Current game state

        Returns:
            New game state after executing this action

        Raises:
            ValueError: If the action is not valid
        """
        if not self.is_valid(game_state):
            raise ValueError(f"Invalid action: {self}")

        executor = ACTION_EXECUTORS.get(self.action_type)
        if executor:
            return executor.execute(self, game_state)

        raise NotImplementedError(
            f"Execution not implemented for {self.action_type}"
        )


class ActionValidator(ABC):
    """Base class for validating specific types of actions."""

    @abstractmethod
    def validate(self, action: Action, game_state: GameState) -> bool:
        """Check if the action is valid in the given state."""
        pass


class PlayLandValidator(ActionValidator):
    """Validates land-playing actions."""

    def validate(self, action: Action, game_state: GameState) -> bool:
        player = game_state.players[action.player_id]

        # Check basic conditions
        if not action.card:
            return False

        # Must be in player's hand
        if action.card not in player.hand.cards:
            return False

        # Must be a land
        if "Land" not in action.card.card_type:
            return False

        # Can only play one land per turn (simplified rule)
        if not player.can_play_land():
            return False

        # Must have priority during main phase
        if game_state.priority_player != action.player_id:
            return False

        if game_state.phase not in ["main", "main2"]:
            return False

        # Stack must be empty (simplified rule)
        if len(game_state.stack) > 0:
            return False

        return True


class CastSpellValidator(ActionValidator):
    """Validates spell-casting actions."""

    def validate(self, action: Action, game_state: GameState) -> bool:
        player = game_state.players[action.player_id]

        # Check basic conditions
        if not action.card:
            return False

        # Must be in player's hand
        if action.card not in player.hand.cards:
            return False

        # Must not be a land
        if "Land" in action.card.card_type:
            return False

        # Must have priority
        if game_state.priority_player != action.player_id:
            return False

        # Check timing restrictions (simplified)
        if "Instant" not in action.card.card_type:
            # Sorcery-speed spell
            if game_state.active_player != action.player_id:
                return False
            if game_state.phase not in ["main", "main2"]:
                return False
            if len(game_state.stack) > 0:
                return False

        # Check mana cost (simplified - just check total mana)
        if player.total_mana() < action.card.converted_mana_cost:
            return False

        # TODO: More sophisticated mana checking, target validation, etc.

        return True


class PassPriorityValidator(ActionValidator):
    """Validates passing priority."""

    def validate(self, action: Action, game_state: GameState) -> bool:
        # Can always pass priority when you have it
        return game_state.priority_player == action.player_id


# Registry of validators
ACTION_VALIDATORS = {
    ActionType.PLAY_LAND: PlayLandValidator(),
    ActionType.CAST_SPELL: CastSpellValidator(),
    ActionType.PASS_PRIORITY: PassPriorityValidator(),
    # TODO: Add more validators
}


class ActionExecutor(ABC):
    """Base class for executing specific types of actions."""

    @abstractmethod
    def execute(self, action: Action, game_state: GameState) -> GameState:
        """Execute the action and return the new game state."""
        pass


class PlayLandExecutor(ActionExecutor):
    """Executes land-playing actions."""

    def execute(self, action: Action, game_state: GameState) -> GameState:
        # Create a copy of the game state (TODO: implement efficient copying)
        new_state = game_state.copy()

        player = new_state.players[action.player_id]

        # Move card from hand to battlefield
        player.hand.remove_card(action.card)
        player.battlefield.add_card(action.card)

        # Update lands played this turn
        player.lands_played_this_turn += 1

        # TODO: Trigger any relevant abilities

        return new_state


class CastSpellExecutor(ActionExecutor):
    """Executes spell-casting actions."""

    def execute(self, action: Action, game_state: GameState) -> GameState:
        # Create a copy of the game state
        new_state = game_state.copy()

        player = new_state.players[action.player_id]

        # Pay mana cost (simplified)
        # TODO: Proper mana payment logic

        # Move card from hand to stack
        player.hand.remove_card(action.card)
        new_state.stack.append(
            {
                "card": action.card,
                "controller": action.player_id,
                "targets": action.target,
                "choices": action.additional_choices,
            }
        )

        # TODO: Handle targeting, additional costs, etc.

        return new_state


class PassPriorityExecutor(ActionExecutor):
    """Executes priority passing."""

    def execute(self, action: Action, game_state: GameState) -> GameState:
        new_state = game_state.copy()

        # Pass priority to the other player
        new_state.priority_player = 1 - new_state.priority_player

        # TODO: Handle stack resolution, phase changes, etc.

        return new_state


# Registry of executors
ACTION_EXECUTORS = {
    ActionType.PLAY_LAND: PlayLandExecutor(),
    ActionType.CAST_SPELL: CastSpellExecutor(),
    ActionType.PASS_PRIORITY: PassPriorityExecutor(),
    # TODO: Add more executors
}


class ActionSpace:
    """Manages the space of all possible actions in Magic: The Gathering.

    This class is responsible for:
    1. Generating all legal actions from a given game state
    2. Converting actions to/from neural network representations
    3. Pruning invalid actions for efficiency
    """

    def __init__(self, max_actions: int = 10000):
        """Initialize the action space.

        Args:
            max_actions: Maximum number of actions to consider (for NN sizing)
        """
        self.max_actions = max_actions
        self.action_to_id: Dict[str, int] = {}
        self.id_to_action: Dict[int, str] = {}
        self._build_action_mappings()

    def _build_action_mappings(self) -> None:
        """Build mappings between actions and integer IDs for networks."""
        # TODO: Build comprehensive action vocabulary
        # This is a critical component for the neural network
        action_id = 0

        # Basic actions
        for action_type in ActionType:
            self.action_to_id[action_type.value] = action_id
            self.id_to_action[action_id] = action_type.value
            action_id += 1

        # TODO: Add card-specific actions, target-specific actions, etc.
        # This will likely need to be dynamic based on the current game state

    def get_legal_actions(self, game_state: GameState) -> List[Action]:
        """Generate all legal actions from the current game state.

        Args:
            game_state: Current game state

        Returns:
            List of all legal actions the current priority player can take
        """
        legal_actions = []
        current_player_id = game_state.priority_player
        current_player = game_state.players[current_player_id]

        # Can always pass priority
        legal_actions.append(
            Action(
                action_type=ActionType.PASS_PRIORITY,
                player_id=current_player_id,
            )
        )

        # Check for land plays
        if (
            game_state.active_player == current_player_id
            and current_player.can_play_land()
            and game_state.phase in ["main", "main2"]
            and len(game_state.stack) == 0
        ):

            for card in current_player.hand.cards:
                if "Land" in card.card_type:
                    action = Action(
                        action_type=ActionType.PLAY_LAND,
                        player_id=current_player_id,
                        card=card,
                    )
                    if action.is_valid(game_state):
                        legal_actions.append(action)

        # Check for spell casts
        for card in current_player.hand.cards:
            if "Land" not in card.card_type:
                action = Action(
                    action_type=ActionType.CAST_SPELL,
                    player_id=current_player_id,
                    card=card,
                )
                if action.is_valid(game_state):
                    legal_actions.append(action)

        # TODO: Add more action types (abilities, combat, etc.)

        return legal_actions

    def action_to_vector(self, action: Action) -> List[float]:
        """Convert an action to a vector representation for neural networks.

        Args:
            action: The action to convert

        Returns:
            Vector representation of the action
        """
        # TODO: Implement sophisticated action encoding
        # This is critical for the policy network
        vector = [0.0] * self.max_actions

        if action.action_type.value in self.action_to_id:
            action_idx = self.action_to_id[action.action_type.value]
            vector[action_idx] = 1.0

        return vector

    def vector_to_action(
        self, vector: List[float], game_state: GameState
    ) -> Optional[Action]:
        """Convert a vector representation back to an action.

        Args:
            vector: Vector representation from neural network
            game_state: Current game state for context

        Returns:
            The corresponding action, or None if invalid
        """
        # TODO: Implement sophisticated action decoding
        # Find the highest probability legal action
        legal_actions = self.get_legal_actions(game_state)

        if not legal_actions:
            return None

        # For now, just return the first legal action
        # TODO: Use the vector to select the best action
        return legal_actions[0]
