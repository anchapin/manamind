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
        targets: List[Any] = []
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


class DeclareAttackersValidator(ActionValidator):
    """Validates attacking creature selection during declare attackers step."""

    def validate(self, action: Action, game_state: GameState) -> bool:
        """Validate attacker declarations.

        Args:
            action: The declare attackers action
            game_state: Current game state

        Returns:
            True if the attacker declarations are valid
        """
        # Must be during declare attackers phase
        if game_state.phase != "combat":
            return False

        # Must be the active player
        if action.player_id != game_state.active_player:
            return False

        # Must have priority
        if game_state.priority_player != action.player_id:
            return False

        player = game_state.players[action.player_id]

        # Check each attacker
        for attacker_id in action.attackers:
            # Find the attacking creature
            attacker = None
            for card in player.battlefield.cards:
                if card.instance_id == attacker_id:
                    attacker = card
                    break

            if not attacker:
                return False

            # Must be a creature
            if not attacker.is_creature():
                return False

            # Must not be tapped
            if attacker.tapped:
                return False

            # Must not have summoning sickness (unless has haste)
            if attacker.summoning_sick and "Haste" not in attacker.keywords:
                return False

            # Creature can't attack if it has defender
            if "Defender" in attacker.keywords:
                return False

        return True


class DeclareBlockersValidator(ActionValidator):
    """Validates blocking assignments during declare blockers step."""

    def validate(self, action: Action, game_state: GameState) -> bool:
        """Validate blocker declarations.

        Args:
            action: The declare blockers action
            game_state: Current game state

        Returns:
            True if the blocker declarations are valid
        """
        # Must be during declare blockers phase
        if game_state.phase != "combat":
            return False

        # Must be the non-active player (defending player)
        if action.player_id == game_state.active_player:
            return False

        # Must have priority
        if game_state.priority_player != action.player_id:
            return False

        player = game_state.players[action.player_id]

        # Track which creatures are blocking
        blocking_creatures = set()

        # Check each blocking assignment
        for attacker_id, blocker_ids in action.blockers.items():
            # Verify attacker exists and is actually attacking
            attacker_exists = False
            for p in game_state.players:
                for card in p.battlefield.cards:
                    if card.instance_id == attacker_id and card.attacking:
                        attacker_exists = True
                        break

            if not attacker_exists:
                return False

            # Check each blocker
            for blocker_id in blocker_ids:
                # Can't use same creature to block multiple attackers
                if blocker_id in blocking_creatures:
                    return False
                blocking_creatures.add(blocker_id)

                # Find the blocking creature
                blocker = None
                for card in player.battlefield.cards:
                    if card.instance_id == blocker_id:
                        blocker = card
                        break

                if not blocker:
                    return False

                # Must be a creature
                if not blocker.is_creature():
                    return False

                # Must not be tapped
                if blocker.tapped:
                    return False

                # Can't block if creature can't block
                if "Can't block" in blocker.oracle_text:
                    return False

        return True


class AssignCombatDamageValidator(ActionValidator):
    """Validates combat damage assignment during first strike or damage."""

    def validate(self, action: Action, game_state: GameState) -> bool:
        """Validate combat damage assignments.

        Args:
            action: The assign combat damage action
            game_state: Current game state

        Returns:
            True if the damage assignments are valid
        """
        # Must be during combat damage phase
        if game_state.phase != "combat":
            return False

        # Must have priority
        if game_state.priority_player != action.player_id:
            return False

        # Check each damage assignment
        for source_id, damage_map in action.damage_assignment.items():
            # Find the damage source
            source_card = None
            for player in game_state.players:
                for card in player.battlefield.cards:
                    if card.instance_id == source_id:
                        source_card = card
                        break

            if not source_card:
                return False

            # Must be a creature in combat
            if not source_card.is_creature():
                return False

            if not (source_card.attacking or source_card.blocking):
                return False

            # Get creature's power for damage calculation
            power = source_card.current_power()
            if power is None or power <= 0:
                continue

            # Sum assigned damage
            total_damage = sum(damage_map.values())

            # Can't assign more damage than power
            if total_damage > power:
                return False

            # All damage must be assigned (unless no valid targets)
            valid_targets_exist = False
            for target_id in damage_map.keys():
                # Verify target is valid
                for player in game_state.players:
                    for card in player.battlefield.cards:
                        if card.instance_id == target_id:
                            valid_targets_exist = True
                            break

            if valid_targets_exist and total_damage < power:
                return False

        return True


class OrderBlockersValidator(ActionValidator):
    """Validates blocker ordering for damage assignment."""

    def validate(self, action: Action, game_state: GameState) -> bool:
        """Validate blocker ordering choices.

        Args:
            action: The order blockers action
            game_state: Current game state

        Returns:
            True if the blocker ordering is valid
        """
        # Must be during combat phase
        if game_state.phase != "combat":
            return False

        # Must be the attacking player (they choose blocker order)
        if action.player_id != game_state.active_player:
            return False

        # Must have priority
        if game_state.priority_player != action.player_id:
            return False

        # Validate the ordering makes sense with current combat state
        # For each attacker with multiple blockers, order must be specified
        for attacker_id, blocker_order in action.additional_choices.items():
            if not isinstance(blocker_order, list):
                return False

            # Find the attacking creature
            attacker = None
            for card in game_state.players[action.player_id].battlefield.cards:
                if card.instance_id == int(attacker_id) and card.attacking:
                    attacker = card
                    break

            if not attacker:
                return False

            # Verify all blockers in the order actually exist and are
            # blocking this attacker
            for blocker_id in blocker_order:
                blocker_found = False
                for player in game_state.players:
                    for card in player.battlefield.cards:
                        if (
                            card.instance_id == blocker_id
                            and card.blocking == int(attacker_id)
                        ):
                            blocker_found = True
                            break

                if not blocker_found:
                    return False

        return True


class ActivateAbilityValidator(ActionValidator):
    """Validates activated ability activation (non-mana abilities)."""

    def validate(self, action: Action, game_state: GameState) -> bool:
        """Validate activated ability activation.

        Args:
            action: The activate ability action
            game_state: Current game state

        Returns:
            True if the ability activation is valid
        """
        # Must have priority
        if game_state.priority_player != action.player_id:
            return False

        # Must have a source card
        if not action.card:
            return False

        player = game_state.players[action.player_id]

        # Find the source permanent
        source_card = None
        for card in player.battlefield.cards:
            if card.instance_id == action.card.instance_id:
                source_card = card
                break

        # Also check other zones for abilities that can be activated
        # from non-battlefield zones (e.g., graveyard, hand)
        if not source_card:
            # Check hand for abilities like cycling, flashback, etc.
            for card in player.hand.cards:
                if card.instance_id == action.card.instance_id:
                    source_card = card
                    break

            # Check graveyard for abilities like flashback, unearth, etc.
            if not source_card:
                for card in player.graveyard.cards:
                    if card.instance_id == action.card.instance_id:
                        source_card = card
                        break

        if not source_card:
            return False

        # Verify the ability exists on the card
        ability_text = action.additional_choices.get("ability_text", "")
        if not ability_text:
            return False

        # Check if the ability text appears in the card's oracle text
        if ability_text not in source_card.oracle_text:
            return False

        # Validate ability format: [Cost]: [Effect]
        if ":" not in ability_text:
            return False

        cost_part = ability_text.split(":")[0].strip()

        # Validate costs can be paid
        return self._validate_ability_costs(
            action, game_state, source_card, cost_part
        )

    def _validate_ability_costs(
        self,
        action: Action,
        game_state: GameState,
        source_card: Card,
        cost_text: str,
    ) -> bool:
        """Validate that all ability costs can be paid.

        Args:
            action: The activate ability action
            game_state: Current game state
            source_card: The card with the ability
            cost_text: The cost portion of the ability

        Returns:
            True if all costs can be paid
        """
        player = game_state.players[action.player_id]

        # Parse common cost types
        costs = [cost.strip() for cost in cost_text.split(",")]

        for cost in costs:
            # Mana costs (e.g., "1", "W", "UU", "2R")
            if self._is_mana_cost(cost):
                required_mana = self._parse_mana_cost(cost)
                if not self._can_pay_mana(player, required_mana):
                    return False

            # Tap cost
            elif cost == "T" or cost == "{T}":
                if source_card.tapped:
                    return False

            # Sacrifice costs
            elif cost.startswith("Sacrifice"):
                # Extract what to sacrifice
                sacrifice_target = cost.replace("Sacrifice ", "").strip()
                if not self._can_sacrifice(
                    player, source_card, sacrifice_target
                ):
                    return False

            # Discard costs
            elif cost.startswith("Discard"):
                cards_to_discard = 1  # Default
                if "a card" in cost or "Discard a card" in cost:
                    cards_to_discard = 1
                elif "two cards" in cost:
                    cards_to_discard = 2

                if len(player.hand.cards) < cards_to_discard:
                    return False

            # Life payment costs
            elif "Pay" in cost and "life" in cost:
                # Extract life amount (e.g., "Pay 2 life")
                import re

                life_match = re.search(r"Pay (\d+) life", cost)
                if life_match:
                    life_cost = int(life_match.group(1))
                    if player.life <= life_cost:
                        return False

            # Other costs - for now, assume they can be paid
            # TODO: Add more sophisticated cost parsing

        return True

    def _is_mana_cost(self, cost: str) -> bool:
        """Check if a cost string represents a mana cost."""
        import re

        # Remove braces if present
        cost = cost.replace("{", "").replace("}", "")

        # Check for common mana symbols
        mana_symbols = r"^[0-9WUBRG]*$"
        return bool(re.match(mana_symbols, cost))

    def _parse_mana_cost(self, cost: str) -> Dict[str, int]:
        """Parse a mana cost string into a dict of mana requirements.

        Args:
            cost: Mana cost string (e.g., "2RR", "WWU")

        Returns:
            Dict mapping mana colors to amounts needed
        """
        cost = cost.replace("{", "").replace("}", "")
        mana_dict = {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0, "colorless": 0}

        i = 0
        while i < len(cost):
            char = cost[i]
            if char.isdigit():
                # Handle multi-digit numbers
                number = ""
                while i < len(cost) and cost[i].isdigit():
                    number += cost[i]
                    i += 1
                mana_dict["colorless"] += int(number)
            elif char in "WUBRG":
                mana_dict[char] += 1
                i += 1
            else:
                i += 1

        return mana_dict

    def _can_pay_mana(
        self, player: Any, required_mana: Dict[str, int]
    ) -> bool:
        """Check if player can pay the required mana."""
        # Get player's available mana
        available_mana = player.total_mana()

        # Calculate total required mana
        total_required = sum(required_mana.values())

        # Simplified check - just verify total mana available
        # TODO: Implement proper colored mana checking
        return bool(available_mana >= total_required)

    def _can_sacrifice(
        self, player: Any, source_card: Card, sacrifice_target: str
    ) -> bool:
        """Check if the sacrifice cost can be paid."""
        # Common sacrifice patterns
        if sacrifice_target.lower() in [
            "this",
            "this permanent",
            source_card.name.lower(),
        ]:
            # Can sacrifice the source card itself
            return True

        if "creature" in sacrifice_target.lower():
            # Need a creature to sacrifice
            for card in player.battlefield.cards:
                if card.is_creature() and card != source_card:
                    return True
            return False

        if "artifact" in sacrifice_target.lower():
            # Need an artifact to sacrifice
            for card in player.battlefield.cards:
                if "Artifact" in card.card_type and card != source_card:
                    return True
            return False

        # For specific card names or complex conditions, assume valid
        # TODO: Implement more sophisticated sacrifice validation
        return True


class MulliganValidator(ActionValidator):
    """Validates mulligan decisions during game start."""

    def validate(self, action: Action, game_state: GameState) -> bool:
        """Check if the mulligan action is valid.

        Args:
            action: The mulligan action
            game_state: Current game state

        Returns:
            True if the mulligan is valid
        """
        # Mulligan only allowed before the game properly begins
        if game_state.phase not in ["pregame", "mulligan"]:
            return False

        player = game_state.players[action.player_id]

        # Player must have cards in hand to mulligan
        if len(player.hand.cards) == 0:
            return False

        # Can't mulligan if already at 0 cards
        if len(player.hand.cards) <= 1:
            return False

        return True


class KeepHandValidator(ActionValidator):
    """Validates keeping opening hands."""

    def validate(self, action: Action, game_state: GameState) -> bool:
        """Check if the keep hand action is valid.

        Args:
            action: The keep hand action
            game_state: Current game state

        Returns:
            True if keeping the hand is valid
        """
        # Keep hand only allowed during mulligan phase
        if game_state.phase not in ["pregame", "mulligan"]:
            return False

        player = game_state.players[action.player_id]

        # Player must have cards in hand to keep
        if len(player.hand.cards) == 0:
            return False

        return True


class ConcedeValidator(ActionValidator):
    """Validates conceding the game."""

    def validate(self, action: Action, game_state: GameState) -> bool:
        """Check if the concede action is valid.

        Args:
            action: The concede action
            game_state: Current game state

        Returns:
            True if conceding is valid
        """
        # Can always concede if the game is active
        if game_state.is_game_over():
            return False

        # Must be a valid player
        if action.player_id < 0 or action.player_id >= len(game_state.players):
            return False

        return True


class DiscardValidator(ActionValidator):
    """Validates discarding cards from hand."""

    def validate(self, action: Action, game_state: GameState) -> bool:
        """Check if the discard action is valid.

        Args:
            action: The discard action
            game_state: Current game state

        Returns:
            True if the discard is valid
        """
        player = game_state.players[action.player_id]

        # Must have a card to discard
        if not action.card:
            return False

        # Card must be in player's hand
        if action.card not in player.hand.cards:
            return False

        return True


class SacrificeValidator(ActionValidator):
    """Validates sacrificing permanents."""

    def validate(self, action: Action, game_state: GameState) -> bool:
        """Check if the sacrifice action is valid.

        Args:
            action: The sacrifice action
            game_state: Current game state

        Returns:
            True if the sacrifice is valid
        """
        player = game_state.players[action.player_id]

        # Must have a card to sacrifice
        if not action.card:
            return False

        # Card must be a permanent on the battlefield
        if action.card not in player.battlefield.cards:
            return False

        # Player must control the permanent
        return True


class DestroyValidator(ActionValidator):
    """Validates destroying permanents."""

    def validate(self, action: Action, game_state: GameState) -> bool:
        """Check if the destroy action is valid.

        Args:
            action: The destroy action
            game_state: Current game state

        Returns:
            True if the destruction is valid
        """
        # Must have a target permanent
        if not action.card:
            return False

        # Find the permanent on any player's battlefield
        permanent_found = False
        for player in game_state.players:
            if action.card in player.battlefield.cards:
                permanent_found = True
                break

        if not permanent_found:
            return False

        return True


class ExileValidator(ActionValidator):
    """Validates exiling cards/permanents."""

    def validate(self, action: Action, game_state: GameState) -> bool:
        """Check if the exile action is valid.

        Args:
            action: The exile action
            game_state: Current game state

        Returns:
            True if the exile is valid
        """
        # Must have a target card
        if not action.card:
            return False

        # Find the card in any zone
        card_found = False
        for player in game_state.players:
            zones = [player.hand, player.battlefield, player.graveyard]
            for zone in zones:
                if action.card in zone.cards:
                    card_found = True
                    break
            if card_found:
                break

        return card_found


class ChooseTargetValidator(ActionValidator):
    """Validates target selection for spells and abilities."""

    def validate(self, action: Action, game_state: GameState) -> bool:
        """Validate target selection.

        Args:
            action: The choose target action
            game_state: Current game state

        Returns:
            True if the target selection is valid
        """
        # Must have targets to validate
        all_targets = action.target_cards + action.target_players
        if not all_targets:
            return False

        # Validate each target
        for target in all_targets:
            if isinstance(target, Card):
                if not self._is_valid_card_target(target, game_state):
                    return False
            elif isinstance(target, int):
                if not self._is_valid_player_target(target, game_state):
                    return False

        # Check targeting restrictions from spell/ability requirements
        spell_requirements = action.additional_choices.get(
            "targeting_requirements", {}
        )
        if not self._meets_targeting_requirements(
            all_targets, spell_requirements, game_state
        ):
            return False

        return True

    def _is_valid_card_target(
        self, target: Card, game_state: GameState
    ) -> bool:
        """Check if a card is a valid target."""
        # Check if target exists in any game zone
        target_exists = False
        for player in game_state.players:
            zones = [player.battlefield, player.graveyard, player.hand]
            for zone in zones:
                if target in zone.cards:
                    target_exists = True
                    break
            if target_exists:
                break

        if not target_exists:
            return False

        # Check targeting restrictions
        if "Hexproof" in target.keywords:
            return False

        if "Shroud" in target.keywords:
            return False

        # Check protection
        for keyword in target.keywords:
            if keyword.startswith("Protection from"):
                # Simplified protection check
                # TODO: Implement full protection rules
                return False

        return True

    def _is_valid_player_target(
        self, target_player_id: int, game_state: GameState
    ) -> bool:
        """Check if a player is a valid target."""
        return 0 <= target_player_id < len(game_state.players)

    def _meets_targeting_requirements(
        self,
        targets: List[Any],
        requirements: Dict[str, Any],
        game_state: GameState,
    ) -> bool:
        """Check if targets meet spell/ability requirements."""
        if not requirements:
            return True

        target_count = requirements.get("count", 1)
        if len(targets) != target_count:
            return False

        target_type = requirements.get("type", "any")
        if target_type == "creature":
            for target in targets:
                if isinstance(target, Card) and not target.is_creature():
                    return False
        elif target_type == "player":
            for target in targets:
                if not isinstance(target, int):
                    return False

        return True


class ChooseModeValidator(ActionValidator):
    """Validates mode selection for modal spells."""

    def validate(self, action: Action, game_state: GameState) -> bool:
        """Validate mode selection.

        Args:
            action: The choose mode action
            game_state: Current game state

        Returns:
            True if the mode selection is valid
        """
        if not action.modes_chosen:
            return False

        if not action.card:
            return False

        # Get available modes from the spell
        available_modes = self._get_available_modes(action.card)
        if not available_modes:
            return False

        # Check that chosen modes are valid
        for mode in action.modes_chosen:
            if mode not in available_modes:
                return False

        # Check mode selection constraints
        mode_constraints = self._get_mode_constraints(action.card)
        return self._validate_mode_constraints(
            action.modes_chosen, mode_constraints
        )

    def _get_available_modes(self, card: Card) -> List[str]:
        """Extract available modes from a modal spell."""
        modes = []
        oracle_text = card.oracle_text.lower()

        # Look for common modal patterns
        if (
            "choose one" in oracle_text
            or "choose any number" in oracle_text
            or "choose two" in oracle_text
            or "choose three" in oracle_text
            or "choose" in oracle_text
            and ("—" in oracle_text or "•" in oracle_text)
        ):
            # Parse modes from oracle text (simplified)
            # In a full implementation, this would be more sophisticated
            if "• " in oracle_text:
                parts = oracle_text.split("• ")
                for i, part in enumerate(parts[1:]):
                    # mode_text unused - just for parsing structure
                    modes.append(f"mode_{i + 1}")

        return modes

    def _get_mode_constraints(self, card: Card) -> Dict[str, Any]:
        """Get constraints on mode selection."""
        oracle_text = card.oracle_text.lower()
        constraints = {}

        if "choose one" in oracle_text:
            constraints["min_modes"] = 1
            constraints["max_modes"] = 1
        elif "choose any number" in oracle_text:
            constraints["min_modes"] = 0
            constraints["max_modes"] = len(self._get_available_modes(card))
        elif "choose two" in oracle_text:
            constraints["min_modes"] = 2
            constraints["max_modes"] = 2

        return constraints

    def _validate_mode_constraints(
        self, chosen_modes: List[str], constraints: Dict[str, Any]
    ) -> bool:
        """Validate that mode selection meets constraints."""
        mode_count = len(chosen_modes)
        min_modes = constraints.get("min_modes", 1)
        max_modes = constraints.get("max_modes", 1)

        return min_modes <= mode_count <= max_modes


class ChooseXValueValidator(ActionValidator):
    """Validates X value selection for X-cost spells."""

    def validate(self, action: Action, game_state: GameState) -> bool:
        """Validate X value selection.

        Args:
            action: The choose X value action
            game_state: Current game state

        Returns:
            True if the X value selection is valid
        """
        if action.x_value is None:
            return False

        # X must be non-negative
        if action.x_value < 0:
            return False

        if not action.card:
            return False

        # Check that the spell actually has X in its cost
        if not self._has_x_cost(action.card):
            return False

        # Check that player can pay for X
        player = game_state.players[action.player_id]
        total_cost = self._calculate_total_cost(action.card, action.x_value)

        if not self._can_pay_cost(player, total_cost):
            return False

        return True

    def _has_x_cost(self, card: Card) -> bool:
        """Check if a card has X in its mana cost."""
        return "X" in card.mana_cost

    def _calculate_total_cost(
        self, card: Card, x_value: int
    ) -> Dict[str, int]:
        """Calculate total mana cost including X value."""
        # Simplified cost calculation
        base_cost = self._parse_mana_cost(card.mana_cost.replace("X", "0"))
        base_cost["colorless"] += x_value
        return base_cost

    def _parse_mana_cost(self, cost: str) -> Dict[str, int]:
        """Parse a mana cost string into components."""
        cost = cost.replace("{", "").replace("}", "")
        mana_dict = {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0, "colorless": 0}

        i = 0
        while i < len(cost):
            char = cost[i]
            if char.isdigit():
                number = ""
                while i < len(cost) and cost[i].isdigit():
                    number += cost[i]
                    i += 1
                mana_dict["colorless"] += int(number)
            elif char in "WUBRG":
                mana_dict[char] += 1
                i += 1
            else:
                i += 1

        return mana_dict

    def _can_pay_cost(
        self, player: Any, required_mana: Dict[str, int]
    ) -> bool:
        """Check if player can pay the required mana cost."""
        total_required = sum(required_mana.values())
        available_mana = player.total_mana()
        return available_mana >= total_required


class TapForManaValidator(ActionValidator):
    """Validates tapping permanents for mana."""

    def validate(self, action: Action, game_state: GameState) -> bool:
        """Validate tapping for mana.

        Args:
            action: The tap for mana action
            game_state: Current game state

        Returns:
            True if tapping for mana is valid
        """
        if not action.card:
            return False

        player = game_state.players[action.player_id]

        # Find the card on the battlefield
        source_card = None
        for card in player.battlefield.cards:
            if card.instance_id == action.card.instance_id:
                source_card = card
                break

        if not source_card:
            return False

        # Card must not already be tapped
        if source_card.tapped:
            return False

        # Check if the permanent can produce mana
        if not self._can_produce_mana(source_card):
            return False

        # Verify mana ability requirements
        mana_ability = action.additional_choices.get("mana_ability", "")
        if mana_ability and not self._validate_mana_ability(
            source_card, mana_ability
        ):
            return False

        return True

    def _can_produce_mana(self, card: Card) -> bool:
        """Check if a permanent can produce mana."""
        # Basic lands produce mana
        basic_lands = ["Plains", "Island", "Swamp", "Mountain", "Forest"]
        if any(land in card.name for land in basic_lands):
            return True

        # Check for mana abilities in oracle text
        oracle_lower = card.oracle_text.lower()
        mana_keywords = ["add", "mana", "{w}", "{u}", "{b}", "{r}", "{g}"]
        return any(keyword in oracle_lower for keyword in mana_keywords)

    def _validate_mana_ability(self, card: Card, ability_text: str) -> bool:
        """Validate specific mana ability."""
        if not ability_text:
            return True

        # Check if ability exists on the card
        return ability_text.lower() in card.oracle_text.lower()


class PayManaValidator(ActionValidator):
    """Validates mana payment from mana pool."""

    def validate(self, action: Action, game_state: GameState) -> bool:
        """Validate mana payment.

        Args:
            action: The pay mana action
            game_state: Current game state

        Returns:
            True if the mana payment is valid
        """
        if not action.mana_payment:
            return False

        player = game_state.players[action.player_id]

        # Check that player has sufficient mana in pool
        return self._can_pay_from_pool(player, action.mana_payment)

    def _can_pay_from_pool(
        self, player: Any, mana_payment: Dict[str, int]
    ) -> bool:
        """Check if player can pay mana from their mana pool."""
        # Get player's mana pool (create if doesn't exist)
        if not hasattr(player, "mana_pool"):
            player.mana_pool = {
                "W": 0,
                "U": 0,
                "B": 0,
                "R": 0,
                "G": 0,
                "colorless": 0,
            }

        # Check each mana type
        for mana_type, amount in mana_payment.items():
            if amount < 0:
                return False

            available = player.mana_pool.get(mana_type, 0)
            if available < amount:
                # Try to pay with colorless if specific color unavailable
                if mana_type in ["W", "U", "B", "R", "G"]:
                    colorless_available = player.mana_pool.get("colorless", 0)
                    if colorless_available < amount - available:
                        return False
                else:
                    return False

        return True


class ActivateManaAbilityValidator(ActionValidator):
    """Validates mana ability activation (fast-resolution abilities)."""

    def validate(self, action: Action, game_state: GameState) -> bool:
        """Validate mana ability activation.

        Args:
            action: The activate mana ability action
            game_state: Current game state

        Returns:
            True if the mana ability activation is valid
        """
        # Must have a source card
        if not action.card:
            return False

        player = game_state.players[action.player_id]

        # Find the source permanent
        source_card = None
        for card in player.battlefield.cards:
            if card.instance_id == action.card.instance_id:
                source_card = card
                break

        if not source_card:
            return False

        # Verify the ability exists and produces mana
        ability_text = action.additional_choices.get("ability_text", "")
        if not ability_text:
            # Check for basic land mana abilities
            if self._is_basic_land_mana_ability(source_card):
                return not source_card.tapped
            return False

        # Verify it's actually a mana ability
        if not self._is_mana_ability(ability_text):
            return False

        # Check if the ability text appears in the card's oracle text
        if ability_text not in source_card.oracle_text:
            return False

        # Validate costs can be paid (mana abilities usually just tap)
        if ":" in ability_text:
            cost_part = ability_text.split(":")[0].strip()
            if cost_part == "T" or cost_part == "{T}":
                return not source_card.tapped

        return True

    def _is_basic_land_mana_ability(self, card: Card) -> bool:
        """Check if this is a basic land with implicit mana ability."""
        basic_lands = ["Plains", "Island", "Swamp", "Mountain", "Forest"]
        return any(land_type in card.name for land_type in basic_lands)

    def _is_mana_ability(self, ability_text: str) -> bool:
        """Check if an ability text represents a mana ability.

        Mana abilities are abilities that:
        1. Could add mana to a player's mana pool
        2. Don't target
        3. Aren't loyalty abilities
        """
        # Look for mana production keywords
        mana_keywords = ["add", "mana", "{W}", "{U}", "{B}", "{R}", "{G}"]

        return any(
            keyword in ability_text.lower() for keyword in mana_keywords
        )


# Registry of validators
ACTION_VALIDATORS = {
    ActionType.PLAY_LAND: PlayLandValidator(),
    ActionType.CAST_SPELL: CastSpellValidator(),
    ActionType.PASS_PRIORITY: PassPriorityValidator(),
    ActionType.DECLARE_ATTACKERS: DeclareAttackersValidator(),
    ActionType.DECLARE_BLOCKERS: DeclareBlockersValidator(),
    ActionType.ASSIGN_COMBAT_DAMAGE: AssignCombatDamageValidator(),
    ActionType.ORDER_BLOCKERS: OrderBlockersValidator(),
    ActionType.ACTIVATE_ABILITY: ActivateAbilityValidator(),
    ActionType.ACTIVATE_MANA_ABILITY: ActivateManaAbilityValidator(),
    ActionType.MULLIGAN: MulliganValidator(),
    ActionType.KEEP_HAND: KeepHandValidator(),
    ActionType.CONCEDE: ConcedeValidator(),
    ActionType.DISCARD: DiscardValidator(),
    ActionType.SACRIFICE: SacrificeValidator(),
    ActionType.DESTROY: DestroyValidator(),
    ActionType.EXILE: ExileValidator(),
    ActionType.CHOOSE_TARGET: ChooseTargetValidator(),
    ActionType.CHOOSE_MODE: ChooseModeValidator(),
    ActionType.CHOOSE_X_VALUE: ChooseXValueValidator(),
    ActionType.TAP_FOR_MANA: TapForManaValidator(),
    ActionType.PAY_MANA: PayManaValidator(),
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
        if action.card:
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
        if action.card:
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


class DeclareAttackersExecutor(ActionExecutor):
    """Executes attacker declarations."""

    def execute(self, action: Action, game_state: GameState) -> GameState:
        """Execute attacker declarations.

        Args:
            action: The declare attackers action
            game_state: Current game state

        Returns:
            New game state with attackers declared
        """
        new_state = game_state.copy()
        player = new_state.players[action.player_id]

        # Set attacking status for declared attackers
        for attacker_id in action.attackers:
            for card in player.battlefield.cards:
                if card.instance_id == attacker_id:
                    card.attacking = True
                    card.tapped = True  # Attacking creatures become tapped
                    break

        # Add to game history
        new_state.history.append(
            {
                "action": "declare_attackers",
                "player": action.player_id,
                "attackers": action.attackers.copy(),
                "turn": new_state.turn_number,
            }
        )

        # Pass priority to defending player for blocker declarations
        new_state.priority_player = 1 - action.player_id

        return new_state


class DeclareBlockersExecutor(ActionExecutor):
    """Executes blocking assignments."""

    def execute(self, action: Action, game_state: GameState) -> GameState:
        """Execute blocker declarations.

        Args:
            action: The declare blockers action
            game_state: Current game state

        Returns:
            New game state with blockers declared
        """
        new_state = game_state.copy()
        player = new_state.players[action.player_id]

        # Set blocking status for declared blockers
        for attacker_id, blocker_ids in action.blockers.items():
            for blocker_id in blocker_ids:
                for card in player.battlefield.cards:
                    if card.instance_id == blocker_id:
                        card.blocking = attacker_id
                        break

            # Update the attacker's blocked_by list
            for p in new_state.players:
                for card in p.battlefield.cards:
                    if card.instance_id == attacker_id:
                        card.blocked_by.extend(blocker_ids)
                        break

        # Add to game history
        new_state.history.append(
            {
                "action": "declare_blockers",
                "player": action.player_id,
                "blockers": action.blockers.copy(),
                "turn": new_state.turn_number,
            }
        )

        # Pass priority back to attacking player for damage assignment
        new_state.priority_player = 1 - action.player_id

        return new_state


class AssignCombatDamageExecutor(ActionExecutor):
    """Executes combat damage assignment."""

    def execute(self, action: Action, game_state: GameState) -> GameState:
        """Execute combat damage assignments.

        Args:
            action: The assign combat damage action
            game_state: Current game state

        Returns:
            New game state with combat damage assigned and resolved
        """
        new_state = game_state.copy()

        # Apply damage to creatures and players
        for source_id, damage_map in action.damage_assignment.items():
            for target_id, damage in damage_map.items():
                if damage <= 0:
                    continue

                # Find target (creature or player)
                target_found = False

                # Check for creature targets
                for player in new_state.players:
                    for card in player.battlefield.cards:
                        if card.instance_id == target_id:
                            # Apply damage to creature
                            current_damage = card.counters.get("damage", 0)
                            card.counters["damage"] = current_damage + damage

                            # Check if creature dies (damage >= toughness)
                            toughness = card.current_toughness()
                            if (
                                toughness is not None
                                and card.counters["damage"] >= toughness
                            ):
                                # Move to graveyard
                                player.battlefield.remove_card(card)
                                player.graveyard.add_card(card)
                                card.zone = "graveyard"

                            target_found = True
                            break

                # Check for player targets
                if not target_found:
                    for i, player in enumerate(new_state.players):
                        if i == target_id:
                            player.life -= damage
                            break

        # Add to game history
        new_state.history.append(
            {
                "action": "assign_combat_damage",
                "player": action.player_id,
                "damage_assignment": action.damage_assignment.copy(),
                "turn": new_state.turn_number,
            }
        )

        # Clean up combat (remove attacking/blocking status)
        for player in new_state.players:
            for card in player.battlefield.cards:
                card.attacking = False
                card.blocking = None
                card.blocked_by.clear()

        # Move to end of combat
        new_state.priority_player = new_state.active_player

        return new_state


class OrderBlockersExecutor(ActionExecutor):
    """Executes blocker ordering for damage assignment."""

    def execute(self, action: Action, game_state: GameState) -> GameState:
        """Execute blocker ordering choices.

        Args:
            action: The order blockers action
            game_state: Current game state

        Returns:
            New game state with blocker order established
        """
        new_state = game_state.copy()

        # Store blocker order information for damage assignment
        # This would typically be used by the damage assignment step
        blocker_orders = {}
        for attacker_id, blocker_order in action.additional_choices.items():
            blocker_orders[int(attacker_id)] = blocker_order.copy()

        # Add to game history for reference during damage assignment
        new_state.history.append(
            {
                "action": "order_blockers",
                "player": action.player_id,
                "blocker_orders": blocker_orders,
                "turn": new_state.turn_number,
            }
        )

        # Pass priority for damage assignment
        new_state.priority_player = action.player_id

        return new_state


class ActivateAbilityExecutor(ActionExecutor):
    """Executes activated ability activation (non-mana abilities)."""

    def execute(self, action: Action, game_state: GameState) -> GameState:
        """Execute activated ability activation.

        Args:
            action: The activate ability action
            game_state: Current game state

        Returns:
            New game state with ability activated and put on the stack
        """
        new_state = game_state.copy()

        # Find the source card in the new state
        source_card = self._find_source_card(action, new_state)
        if not source_card:
            raise ValueError("Source card not found in game state")

        # Get the ability being activated
        ability_text = action.additional_choices.get("ability_text", "")
        cost_part = ability_text.split(":")[0].strip()

        # Pay the costs
        self._pay_ability_costs(action, new_state, source_card, cost_part)

        # Put the ability on the stack (non-mana abilities use the stack)
        ability_object = {
            "type": "activated_ability",
            "source": source_card,
            "controller": action.player_id,
            "ability_text": ability_text,
            "targets": action.target_cards + action.target_players,
            "choices": action.additional_choices.copy(),
        }

        new_state.stack.append(ability_object)

        # Add to game history
        new_state.history.append(
            {
                "action": "activate_ability",
                "player": action.player_id,
                "source": source_card.name,
                "ability": ability_text,
                "turn": new_state.turn_number,
            }
        )

        return new_state

    def _find_source_card(
        self, action: Action, game_state: GameState
    ) -> Optional[Card]:
        """Find the source card for the ability in the game state."""
        player = game_state.players[action.player_id]

        # Check battlefield first
        for card in player.battlefield.cards:
            if action.card and card.instance_id == action.card.instance_id:
                return card

        # Check hand for abilities like cycling
        for card in player.hand.cards:
            if action.card and card.instance_id == action.card.instance_id:
                return card

        # Check graveyard for abilities like flashback
        for card in player.graveyard.cards:
            if action.card and card.instance_id == action.card.instance_id:
                return card

        return None

    def _pay_ability_costs(
        self,
        action: Action,
        game_state: GameState,
        source_card: Card,
        cost_text: str,
    ) -> None:
        """Pay the costs for activating the ability.

        Args:
            action: The activate ability action
            game_state: Current game state (will be modified)
            source_card: The card with the ability
            cost_text: The cost portion of the ability
        """
        player = game_state.players[action.player_id]
        costs = [cost.strip() for cost in cost_text.split(",")]

        for cost in costs:
            # Mana costs
            if self._is_mana_cost(cost):
                required_mana = self._parse_mana_cost(cost)
                self._pay_mana(player, required_mana)

            # Tap cost
            elif cost == "T" or cost == "{T}":
                source_card.tapped = True

            # Sacrifice costs
            elif cost.startswith("Sacrifice"):
                sacrifice_target = cost.replace("Sacrifice ", "").strip()
                self._pay_sacrifice_cost(player, source_card, sacrifice_target)

            # Discard costs
            elif cost.startswith("Discard"):
                cards_to_discard = 1  # Default
                if "two cards" in cost:
                    cards_to_discard = 2

                # For simplicity, discard from end of hand
                for _ in range(min(cards_to_discard, len(player.hand.cards))):
                    if player.hand.cards:
                        discarded = player.hand.cards.pop()
                        player.graveyard.add_card(discarded)
                        discarded.zone = "graveyard"

            # Life payment costs
            elif "Pay" in cost and "life" in cost:
                import re

                life_match = re.search(r"Pay (\d+) life", cost)
                if life_match:
                    life_cost = int(life_match.group(1))
                    player.life -= life_cost

    def _is_mana_cost(self, cost: str) -> bool:
        """Check if a cost string represents a mana cost."""
        import re

        cost = cost.replace("{", "").replace("}", "")
        mana_symbols = r"^[0-9WUBRG]*$"
        return bool(re.match(mana_symbols, cost))

    def _parse_mana_cost(self, cost: str) -> Dict[str, int]:
        """Parse a mana cost string into a dict of mana requirements."""
        cost = cost.replace("{", "").replace("}", "")
        mana_dict = {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0, "colorless": 0}

        i = 0
        while i < len(cost):
            char = cost[i]
            if char.isdigit():
                number = ""
                while i < len(cost) and cost[i].isdigit():
                    number += cost[i]
                    i += 1
                mana_dict["colorless"] += int(number)
            elif char in "WUBRG":
                mana_dict[char] += 1
                i += 1
            else:
                i += 1

        return mana_dict

    def _pay_mana(self, player: Any, required_mana: Dict[str, int]) -> None:
        """Pay mana costs (simplified implementation)."""
        # TODO: Implement proper mana pool management
        # For now, assume mana is automatically tapped from lands
        total_required = sum(required_mana.values())

        # Tap lands to pay for mana (simplified)
        mana_paid = 0
        for card in player.battlefield.cards:
            if mana_paid >= total_required:
                break
            if "Land" in card.card_type and not card.tapped:
                card.tapped = True
                mana_paid += 1

    def _pay_sacrifice_cost(
        self, player: Any, source_card: Card, sacrifice_target: str
    ) -> None:
        """Pay sacrifice costs."""
        if sacrifice_target.lower() in [
            "this",
            "this permanent",
            source_card.name.lower(),
        ]:
            # Sacrifice the source card itself
            player.battlefield.remove_card(source_card)
            player.graveyard.add_card(source_card)
            source_card.zone = "graveyard"
        elif "creature" in sacrifice_target.lower():
            # Sacrifice a creature (choose first available)
            for card in player.battlefield.cards:
                if card.is_creature() and card != source_card:
                    player.battlefield.remove_card(card)
                    player.graveyard.add_card(card)
                    card.zone = "graveyard"
                    break
        # TODO: Handle more sacrifice patterns


class ActivateManaAbilityExecutor(ActionExecutor):
    """Executes mana ability activation (special fast-resolution abilities)."""

    def execute(self, action: Action, game_state: GameState) -> GameState:
        """Execute mana ability activation.

        Args:
            action: The activate mana ability action
            game_state: Current game state

        Returns:
            New game state with mana ability resolved immediately
        """
        new_state = game_state.copy()
        player = new_state.players[action.player_id]

        # Find the source card
        source_card = None
        for card in player.battlefield.cards:
            if action.card and card.instance_id == action.card.instance_id:
                source_card = card
                break

        if not source_card:
            raise ValueError("Source card not found in game state")

        # Get the ability being activated
        ability_text = action.additional_choices.get("ability_text", "")

        # Pay costs and resolve immediately (mana abilities don't use stack)
        if ability_text:
            cost_part = ability_text.split(":")[0].strip()
            if cost_part == "T" or cost_part == "{T}":
                source_card.tapped = True

            # Add mana to player's mana pool
            mana_produced = self._parse_mana_production(ability_text)
            self._add_mana_to_pool(player, mana_produced)
        else:
            # Basic land mana ability
            if self._is_basic_land_mana_ability(source_card):
                source_card.tapped = True
                mana_type = self._get_basic_land_mana_type(source_card)
                self._add_mana_to_pool(player, {mana_type: 1})

        # Add to game history
        new_state.history.append(
            {
                "action": "activate_mana_ability",
                "player": action.player_id,
                "source": source_card.name,
                "mana_produced": ability_text or "basic land mana",
                "turn": new_state.turn_number,
            }
        )

        return new_state

    def _is_basic_land_mana_ability(self, card: Card) -> bool:
        """Check if this is a basic land with implicit mana ability."""
        basic_lands = ["Plains", "Island", "Swamp", "Mountain", "Forest"]
        return any(land_type in card.name for land_type in basic_lands)

    def _get_basic_land_mana_type(self, card: Card) -> str:
        """Get the mana type produced by a basic land."""
        if "Plains" in card.name:
            return "W"
        elif "Island" in card.name:
            return "U"
        elif "Swamp" in card.name:
            return "B"
        elif "Mountain" in card.name:
            return "R"
        elif "Forest" in card.name:
            return "G"
        return "colorless"

    def _parse_mana_production(self, ability_text: str) -> Dict[str, int]:
        """Parse the mana production from ability text.

        Args:
            ability_text: The full ability text

        Returns:
            Dict mapping mana colors to amounts produced
        """
        import re

        mana_dict = {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0, "colorless": 0}

        # Look for patterns like "Add {W}", "Add {2}", "Add {R}{R}"
        # This is a simplified parser
        effect_part = ability_text.split(":")[-1].strip().lower()

        # Find mana symbols in braces
        mana_matches = re.findall(r"\{([wubrgxc]|\d+)\}", effect_part)
        for match in mana_matches:
            if match.isdigit():
                mana_dict["colorless"] += int(match)
            elif match == "c":
                mana_dict["colorless"] += 1
            elif match in "wubrg":
                mana_dict[match.upper()] += 1

        return mana_dict

    def _add_mana_to_pool(
        self, player: Any, mana_produced: Dict[str, int]
    ) -> None:
        """Add mana to the player's mana pool.

        Args:
            player: The player receiving the mana
            mana_produced: Dict of mana types and amounts to add
        """
        # Initialize mana pool if it doesn't exist or is empty
        if not hasattr(player, "mana_pool") or not player.mana_pool:
            player.mana_pool = {
                "W": 0,
                "U": 0,
                "B": 0,
                "R": 0,
                "G": 0,
                "colorless": 0,
            }

        # Ensure all required keys exist
        for mana_type in ["W", "U", "B", "R", "G", "colorless"]:
            if mana_type not in player.mana_pool:
                player.mana_pool[mana_type] = 0

        for mana_type, amount in mana_produced.items():
            player.mana_pool[mana_type] += amount


class MulliganExecutor(ActionExecutor):
    """Executes mulligan (reshuffle, draw new hand)."""

    def execute(self, action: Action, game_state: GameState) -> GameState:
        """Execute mulligan action.

        Args:
            action: The mulligan action
            game_state: Current game state

        Returns:
            New game state after mulligan
        """
        new_state = game_state.copy()
        player = new_state.players[action.player_id]

        # Put hand back into library
        while player.hand.cards:
            card = player.hand.cards.pop()
            player.library.add_card(card)
            card.zone = "library"

        # Shuffle library (manual shuffle since Zone doesn't have method)
        import random

        random.shuffle(player.library.cards)

        # Count existing mulligans to determine how many cards to draw
        existing_mulligans = sum(
            1
            for entry in new_state.history
            if entry.get("action") == "mulligan"
            and entry.get("player") == action.player_id
        )
        # Draw one fewer card than before
        cards_to_draw = max(0, 7 - existing_mulligans - 1)
        for _ in range(cards_to_draw):
            if player.library.cards:
                card = player.library.cards.pop(0)
                player.hand.add_card(card)
                card.zone = "hand"

        # Track mulligan count in history
        # (since player model doesn't allow dynamic attributes)
        # Count existing mulligans in history
        mulligan_count = (
            sum(
                1
                for entry in new_state.history
                if (
                    entry.get("action") == "mulligan"
                    and entry.get("player") == action.player_id
                )
            )
            + 1
        )

        # Add to game history
        new_state.history.append(
            {
                "action": "mulligan",
                "player": action.player_id,
                "cards_drawn": cards_to_draw,
                "mulligan_count": mulligan_count,
                "turn": new_state.turn_number,
            }
        )

        return new_state


class KeepHandExecutor(ActionExecutor):
    """Executes keeping opening hand."""

    def execute(self, action: Action, game_state: GameState) -> GameState:
        """Execute keep hand action.

        Args:
            action: The keep hand action
            game_state: Current game state

        Returns:
            New game state after keeping hand
        """
        new_state = game_state.copy()
        player = new_state.players[action.player_id]

        # Mark that player has kept their hand (record in history)

        # Add to game history
        new_state.history.append(
            {
                "action": "keep_hand",
                "player": action.player_id,
                "hand_size": len(player.hand.cards),
                "turn": new_state.turn_number,
            }
        )

        return new_state


class ConcedeExecutor(ActionExecutor):
    """Executes conceding the game."""

    def execute(self, action: Action, game_state: GameState) -> GameState:
        """Execute concede action.

        Args:
            action: The concede action
            game_state: Current game state

        Returns:
            New game state after conceding
        """
        new_state = game_state.copy()

        # Mark game as over by setting life to 0
        # This will cause is_game_over() to return True and winner() to work
        player = new_state.players[action.player_id]
        player.life = 0

        # Add to game history
        new_state.history.append(
            {
                "action": "concede",
                "player": action.player_id,
                "winner": new_state.winner(),
                "turn": new_state.turn_number,
            }
        )

        return new_state


class DiscardExecutor(ActionExecutor):
    """Executes card discard."""

    def execute(self, action: Action, game_state: GameState) -> GameState:
        """Execute discard action.

        Args:
            action: The discard action
            game_state: Current game state

        Returns:
            New game state after discarding
        """
        new_state = game_state.copy()
        player = new_state.players[action.player_id]

        if action.card:
            # Move card from hand to graveyard
            player.hand.remove_card(action.card)
            player.graveyard.add_card(action.card)
            action.card.zone = "graveyard"

        # Add to game history
        new_state.history.append(
            {
                "action": "discard",
                "player": action.player_id,
                "card": action.card.name if action.card else "unknown",
                "turn": new_state.turn_number,
            }
        )

        return new_state


class SacrificeExecutor(ActionExecutor):
    """Executes permanent sacrifice."""

    def execute(self, action: Action, game_state: GameState) -> GameState:
        """Execute sacrifice action.

        Args:
            action: The sacrifice action
            game_state: Current game state

        Returns:
            New game state after sacrificing
        """
        new_state = game_state.copy()
        player = new_state.players[action.player_id]

        if action.card:
            # Move permanent from battlefield to graveyard
            player.battlefield.remove_card(action.card)
            player.graveyard.add_card(action.card)
            action.card.zone = "graveyard"

            # Reset creature states
            action.card.attacking = False
            action.card.blocking = None
            action.card.tapped = False

        # Add to game history
        new_state.history.append(
            {
                "action": "sacrifice",
                "player": action.player_id,
                "card": action.card.name if action.card else "unknown",
                "turn": new_state.turn_number,
            }
        )

        return new_state


class DestroyExecutor(ActionExecutor):
    """Executes permanent destruction."""

    def execute(self, action: Action, game_state: GameState) -> GameState:
        """Execute destroy action.

        Args:
            action: The destroy action
            game_state: Current game state

        Returns:
            New game state after destroying
        """
        new_state = game_state.copy()

        if action.card:
            # Find which player owns the permanent
            owner_id = None
            for i, player in enumerate(new_state.players):
                if action.card in player.battlefield.cards:
                    owner_id = i
                    break

            if owner_id is not None:
                owner = new_state.players[owner_id]

                # Move permanent from battlefield to graveyard
                owner.battlefield.remove_card(action.card)
                owner.graveyard.add_card(action.card)
                action.card.zone = "graveyard"

                # Reset creature states
                action.card.attacking = False
                action.card.blocking = None
                action.card.tapped = False

        # Add to game history
        new_state.history.append(
            {
                "action": "destroy",
                "player": action.player_id,
                "card": action.card.name if action.card else "unknown",
                "turn": new_state.turn_number,
            }
        )

        return new_state


class ChooseTargetExecutor(ActionExecutor):
    """Executes target selection."""

    def execute(self, action: Action, game_state: GameState) -> GameState:
        """Execute target selection.

        Args:
            action: The choose target action
            game_state: Current game state

        Returns:
            New game state with targets selected
        """
        new_state = game_state.copy()

        # Store target selections in game state or pending spell
        if hasattr(new_state, "pending_targets"):
            new_state.pending_targets.update(
                {
                    "cards": action.target_cards,
                    "players": action.target_players,
                    "permanents": action.target_permanents,
                }
            )
        else:
            new_state.pending_targets = {
                "cards": action.target_cards,
                "players": action.target_players,
                "permanents": action.target_permanents,
            }

        # Add to game history
        new_state.history.append(
            {
                "action": "choose_target",
                "player": action.player_id,
                "target_count": len(
                    action.target_cards + action.target_players
                ),
                "turn": new_state.turn_number,
            }
        )

        return new_state


class ChooseModeExecutor(ActionExecutor):
    """Executes mode selection."""

    def execute(self, action: Action, game_state: GameState) -> GameState:
        """Execute mode selection.

        Args:
            action: The choose mode action
            game_state: Current game state

        Returns:
            New game state with modes selected
        """
        new_state = game_state.copy()

        # Store mode selection for the spell being cast
        if hasattr(new_state, "pending_modes"):
            new_state.pending_modes = action.modes_chosen
        else:
            new_state.pending_modes = action.modes_chosen

        # Add to game history
        new_state.history.append(
            {
                "action": "choose_mode",
                "player": action.player_id,
                "modes": action.modes_chosen.copy(),
                "spell": action.card.name if action.card else "unknown",
                "turn": new_state.turn_number,
            }
        )

        return new_state


class ChooseXValueExecutor(ActionExecutor):
    """Executes X value selection."""

    def execute(self, action: Action, game_state: GameState) -> GameState:
        """Execute X value selection.

        Args:
            action: The choose X value action
            game_state: Current game state

        Returns:
            New game state with X value set
        """
        new_state = game_state.copy()

        # Store X value for the spell being cast
        if hasattr(new_state, "pending_x_value"):
            new_state.pending_x_value = action.x_value
        else:
            new_state.pending_x_value = action.x_value

        # Update mana cost of the spell if it's being cast
        if action.card and hasattr(new_state, "pending_spell_cost"):
            total_cost = self._calculate_total_cost(
                action.card, action.x_value
            )
            new_state.pending_spell_cost = total_cost

        # Add to game history
        new_state.history.append(
            {
                "action": "choose_x_value",
                "player": action.player_id,
                "x_value": action.x_value,
                "spell": action.card.name if action.card else "unknown",
                "turn": new_state.turn_number,
            }
        )

        return new_state

    def _calculate_total_cost(
        self, card: Card, x_value: int
    ) -> Dict[str, int]:
        """Calculate total mana cost including X value."""
        # Simplified cost calculation
        base_cost = self._parse_mana_cost(card.mana_cost.replace("X", "0"))
        base_cost["colorless"] += x_value
        return base_cost

    def _parse_mana_cost(self, cost: str) -> Dict[str, int]:
        """Parse a mana cost string into components."""
        cost = cost.replace("{", "").replace("}", "")
        mana_dict = {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0, "colorless": 0}

        i = 0
        while i < len(cost):
            char = cost[i]
            if char.isdigit():
                number = ""
                while i < len(cost) and cost[i].isdigit():
                    number += cost[i]
                    i += 1
                mana_dict["colorless"] += int(number)
            elif char in "WUBRG":
                mana_dict[char] += 1
                i += 1
            else:
                i += 1

        return mana_dict


class TapForManaExecutor(ActionExecutor):
    """Executes tapping permanents for mana."""

    def execute(self, action: Action, game_state: GameState) -> GameState:
        """Execute tapping for mana.

        Args:
            action: The tap for mana action
            game_state: Current game state

        Returns:
            New game state with mana added to pool
        """
        new_state = game_state.copy()
        player = new_state.players[action.player_id]

        # Find the source card
        source_card = None
        for card in player.battlefield.cards:
            if action.card and card.instance_id == action.card.instance_id:
                source_card = card
                break

        if not source_card:
            raise ValueError("Source card not found in game state")

        # Tap the permanent
        source_card.tapped = True

        # Add mana to player's mana pool
        mana_produced = self._determine_mana_production(
            source_card, action.additional_choices.get("mana_ability", "")
        )
        self._add_mana_to_pool(player, mana_produced)

        # Add to game history
        new_state.history.append(
            {
                "action": "tap_for_mana",
                "player": action.player_id,
                "source": source_card.name,
                "mana_produced": mana_produced,
                "turn": new_state.turn_number,
            }
        )

        return new_state

    def _determine_mana_production(
        self, card: Card, ability_text: str
    ) -> Dict[str, int]:
        """Determine what mana is produced by tapping."""
        # Handle basic lands
        if "Plains" in card.name:
            return {"W": 1, "U": 0, "B": 0, "R": 0, "G": 0, "colorless": 0}
        elif "Island" in card.name:
            return {"W": 0, "U": 1, "B": 0, "R": 0, "G": 0, "colorless": 0}
        elif "Swamp" in card.name:
            return {"W": 0, "U": 0, "B": 1, "R": 0, "G": 0, "colorless": 0}
        elif "Mountain" in card.name:
            return {"W": 0, "U": 0, "B": 0, "R": 1, "G": 0, "colorless": 0}
        elif "Forest" in card.name:
            return {"W": 0, "U": 0, "B": 0, "R": 0, "G": 1, "colorless": 0}

        # Handle explicit mana abilities
        if ability_text:
            return self._parse_mana_ability_production(ability_text)

        # Default to colorless mana
        return {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0, "colorless": 1}

    def _parse_mana_ability_production(
        self, ability_text: str
    ) -> Dict[str, int]:
        """Parse mana production from ability text."""
        import re

        mana_dict = {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0, "colorless": 0}

        # Find mana symbols in braces
        mana_matches = re.findall(r"\{([wubrgc]|\d+)\}", ability_text.lower())
        for match in mana_matches:
            if match.isdigit():
                mana_dict["colorless"] += int(match)
            elif match == "c":
                mana_dict["colorless"] += 1
            elif match in "wubrg":
                mana_dict[match.upper()] += 1

        return mana_dict

    def _add_mana_to_pool(
        self, player: Any, mana_produced: Dict[str, int]
    ) -> None:
        """Add mana to the player's mana pool."""
        # Initialize mana pool if it doesn't exist or is empty
        if not hasattr(player, "mana_pool") or not player.mana_pool:
            player.mana_pool = {
                "W": 0,
                "U": 0,
                "B": 0,
                "R": 0,
                "G": 0,
                "colorless": 0,
            }

        # Ensure all required keys exist
        for mana_type in ["W", "U", "B", "R", "G", "colorless"]:
            if mana_type not in player.mana_pool:
                player.mana_pool[mana_type] = 0

        for mana_type, amount in mana_produced.items():
            player.mana_pool[mana_type] += amount


class PayManaExecutor(ActionExecutor):
    """Executes mana payment from mana pool."""

    def execute(self, action: Action, game_state: GameState) -> GameState:
        """Execute mana payment.

        Args:
            action: The pay mana action
            game_state: Current game state

        Returns:
            New game state with mana paid
        """
        new_state = game_state.copy()
        player = new_state.players[action.player_id]

        if not action.mana_payment:
            return new_state

        # Ensure mana pool exists
        if not hasattr(player, "mana_pool"):
            player.mana_pool = {
                "W": 0,
                "U": 0,
                "B": 0,
                "R": 0,
                "G": 0,
                "colorless": 0,
            }

        # Pay mana from pool
        for mana_type, amount in action.mana_payment.items():
            if amount <= 0:
                continue

            # Pay from specific mana type first
            available = player.mana_pool.get(mana_type, 0)
            if available >= amount:
                player.mana_pool[mana_type] -= amount
            else:
                # Pay what we can from specific type
                player.mana_pool[mana_type] = 0
                remaining = amount - available

                # Pay remaining from colorless (if applicable)
                if mana_type in ["W", "U", "B", "R", "G"]:
                    colorless_available = player.mana_pool.get("colorless", 0)
                    if colorless_available >= remaining:
                        player.mana_pool["colorless"] -= remaining
                    else:
                        # Not enough mana - should not happen if validated
                        raise ValueError(
                            f"Insufficient mana to pay {mana_type}: {amount}"
                        )

        # Add to game history
        new_state.history.append(
            {
                "action": "pay_mana",
                "player": action.player_id,
                "mana_paid": action.mana_payment.copy(),
                "turn": new_state.turn_number,
            }
        )

        return new_state


class ExileExecutor(ActionExecutor):
    """Executes exiling cards."""

    def execute(self, action: Action, game_state: GameState) -> GameState:
        """Execute exile action.

        Args:
            action: The exile action
            game_state: Current game state

        Returns:
            New game state after exiling
        """
        new_state = game_state.copy()

        if action.card:
            # Find which player owns the card and which zone it's in
            for player in new_state.players:
                zones = [
                    (player.hand, "hand"),
                    (player.battlefield, "battlefield"),
                    (player.graveyard, "graveyard"),
                ]

                for zone, zone_name in zones:
                    if action.card in zone.cards:
                        # Remove from current zone
                        zone.remove_card(action.card)

                        # Add to exile zone (player already has exile zone)
                        player.exile.add_card(action.card)
                        action.card.zone = "exile"

                        # Reset creature states if coming from battlefield
                        if zone_name == "battlefield":
                            action.card.attacking = False
                            action.card.blocking = None
                            action.card.tapped = False

                        break

        # Add to game history
        new_state.history.append(
            {
                "action": "exile",
                "player": action.player_id,
                "card": action.card.name if action.card else "unknown",
                "turn": new_state.turn_number,
            }
        )

        return new_state


# Registry of executors
ACTION_EXECUTORS = {
    ActionType.PLAY_LAND: PlayLandExecutor(),
    ActionType.CAST_SPELL: CastSpellExecutor(),
    ActionType.PASS_PRIORITY: PassPriorityExecutor(),
    ActionType.DECLARE_ATTACKERS: DeclareAttackersExecutor(),
    ActionType.DECLARE_BLOCKERS: DeclareBlockersExecutor(),
    ActionType.ASSIGN_COMBAT_DAMAGE: AssignCombatDamageExecutor(),
    ActionType.ORDER_BLOCKERS: OrderBlockersExecutor(),
    ActionType.ACTIVATE_ABILITY: ActivateAbilityExecutor(),
    ActionType.ACTIVATE_MANA_ABILITY: ActivateManaAbilityExecutor(),
    ActionType.MULLIGAN: MulliganExecutor(),
    ActionType.KEEP_HAND: KeepHandExecutor(),
    ActionType.CONCEDE: ConcedeExecutor(),
    ActionType.DISCARD: DiscardExecutor(),
    ActionType.SACRIFICE: SacrificeExecutor(),
    ActionType.DESTROY: DestroyExecutor(),
    ActionType.EXILE: ExileExecutor(),
    ActionType.CHOOSE_TARGET: ChooseTargetExecutor(),
    ActionType.CHOOSE_MODE: ChooseModeExecutor(),
    ActionType.CHOOSE_X_VALUE: ChooseXValueExecutor(),
    ActionType.TAP_FOR_MANA: TapForManaExecutor(),
    ActionType.PAY_MANA: PayManaExecutor(),
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

        # Check for combat actions during combat phase
        if game_state.phase == "combat":
            # Declare attackers (active player only, at start of combat)
            if game_state.active_player == current_player_id and not any(
                card.attacking for card in current_player.battlefield.cards
            ):
                # Generate attacker combinations
                potential_attackers = []
                for card in current_player.battlefield.cards:
                    if (
                        card.is_creature()
                        and not card.tapped
                        and not card.summoning_sick
                        and "Defender" not in card.keywords
                        and card.instance_id is not None
                    ):
                        potential_attackers.append(card.instance_id)

                if potential_attackers:
                    # Add option to attack with any combination of creatures
                    # For simplicity, add actions for individual and all
                    # attackers
                    for attacker_id in potential_attackers:
                        action = Action(
                            action_type=ActionType.DECLARE_ATTACKERS,
                            player_id=current_player_id,
                            attackers=[attacker_id],
                        )
                        if action.is_valid(game_state):
                            legal_actions.append(action)

                    # Option to attack with all creatures
                    action = Action(
                        action_type=ActionType.DECLARE_ATTACKERS,
                        player_id=current_player_id,
                        attackers=potential_attackers,
                    )
                    if action.is_valid(game_state):
                        legal_actions.append(action)

            # Declare blockers (defending player only)
            elif game_state.active_player != current_player_id and any(
                card.attacking
                for card in game_state.players[
                    1 - current_player_id
                ].battlefield.cards
            ):
                # Find attacking creatures
                attackers = []
                for card in game_state.players[
                    1 - current_player_id
                ].battlefield.cards:
                    if card.attacking and card.instance_id is not None:
                        attackers.append(card.instance_id)

                # Find potential blockers
                potential_blockers = []
                for card in current_player.battlefield.cards:
                    if (
                        card.is_creature()
                        and not card.tapped
                        and "Can't block" not in card.oracle_text
                        and card.instance_id is not None
                    ):
                        potential_blockers.append(card.instance_id)

                # Generate blocking options (simplified - just block or don't
                # block)
                if potential_blockers and attackers:
                    # Option to not block anything
                    action = Action(
                        action_type=ActionType.DECLARE_BLOCKERS,
                        player_id=current_player_id,
                        blockers={},
                    )
                    if action.is_valid(game_state):
                        legal_actions.append(action)

                    # Option to block each attacker with one blocker
                    for attacker_id in attackers:
                        for blocker_id in potential_blockers:
                            action = Action(
                                action_type=ActionType.DECLARE_BLOCKERS,
                                player_id=current_player_id,
                                blockers={attacker_id: [blocker_id]},
                            )
                            if action.is_valid(game_state):
                                legal_actions.append(action)

        # TODO: Add more action types (abilities, etc.)

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
