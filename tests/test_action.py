"""Tests for action representation and validation."""

import pytest
import torch

from manamind.core.action import (
    Action,
    ActionType,
    ActionSpace,
    PlayLandValidator,
    CastSpellValidator,
    PassPriorityValidator,
    PlayLandExecutor,
    CastSpellExecutor,
    PassPriorityExecutor,
)
from manamind.core.game_state import (
    Card,
    create_empty_game_state,
    Player,
)


class TestActionType:
    """Test ActionType enum."""

    def test_action_type_values(self):
        """Test that all action types have correct values."""
        assert ActionType.PLAY_LAND.value == "play_land"
        assert ActionType.CAST_SPELL.value == "cast_spell"
        assert ActionType.PASS_PRIORITY.value == "pass_priority"


class TestAction:
    """Test Action class."""

    def test_action_creation(self):
        """Test basic action creation."""
        card = Card(name="Lightning Bolt")
        action = Action(
            action_type=ActionType.CAST_SPELL,
            player_id=0,
            card=card,
        )

        assert action.action_type == ActionType.CAST_SPELL
        assert action.player_id == 0
        assert action.card == card
        assert action.timestamp is not None

    def test_action_complexity_score(self):
        """Test action complexity scoring."""
        # Simple action
        simple_action = Action(
            action_type=ActionType.PASS_PRIORITY,
            player_id=0,
        )
        assert simple_action.get_complexity_score() == 1

        # Action with targets
        card = Card(name="Lightning Bolt")
        targeted_action = Action(
            action_type=ActionType.CAST_SPELL,
            player_id=0,
            card=card,
            target_cards=[card],
        )
        assert targeted_action.get_complexity_score() == 2

    def test_action_targets(self):
        """Test getting all targets from an action."""
        card1 = Card(name="Lightning Bolt")
        card2 = Card(name="Grizzly Bears")
        
        action = Action(
            action_type=ActionType.CAST_SPELL,
            player_id=0,
            card=card1,
            target_cards=[card2],
            target_players=[1],
        )
        
        targets = action.get_all_targets()
        assert card1 in targets
        assert card2 in targets
        assert 1 in targets


class TestPlayLandValidator:
    """Test PlayLandValidator."""

    def test_valid_land_play(self):
        """Test valid land play scenario."""
        game_state = create_empty_game_state()
        
        # Add a land to player's hand
        land = Card(name="Mountain", card_type="Land")
        game_state.players[0].hand.add_card(land)
        
        action = Action(
            action_type=ActionType.PLAY_LAND,
            player_id=0,
            card=land,
        )
        
        validator = PlayLandValidator()
        assert validator.validate(action, game_state) is True

    def test_invalid_land_play_wrong_zone(self):
        """Test land play with card not in hand."""
        game_state = create_empty_game_state()
        
        # Put land on battlefield instead of hand
        land = Card(name="Mountain", card_type="Land")
        game_state.players[0].battlefield.add_card(land)
        
        action = Action(
            action_type=ActionType.PLAY_LAND,
            player_id=0,
            card=land,
        )
        
        validator = PlayLandValidator()
        assert validator.validate(action, game_state) is False

    def test_invalid_land_play_not_land(self):
        """Test land play with non-land card."""
        game_state = create_empty_game_state()
        
        # Add non-land to player's hand
        spell = Card(name="Lightning Bolt", card_type="Instant")
        game_state.players[0].hand.add_card(spell)
        
        action = Action(
            action_type=ActionType.PLAY_LAND,
            player_id=0,
            card=spell,
        )
        
        validator = PlayLandValidator()
        assert validator.validate(action, game_state) is False


class TestCastSpellValidator:
    """Test CastSpellValidator."""

    def test_valid_spell_cast_sorcery(self):
        """Test valid sorcery cast."""
        game_state = create_empty_game_state()
        
        # Add spell to player's hand
        spell = Card(
            name="Lightning Bolt",
            card_type="Sorcery",
            converted_mana_cost=1,
        )
        game_state.players[0].hand.add_card(spell)
        
        # Add mana
        game_state.players[0].mana_pool = {"R": 1}
        
        action = Action(
            action_type=ActionType.CAST_SPELL,
            player_id=0,
            card=spell,
        )
        
        validator = CastSpellValidator()
        assert validator.validate(action, game_state) is True

    def test_invalid_spell_cast_no_mana(self):
        """Test spell cast without enough mana."""
        game_state = create_empty_game_state()
        
        # Add spell to player's hand
        spell = Card(
            name="Lightning Bolt",
            card_type="Instant",
            converted_mana_cost=1,
        )
        game_state.players[0].hand.add_card(spell)
        
        # No mana in pool
        game_state.players[0].mana_pool = {}
        
        action = Action(
            action_type=ActionType.CAST_SPELL,
            player_id=0,
            card=spell,
        )
        
        validator = CastSpellValidator()
        assert validator.validate(action, game_state) is False


class TestPassPriorityValidator:
    """Test PassPriorityValidator."""

    def test_valid_priority_pass(self):
        """Test valid priority pass."""
        game_state = create_empty_game_state()
        
        action = Action(
            action_type=ActionType.PASS_PRIORITY,
            player_id=0,
        )
        
        validator = PassPriorityValidator()
        assert validator.validate(action, game_state) is True

    def test_invalid_priority_pass(self):
        """Test priority pass when not having priority."""
        game_state = create_empty_game_state()
        game_state.priority_player = 1  # Other player has priority
        
        action = Action(
            action_type=ActionType.PASS_PRIORITY,
            player_id=0,
        )
        
        validator = PassPriorityValidator()
        assert validator.validate(action, game_state) is False


class TestPlayLandExecutor:
    """Test PlayLandExecutor."""

    def test_execute_land_play(self):
        """Test executing a land play."""
        game_state = create_empty_game_state()
        
        # Add a land to player's hand
        land = Card(name="Mountain", card_type="Land")
        game_state.players[0].hand.add_card(land)
        
        action = Action(
            action_type=ActionType.PLAY_LAND,
            player_id=0,
            card=land,
        )
        
        executor = PlayLandExecutor()
        new_state = executor.execute(action, game_state)
        
        # Check that land was moved to battlefield
        assert land not in new_state.players[0].hand.cards
        assert land in new_state.players[0].battlefield.cards
        assert new_state.players[0].lands_played_this_turn == 1


class TestCastSpellExecutor:
    """Test CastSpellExecutor."""

    def test_execute_spell_cast(self):
        """Test executing a spell cast."""
        game_state = create_empty_game_state()
        
        # Add spell to player's hand
        spell = Card(name="Lightning Bolt", card_type="Instant")
        game_state.players[0].hand.add_card(spell)
        
        action = Action(
            action_type=ActionType.CAST_SPELL,
            player_id=0,
            card=spell,
        )
        
        executor = CastSpellExecutor()
        new_state = executor.execute(action, game_state)
        
        # Check that spell was moved to stack
        assert spell not in new_state.players[0].hand.cards
        assert len(new_state.stack) == 1


class TestPassPriorityExecutor:
    """Test PassPriorityExecutor."""

    def test_execute_pass_priority(self):
        """Test executing priority pass."""
        game_state = create_empty_game_state()
        game_state.priority_player = 0
        
        action = Action(
            action_type=ActionType.PASS_PRIORITY,
            player_id=0,
        )
        
        executor = PassPriorityExecutor()
        new_state = executor.execute(action, game_state)
        
        # Check that priority was passed
        assert new_state.priority_player == 1


class TestActionSpace:
    """Test ActionSpace class."""

    def test_action_space_creation(self):
        """Test action space creation."""
        action_space = ActionSpace()
        
        assert action_space.max_actions == 10000
        assert len(action_space.action_to_id) > 0
        assert len(action_space.id_to_action) > 0

    def test_get_legal_actions(self):
        """Test getting legal actions from game state."""
        game_state = create_empty_game_state()
        
        # Add a land to player's hand
        land = Card(name="Mountain", card_type="Land")
        game_state.players[0].hand.add_card(land)
        
        # Set up game state for land play
        game_state.active_player = 0
        game_state.priority_player = 0
        game_state.phase = "main"
        
        action_space = ActionSpace()
        legal_actions = action_space.get_legal_actions(game_state)
        
        # Should have at least pass priority and play land actions
        assert len(legal_actions) >= 2
        
        # Check for play land action
        play_land_actions = [
            action for action in legal_actions 
            if action.action_type == ActionType.PLAY_LAND
        ]
        assert len(play_land_actions) == 1
        assert play_land_actions[0].card == land

    def test_action_to_vector(self):
        """Test converting action to vector."""
        action_space = ActionSpace()
        
        action = Action(
            action_type=ActionType.PASS_PRIORITY,
            player_id=0,
        )
        
        vector = action_space.action_to_vector(action)
        
        assert isinstance(vector, list)
        assert len(vector) == action_space.max_actions
        assert sum(vector) <= 1  # At most one element should be 1.0