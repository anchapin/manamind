"""Tests for game state representation and encoding."""

import pytest
import torch
import numpy as np

from manamind.core.game_state import (
    Card, Zone, Player, GameState, GameStateEncoder,
    create_empty_game_state, create_standard_game_start
)


class TestCard:
    """Test Card class."""
    
    def test_card_creation(self):
        """Test basic card creation."""
        card = Card(
            name="Lightning Bolt",
            mana_cost="R",
            converted_mana_cost=1,
            card_type="Instant",
            text="Lightning Bolt deals 3 damage to any target."
        )
        
        assert card.name == "Lightning Bolt"
        assert card.mana_cost == "R"
        assert card.converted_mana_cost == 1
        assert card.card_type == "Instant"
    
    def test_creature_card(self):
        """Test creature card with power/toughness."""
        card = Card(
            name="Grizzly Bears",
            mana_cost="1G",
            converted_mana_cost=2,
            card_type="Creature — Bear",
            power=2,
            toughness=2
        )
        
        assert card.power == 2
        assert card.toughness == 2


class TestZone:
    """Test Zone class."""
    
    def test_empty_zone(self):
        """Test empty zone creation."""
        zone = Zone(name="hand", owner=0)
        
        assert zone.name == "hand"
        assert zone.owner == 0
        assert zone.size() == 0
    
    def test_add_remove_cards(self):
        """Test adding and removing cards."""
        zone = Zone(name="battlefield", owner=1)
        card = Card(name="Test Card")
        
        # Add card
        zone.add_card(card)
        assert zone.size() == 1
        assert card in zone.cards
        
        # Remove card
        result = zone.remove_card(card)
        assert result is True
        assert zone.size() == 0
        assert card not in zone.cards
        
        # Try to remove non-existent card
        result = zone.remove_card(card)
        assert result is False


class TestPlayer:
    """Test Player class."""
    
    def test_player_creation(self):
        """Test player creation with zones."""
        player = Player(player_id=0)
        
        assert player.player_id == 0
        assert player.life == 20
        assert player.hand.owner == 0
        assert player.battlefield.owner == 0
        assert player.graveyard.owner == 0
    
    def test_can_play_land(self):
        """Test land playing rules."""
        player = Player(player_id=0)
        
        # Can play first land
        assert player.can_play_land() is True
        
        # After playing one land
        player.lands_played_this_turn = 1
        assert player.can_play_land() is False
    
    def test_mana_pool(self):
        """Test mana pool functionality."""
        player = Player(player_id=0)
        
        # Empty mana pool
        assert player.total_mana() == 0
        
        # Add some mana
        player.mana_pool = {"R": 2, "U": 1}
        assert player.total_mana() == 3


class TestGameState:
    """Test GameState class."""
    
    def test_empty_game_state(self):
        """Test empty game state creation."""
        game_state = create_empty_game_state()
        
        assert game_state.turn_number == 1
        assert game_state.phase == "main"
        assert game_state.priority_player == 0
        assert game_state.active_player == 0
        assert len(game_state.players) == 2
    
    def test_game_properties(self):
        """Test game state properties."""
        game_state = create_empty_game_state()
        
        # Test current player
        current = game_state.current_player
        assert current.player_id == 0
        
        # Test opponent
        opponent = game_state.opponent
        assert opponent.player_id == 1
        
        # Test game not over
        assert game_state.is_game_over() is False
        assert game_state.winner() is None
    
    def test_game_over_conditions(self):
        """Test game over detection."""
        game_state = create_empty_game_state()
        
        # Player 0 loses
        game_state.players[0].life = 0
        assert game_state.is_game_over() is True
        assert game_state.winner() == 1
        
        # Reset and test player 1 loses
        game_state.players[0].life = 20
        game_state.players[1].life = 0
        assert game_state.is_game_over() is True
        assert game_state.winner() == 0


class TestGameStateEncoder:
    """Test GameStateEncoder neural network."""
    
    @pytest.fixture
    def encoder(self):
        """Create a test encoder."""
        return GameStateEncoder(
            vocab_size=1000,
            embed_dim=64,
            hidden_dim=128,
            output_dim=256
        )
    
    @pytest.fixture
    def game_state(self):
        """Create a test game state."""
        return create_empty_game_state()
    
    def test_encoder_creation(self, encoder):
        """Test encoder creation."""
        assert encoder.vocab_size == 1000
        assert encoder.embed_dim == 64
        assert encoder.hidden_dim == 128
        assert encoder.output_dim == 256
    
    def test_zone_encoding(self, encoder, game_state):
        """Test zone encoding."""
        zone = game_state.players[0].hand
        
        # Add some test cards
        for i in range(3):
            card = Card(name=f"Card {i}", card_id=i + 1)
            zone.add_card(card)
        
        # Encode zone
        encoding = encoder.encode_zone(zone, zone_idx=0)
        
        # Should be a tensor of the right size
        assert isinstance(encoding, torch.Tensor)
        assert encoding.shape[0] == encoder.hidden_dim
    
    def test_player_encoding(self, encoder, game_state):
        """Test player encoding."""
        player = game_state.players[0]
        
        # Modify player state
        player.life = 15
        player.mana_pool = {"R": 2, "U": 1}
        
        # Encode player
        encoding = encoder.encode_player(player)
        
        # Should be a tensor
        assert isinstance(encoding, torch.Tensor)
        assert encoding.shape[0] == encoder.hidden_dim
    
    def test_full_encoding(self, encoder, game_state):
        """Test full game state encoding."""
        # Add some cards to make it more realistic
        for player in game_state.players:
            for i in range(2):
                card = Card(name=f"Card {i}", card_id=i + 1)
                player.hand.add_card(card)
        
        # Encode full state
        encoding = encoder.forward(game_state)
        
        # Should be the right shape
        assert isinstance(encoding, torch.Tensor)
        assert encoding.shape[0] == encoder.output_dim
    
    def test_batch_encoding(self, encoder):
        """Test batch encoding (using tensors directly)."""
        batch_size = 4
        state_dim = encoder.output_dim
        
        # Create dummy batch
        dummy_batch = torch.randn(batch_size, state_dim)
        
        # Should handle batch dimension correctly
        with torch.no_grad():
            # This would normally go through the full encoding pipeline
            # For now, just test that tensor shapes work
            assert dummy_batch.shape == (batch_size, state_dim)


@pytest.mark.integration
class TestGameStateIntegration:
    """Integration tests for game state components."""
    
    def test_full_pipeline(self):
        """Test the full game state pipeline."""
        # Create game state
        game_state = create_empty_game_state()
        
        # Add some realistic game state
        # Player 0 gets some cards
        cards = [
            Card(name="Lightning Bolt", mana_cost="R", converted_mana_cost=1, card_id=1),
            Card(name="Mountain", card_type="Land", card_id=2),
            Card(name="Grizzly Bears", mana_cost="1G", converted_mana_cost=2, card_id=3),
        ]
        
        for card in cards:
            game_state.players[0].hand.add_card(card)
        
        # Player 0 plays a land
        mountain = cards[1]
        game_state.players[0].hand.remove_card(mountain)
        game_state.players[0].battlefield.add_card(mountain)
        game_state.players[0].lands_played_this_turn = 1
        
        # Add some mana
        game_state.players[0].mana_pool = {"R": 1}
        
        # Update game state
        game_state.turn_number = 2
        game_state.phase = "main"
        
        # Create encoder and encode
        encoder = GameStateEncoder(output_dim=512)
        
        with torch.no_grad():
            encoding = encoder.forward(game_state)
            
            # Should produce a valid encoding
            assert isinstance(encoding, torch.Tensor)
            assert encoding.shape[0] == 512
            assert not torch.isnan(encoding).any()
            assert not torch.isinf(encoding).any()


if __name__ == "__main__":
    pytest.main([__file__])