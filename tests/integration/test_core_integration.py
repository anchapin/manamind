"""Integration tests for core components."""

import torch

from manamind.core.action import Action, ActionSpace, ActionType
from manamind.core.agent import RandomAgent
from manamind.core.game_state import (
    Card,
    create_empty_game_state,
)


class TestCoreIntegration:
    """Integration tests for core components."""

    def test_game_state_action_integration(self):
        """Test integration between game state and actions."""
        # Create a game state
        game_state = create_empty_game_state()

        # Add a land to player's hand
        land = Card(name="Mountain", card_type="Land")
        game_state.players[0].hand.add_card(land)

        # Set up game state for land play
        game_state.active_player = 0
        game_state.priority_player = 0
        game_state.phase = "main"

        # Create a play land action
        action = Action(
            action_type=ActionType.PLAY_LAND,
            player_id=0,
            card=land,
        )

        # Verify the action is valid
        assert action.is_valid(game_state) is True

        # Execute the action
        new_state = action.execute(game_state)

        # Verify the land was moved to battlefield
        assert land not in new_state.players[0].hand.cards
        assert land in new_state.players[0].battlefield.cards
        assert new_state.players[0].lands_played_this_turn == 1

    def test_action_space_integration(self):
        """Test integration with action space."""
        # Create a game state
        game_state = create_empty_game_state()

        # Add cards to player's hand
        land = Card(name="Mountain", card_type="Land")
        spell = Card(
            name="Lightning Bolt",
            card_type="Instant",
            converted_mana_cost=1,
        )
        game_state.players[0].hand.add_card(land)
        game_state.players[0].hand.add_card(spell)

        # Set up game state
        game_state.active_player = 0
        game_state.priority_player = 0
        game_state.phase = "main"
        game_state.players[0].mana_pool = {"R": 1}

        # Get legal actions
        action_space = ActionSpace()
        legal_actions = action_space.get_legal_actions(game_state)

        # Should have at least pass priority and play land actions
        assert len(legal_actions) >= 2

        # Verify we can find the play land action
        play_land_actions = [
            action
            for action in legal_actions
            if action.action_type == ActionType.PLAY_LAND
        ]
        assert len(play_land_actions) == 1
        assert play_land_actions[0].card == land

    def test_agent_game_state_integration(self):
        """Test integration between agents and game state."""
        # Create a game state
        game_state = create_empty_game_state()

        # Add a land to player's hand
        land = Card(name="Mountain", card_type="Land")
        game_state.players[0].hand.add_card(land)

        # Set up game state for land play
        game_state.active_player = 0
        game_state.priority_player = 0
        game_state.phase = "main"

        # Create a random agent
        agent = RandomAgent(player_id=0, seed=42)

        # Select an action
        action = agent.select_action(game_state)

        # Verify it's a valid action
        assert isinstance(action, Action)
        assert action.player_id == 0
        assert action.is_valid(game_state) is True

    def test_full_game_simulation(self):
        """Test a simple game simulation."""
        # Create a game state
        game_state = create_empty_game_state()

        # Add cards to both players' hands
        land_p0 = Card(name="Mountain", card_type="Land")
        spell_p0 = Card(
            name="Lightning Bolt",
            card_type="Instant",
            converted_mana_cost=1,
        )
        game_state.players[0].hand.add_card(land_p0)
        game_state.players[0].hand.add_card(spell_p0)

        land_p1 = Card(name="Forest", card_type="Land")
        game_state.players[1].hand.add_card(land_p1)

        # Set up initial game state
        game_state.active_player = 0
        game_state.priority_player = 0
        game_state.phase = "main"
        game_state.players[0].mana_pool = {"R": 1}

        # Create agents
        agent0 = RandomAgent(player_id=0, seed=42)
        agent1 = RandomAgent(player_id=1, seed=24)

        # Simulate a few turns
        for turn in range(3):
            # Player 0's turn
            game_state.active_player = 0
            game_state.priority_player = 0

            # Main phase 1
            game_state.phase = "main"
            action = agent0.select_action(game_state)
            if action.action_type != ActionType.PASS_PRIORITY:
                game_state = action.execute(game_state)

            # Pass priority to end turn
            game_state.priority_player = 1
            action = Action(
                action_type=ActionType.PASS_PRIORITY,
                player_id=1,
            )
            game_state = action.execute(game_state)

            # Player 1's turn
            game_state.active_player = 1
            game_state.priority_player = 1

            # Main phase 1
            game_state.phase = "main"
            action = agent1.select_action(game_state)
            if action.action_type != ActionType.PASS_PRIORITY:
                game_state = action.execute(game_state)

            # Pass priority to end turn
            game_state.priority_player = 0
            action = Action(
                action_type=ActionType.PASS_PRIORITY,
                player_id=0,
            )
            game_state = action.execute(game_state)

            # Increment turn
            game_state.turn_number += 1

    def test_neural_network_integration(self):
        """Test integration with neural networks."""
        # Create a game state
        game_state = create_empty_game_state()

        # Add some cards to make it more realistic
        for player in game_state.players:
            for i in range(2):
                card = Card(name=f"Card {i}", card_id=i + 1)
                player.hand.add_card(card)

        # Create a mock neural network
        class MockNetwork(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(10, 10)

            def forward(self, game_state):
                # Simple mock implementation
                return torch.tensor([0.0]), torch.tensor(0.0)

        # Test that we can create agents with neural networks
        network = MockNetwork()
        from manamind.core.agent import NeuralAgent

        agent = NeuralAgent(
            player_id=0,
            policy_value_network=network,
        )

        # Should be able to select an action
        action = agent.select_action(game_state)
        assert isinstance(action, Action)
        assert action.player_id == 0
