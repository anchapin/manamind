"""Tests for agent implementations."""

import pytest
import random

from manamind.core.agent import (
    Agent,
    RandomAgent,
    MCTSNode,
    MCTSAgent,
    NeuralAgent,
)
from manamind.core.action import Action, ActionType
from manamind.core.game_state import create_empty_game_state


class TestAgent:
    """Test Agent base class."""

    def test_agent_creation(self):
        """Test agent creation with player ID."""
        agent = RandomAgent(player_id=0)
        assert agent.player_id == 0


class TestRandomAgent:
    """Test RandomAgent implementation."""

    def test_random_agent_creation(self):
        """Test random agent creation."""
        agent = RandomAgent(player_id=1, seed=42)
        assert agent.player_id == 1
        assert agent.rng is not None

    def test_random_agent_select_action(self):
        """Test random agent action selection."""
        agent = RandomAgent(player_id=0, seed=42)
        game_state = create_empty_game_state()
        
        # Add a land to player's hand to have legal actions
        from manamind.core.game_state import Card
        land = Card(name="Mountain", card_type="Land")
        game_state.players[0].hand.add_card(land)
        
        # Set up game state for land play
        game_state.active_player = 0
        game_state.priority_player = 0
        game_state.phase = "main"
        
        action = agent.select_action(game_state)
        assert isinstance(action, Action)
        assert action.player_id == 0

    def test_random_agent_update_from_game(self):
        """Test that random agent update method exists."""
        agent = RandomAgent(player_id=0)
        game_history = []
        agent.update_from_game(game_history)  # Should not raise


class TestMCTSNode:
    """Test MCTSNode implementation."""

    def test_mcts_node_creation(self):
        """Test MCTS node creation."""
        game_state = create_empty_game_state()
        node = MCTSNode(game_state)
        
        assert node.game_state == game_state
        assert node.action is None
        assert node.parent is None
        assert node.visits == 0
        assert node.total_value == 0.0
        assert node.prior_prob == 1.0

    def test_mcts_node_is_fully_expanded(self):
        """Test checking if node is fully expanded."""
        game_state = create_empty_game_state()
        node = MCTSNode(game_state)
        
        # Initially should not be fully expanded (has legal actions)
        assert node.is_fully_expanded() is False

    def test_mcts_node_is_terminal(self):
        """Test checking if node is terminal."""
        game_state = create_empty_game_state()
        node = MCTSNode(game_state)
        
        # Normal game state should not be terminal
        assert node.is_terminal() is False
        
        # Game over state should be terminal
        game_state.players[0].life = 0
        assert node.is_terminal() is True

    def test_mcts_node_ucb1_score(self):
        """Test UCB1 score calculation."""
        game_state = create_empty_game_state()
        parent_node = MCTSNode(game_state)
        parent_node.visits = 2  # Parent needs visits for exploration term
        
        # Create a child node
        child_node = MCTSNode(game_state)
        
        # Child with no visits should have infinite score
        score = parent_node.ucb1_score(child_node)
        assert score == float("inf")
        
        # Child with visits should have finite score
        child_node.visits = 1
        child_node.total_value = 0.5
        score = parent_node.ucb1_score(child_node)
        assert isinstance(score, float)
        assert score != float("inf")

    def test_mcts_node_expand(self):
        """Test expanding the node."""
        game_state = create_empty_game_state()
        
        # Add a land to player's hand to have legal actions
        from manamind.core.game_state import Card
        land = Card(name="Mountain", card_type="Land")
        game_state.players[0].hand.add_card(land)
        
        # Set up game state for land play
        game_state.active_player = 0
        game_state.priority_player = 0
        game_state.phase = "main"
        
        node = MCTSNode(game_state)
        child = node.expand()
        
        assert isinstance(child, MCTSNode)
        assert child.parent == node
        assert child.action is not None
        assert len(node.children) == 1

    def test_mcts_node_backup(self):
        """Test backing up values through the tree."""
        game_state = create_empty_game_state()
        root = MCTSNode(game_state)
        child = root.expand()
        
        # Backup a value
        child.backup(0.5)
        
        # Check that visits and values were updated
        assert child.visits == 1
        assert child.total_value == 0.5
        assert root.visits == 1
        assert root.total_value == -0.5  # Flipped for opponent


class TestMCTSAgent:
    """Test MCTSAgent implementation."""

    def test_mcts_agent_creation(self):
        """Test MCTS agent creation."""
        agent = MCTSAgent(player_id=0)
        assert agent.player_id == 0
        assert agent.simulations == 1000
        assert agent.simulation_time == 1.0

    def test_mcts_agent_select_action(self):
        """Test MCTS agent action selection."""
        agent = MCTSAgent(player_id=0, simulations=10)
        game_state = create_empty_game_state()
        
        # Add a land to player's hand to have legal actions
        from manamind.core.game_state import Card
        land = Card(name="Mountain", card_type="Land")
        game_state.players[0].hand.add_card(land)
        
        # Set up game state for land play
        game_state.active_player = 0
        game_state.priority_player = 0
        game_state.phase = "main"
        
        action = agent.select_action(game_state)
        assert isinstance(action, Action)
        assert action.player_id == 0

    def test_mcts_agent_update_from_game(self):
        """Test that MCTS agent update method exists."""
        agent = MCTSAgent(player_id=0)
        game_history = []
        agent.update_from_game(game_history)  # Should not raise


class TestNeuralAgent:
    """Test NeuralAgent implementation."""

    def test_neural_agent_creation(self):
        """Test neural agent creation."""
        # Create a mock network
        class MockNetwork:
            pass
            
        network = MockNetwork()
        agent = NeuralAgent(player_id=1, policy_value_network=network)
        assert agent.player_id == 1
        assert agent.policy_value_network == network

    def test_neural_agent_select_action(self):
        """Test neural agent action selection."""
        # Create a mock network
        class MockNetwork:
            def __call__(self, game_state):
                import torch
                return torch.tensor([0.0]), torch.tensor(0.0)
        
        network = MockNetwork()
        agent = NeuralAgent(player_id=0, policy_value_network=network)
        game_state = create_empty_game_state()
        
        # Add a land to player's hand to have legal actions
        from manamind.core.game_state import Card
        land = Card(name="Mountain", card_type="Land")
        game_state.players[0].hand.add_card(land)
        
        # Set up game state for land play
        game_state.active_player = 0
        game_state.priority_player = 0
        game_state.phase = "main"
        
        action = agent.select_action(game_state)
        assert isinstance(action, Action)
        assert action.player_id == 0

    def test_neural_agent_update_from_game(self):
        """Test that neural agent update method exists."""
        # Create a mock network
        class MockNetwork:
            pass
            
        network = MockNetwork()
        agent = NeuralAgent(player_id=0, policy_value_network=network)
        game_history = []
        agent.update_from_game(game_history)  # Should not raise