"""Tests for agent implementations."""

import torch
from unittest.mock import Mock, patch

from manamind.core.action import Action, ActionType
from manamind.core.agent import (
    Agent,
    MCTSAgent,
    MCTSNode,
    NeuralAgent,
    RandomAgent,
)
from manamind.core.game_state import Card, create_empty_game_state


class TestAgent:
    """Test Agent base class."""

    def test_agent_creation(self):
        """Test agent creation with player ID."""
        agent = RandomAgent(player_id=0)
        assert agent.player_id == 0

    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        # Should not be able to instantiate base Agent class
        try:
            agent = Agent(0)
            agent.select_action(create_empty_game_state())
            assert False, "Should have raised NotImplementedError"
        except NotImplementedError:
            pass  # Expected


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
        land = Card(name="Mountain", card_types=["Land"])
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

    def test_random_agent_deterministic_with_seed(self):
        """Test that random agent is deterministic with same seed."""
        game_state = create_empty_game_state()
        
        # Add cards to player's hand
        land = Card(name="Mountain", card_types=["Land"])
        game_state.players[0].hand.add_card(land)
        
        # Set up game state
        game_state.active_player = 0
        game_state.priority_player = 0
        game_state.phase = "main"

        # Create two agents with same seed
        agent1 = RandomAgent(player_id=0, seed=123)
        agent2 = RandomAgent(player_id=0, seed=123)
        
        action1 = agent1.select_action(game_state)
        action2 = agent2.select_action(game_state)
        
        # Should select the same action
        assert action1.action_type == action2.action_type


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
        land = Card(name="Mountain", card_types=["Land"])
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

    def test_mcts_node_select_child(self):
        """Test selecting the best child node."""
        game_state = create_empty_game_state()
        
        # Add cards to create legal actions
        land = Card(name="Mountain", card_types=["Land"])
        game_state.players[0].hand.add_card(land)
        
        # Set up game state
        game_state.active_player = 0
        game_state.priority_player = 0
        game_state.phase = "main"

        node = MCTSNode(game_state)
        
        # Expand to create children
        child1 = node.expand()
        child2 = node.expand()
        
        # Backup different values
        child1.backup(0.8)
        child2.backup(0.3)
        
        # Select best child (should be child1 with higher value)
        best_child = node.select_child()
        assert best_child == child1

    def test_mcts_node_get_action_probabilities(self):
        """Test getting action probabilities from node."""
        game_state = create_empty_game_state()
        node = MCTSNode(game_state)
        
        # Should return empty dict for node with no children
        probs = node.get_action_probabilities()
        assert isinstance(probs, dict)
        assert len(probs) == 0


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
        land = Card(name="Mountain", card_types=["Land"])
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

    def test_mcts_agent_with_custom_parameters(self):
        """Test MCTS agent with custom parameters."""
        agent = MCTSAgent(
            player_id=1, 
            simulations=500, 
            simulation_time=2.0,
            exploration_weight=2.0
        )
        assert agent.player_id == 1
        assert agent.simulations == 500
        assert agent.simulation_time == 2.0
        assert agent.exploration_weight == 2.0


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
        # Create a mock network that returns policy and value
        class MockNetwork:
            def __call__(self, game_state):
                # Return uniform policy and zero value
                import torch
                policy = torch.ones(10000) / 10000  # Uniform distribution
                value = torch.tensor(0.0)
                return policy, value

        network = MockNetwork()
        agent = NeuralAgent(player_id=0, policy_value_network=network)
        game_state = create_empty_game_state()

        # Add a land to player's hand to have legal actions
        land = Card(name="Mountain", card_types=["Land"])
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

    def test_neural_agent_with_temperature(self):
        """Test neural agent with different temperature settings."""
        # Create a mock network
        class MockNetwork:
            def __call__(self, game_state):
                import torch
                # Return skewed policy to test temperature effect
                policy = torch.zeros(10000)
                policy[0] = 0.9
                policy[1] = 0.1
                value = torch.tensor(0.0)
                return policy, value

        network = MockNetwork()
        
        # Test with high temperature (more random)
        agent_high_temp = NeuralAgent(
            player_id=0, 
            policy_value_network=network,
            temperature=2.0
        )
        
        # Test with low temperature (more deterministic)
        agent_low_temp = NeuralAgent(
            player_id=0, 
            policy_value_network=network,
            temperature=0.1
        )
        
        # Both should be able to select actions
        game_state = create_empty_game_state()
        land = Card(name="Mountain", card_types=["Land"])
        game_state.players[0].hand.add_card(land)
        game_state.active_player = 0
        game_state.priority_player = 0
        game_state.phase = "main"
        
        action_high = agent_high_temp.select_action(game_state)
        action_low = agent_low_temp.select_action(game_state)
        
        assert isinstance(action_high, Action)
        assert isinstance(action_low, Action)


class TestAgentIntegration:
    """Integration tests for agent classes."""

    def test_agent_action_validity(self):
        """Test that agents select valid actions."""
        game_state = create_empty_game_state()
        
        # Add cards to player's hand
        land = Card(name="Mountain", card_types=["Land"])
        spell = Card(
            name="Lightning Bolt",
            card_types=["Instant"],
            converted_mana_cost=1
        )
        game_state.players[0].hand.add_card(land)
        game_state.players[0].hand.add_card(spell)
        
        # Set up game state
        game_state.active_player = 0
        game_state.priority_player = 0
        game_state.phase = "main"
        game_state.players[0].mana_pool = {"R": 1}

        # Test with different agents
        agents = [
            RandomAgent(player_id=0, seed=42),
            MCTSAgent(player_id=0, simulations=5),
        ]

        for agent in agents:
            action = agent.select_action(game_state)
            assert isinstance(action, Action)
            # Action should be valid
            assert action.is_valid(game_state) is True

    def test_agent_player_id_consistency(self):
        """Test that agents respect their player ID."""
        game_state = create_empty_game_state()
        
        # Add cards to both players' hands
        land_p0 = Card(name="Mountain", card_types=["Land"])
        land_p1 = Card(name="Forest", card_types=["Land"])
        game_state.players[0].hand.add_card(land_p0)
        game_state.players[1].hand.add_card(land_p1)
        
        # Set up game state
        game_state.active_player = 0
        game_state.priority_player = 0
        game_state.phase = "main"

        # Create agents for each player
        agent0 = RandomAgent(player_id=0, seed=42)
        agent1 = RandomAgent(player_id=1, seed=24)

        action0 = agent0.select_action(game_state)
        action1 = agent1.select_action(game_state)

        # Actions should have correct player IDs
        assert action0.player_id == 0
        assert action1.player_id == 1