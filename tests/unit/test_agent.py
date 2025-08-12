"""Tests for agent implementations."""

import math

import pytest

from manamind.core.action import Action
from manamind.core.agent import (
    Agent,
    MCTSAgent,
    MCTSNode,
    NeuralAgent,
    RandomAgent,
)
from manamind.core.game_state import Card, create_empty_game_state


def _main_phase_state_with_land():
    """Player 0 in their main phase with priority and a Mountain in hand."""
    game_state = create_empty_game_state()
    game_state.players[0].hand.add_card(
        Card(name="Mountain", card_types=["Land"])
    )
    game_state.active_player = 0
    game_state.priority_player = 0
    game_state.phase = "main"
    return game_state


class TestAgent:
    """Test Agent base class."""

    def test_agent_creation(self):
        """Test agent creation with player ID."""
        agent = RandomAgent(player_id=0)
        assert agent.player_id == 0

    def test_abstract_base_cannot_be_instantiated(self):
        """The Agent base class leaves select_action abstract."""
        with pytest.raises(TypeError):
            Agent(0)  # type: ignore[abstract]


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

    def test_random_agent_deterministic_with_seed(self):
        """Two random agents with the same seed pick the same action."""
        game_state = _main_phase_state_with_land()

        agent1 = RandomAgent(player_id=0, seed=123)
        agent2 = RandomAgent(player_id=0, seed=123)

        action1 = agent1.select_action(game_state)
        action2 = agent2.select_action(game_state)

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
        """Test PUCT selection score calculation.

        Selection used to return infinity for every unvisited child, which
        made the policy prior irrelevant: the first unvisited child found
        always won. The score is now finite and prior-weighted, so this
        asserts the current contract.
        """
        game_state = create_empty_game_state()
        parent_node = MCTSNode(game_state)
        parent_node.visits = 2  # Parent needs visits for exploration term

        # Create a child node
        child_node = MCTSNode(game_state)

        # An unvisited child scores finitely, driven by its prior
        score = parent_node.ucb1_score(child_node)
        assert math.isfinite(score)
        assert score > 0

        # A higher prior is worth more among equally unvisited children
        eager_child = MCTSNode(game_state)
        eager_child.prior_prob = child_node.prior_prob * 2
        assert parent_node.ucb1_score(eager_child) > score

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

    def test_mcts_node_select_child(self):
        """With equal priors and visits, the higher-valued child wins."""
        game_state = _main_phase_state_with_land()
        node = MCTSNode(game_state)
        assert len(node.untried_actions) >= 2

        child1 = node.expand()
        child2 = node.expand()

        child1.backup(0.8)
        child2.backup(0.3)

        assert node.select_child() is child1


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

    def test_mcts_agent_with_custom_parameters(self):
        """Constructor parameters are stored on the agent."""
        agent = MCTSAgent(
            player_id=1,
            simulations=500,
            simulation_time=2.0,
            c_puct=2.0,
        )
        assert agent.player_id == 1
        assert agent.simulations == 500
        assert agent.simulation_time == 2.0
        assert agent.c_puct == 2.0


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

    def test_neural_agent_with_temperature(self):
        """High and low temperature agents both return an Action."""

        class MockNetwork:
            def __call__(self, game_state):
                import torch

                policy = torch.zeros(10000)
                policy[0] = 0.9
                policy[1] = 0.1
                return policy, torch.tensor(0.0)

        network = MockNetwork()
        game_state = _main_phase_state_with_land()

        for temperature in (2.0, 0.1):
            agent = NeuralAgent(
                player_id=0,
                policy_value_network=network,
                temperature=temperature,
            )
            assert isinstance(agent.select_action(game_state), Action)


class TestAgentIntegration:
    """Integration tests across agent classes."""

    def test_agent_action_validity(self):
        """Random and MCTS agents only select legal actions."""
        game_state = _main_phase_state_with_land()
        bolt = Card(
            name="Lightning Bolt",
            card_types=["Instant"],
            converted_mana_cost=1,
        )
        game_state.players[0].hand.add_card(bolt)
        game_state.players[0].mana_pool = {"R": 1}

        agents = [
            RandomAgent(player_id=0, seed=42),
            MCTSAgent(player_id=0, simulations=5),
        ]
        for agent in agents:
            action = agent.select_action(game_state)
            assert isinstance(action, Action)
            assert action.is_valid(game_state) is True

    def test_agent_player_id_consistency(self):
        """Each agent's action carries its own player ID."""
        game_state = _main_phase_state_with_land()
        game_state.players[1].hand.add_card(
            Card(name="Forest", card_types=["Land"])
        )

        action0 = RandomAgent(player_id=0, seed=42).select_action(game_state)

        # Hand the turn and priority to player 1 for their agent's pick
        game_state.active_player = 1
        game_state.priority_player = 1
        action1 = RandomAgent(player_id=1, seed=24).select_action(game_state)

        assert action0.player_id == 0
        assert action1.player_id == 1
