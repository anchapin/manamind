"""Base agent interface and Monte Carlo Tree Search implementation.

This module defines the core agent interface and implements MCTS for decisions.
"""

from __future__ import annotations

import math
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch

from manamind.core.action import Action, ActionSpace, ActionType
from manamind.core.game_state import GameState


class Agent(ABC):
    """Abstract base class for all ManaMind agents."""

    def __init__(self, player_id: int):
        """Initialize the agent.

        Args:
            player_id: The player ID this agent controls (0 or 1)
        """
        self.player_id = player_id

    @abstractmethod
    def select_action(self, game_state: GameState) -> Action:
        """Select the best action from the current game state.

        Args:
            game_state: Current game state

        Returns:
            The selected action
        """
        pass

    @abstractmethod
    def update_from_game(
        self, game_history: List[Tuple[GameState, Action, float]]
    ) -> None:
        """Update the agent's knowledge from a completed game.

        Args:
            game_history: List of (state, action, reward) tuples from the game
        """
        pass


class RandomAgent(Agent):
    """Simple random agent for testing and baseline comparison."""

    def __init__(self, player_id: int, seed: Optional[int] = None):
        super().__init__(player_id)
        self.action_space = ActionSpace()
        self.rng = random.Random(seed)

    def select_action(self, game_state: GameState) -> Action:
        """Select a random legal action."""
        legal_actions = self.action_space.get_legal_actions(game_state)
        if not legal_actions:
            raise ValueError("No legal actions available")
        return self.rng.choice(legal_actions)

    def update_from_game(
        self, game_history: List[Tuple[GameState, Action, float]]
    ) -> None:
        """Random agent doesn't learn."""
        pass


class MCTSNode:
    """Node in the Monte Carlo Tree Search tree."""

    def __init__(
        self,
        game_state: GameState,
        action: Optional[Action] = None,
        parent: Optional[MCTSNode] = None,
    ):
        """Initialize MCTS node.

        Args:
            game_state: Game state this node represents
            action: Action taken to reach this state (None for root)
            parent: Parent node (None for root)
        """
        self.game_state = game_state
        self.action = action
        self.parent = parent
        self.children: List[Tuple[Action, MCTSNode]] = []

        # MCTS statistics
        self.visits = 0
        self.total_value = 0.0
        self.prior_prob = 1.0  # From policy network

        # Untried actions
        action_space = ActionSpace()
        self.untried_actions = action_space.get_legal_actions(game_state)

    def is_fully_expanded(self) -> bool:
        """Check if all legal actions have been tried."""
        return len(self.untried_actions) == 0

    def is_terminal(self) -> bool:
        """Check if this is a terminal game state."""
        return self.game_state.is_game_over()

    def ucb1_score(self, child_node: MCTSNode, c: float = 1.414) -> float:
        """Calculate UCB1 score for action selection.

        Args:
            child_node: Child node to calculate score for
            c: Exploration parameter

        Returns:
            UCB1 score
        """
        if child_node.visits == 0:
            return float("inf")

        exploitation = child_node.total_value / child_node.visits
        exploration = (
            c * math.sqrt(math.log(self.visits) / child_node.visits)
            if self.visits > 0
            else 0.0
        )
        return exploitation + exploration

    def select_child(self) -> MCTSNode:
        """Select the child with the highest UCB1 score."""
        return max(
            (child for _, child in self.children),
            key=lambda child: self.ucb1_score(child)
        )

    def expand(self) -> MCTSNode:
        """Expand the tree by adding a new child node."""
        if not self.untried_actions:
            raise ValueError("No untried actions to expand")

        action = self.untried_actions.pop()
        new_state = action.execute(self.game_state)
        child_node = MCTSNode(new_state, action, self)
        self.children.append((action, child_node))
        return child_node

    def backup(self, value: float) -> None:
        """Backup the value through the tree."""
        self.visits += 1
        self.total_value += value

        if self.parent:
            # Flip value for opponent
            self.parent.backup(-value)


class MCTSAgent(Agent):
    """Agent using Monte Carlo Tree Search for decision making."""

    def __init__(
        self,
        player_id: int,
        policy_network: Any = None,
        value_network: Any = None,
        simulations: int = 1000,
        simulation_time: float = 1.0,
        c_puct: float = 1.0,
    ) -> None:
        """Initialize MCTS agent.

        Args:
            player_id: Player ID this agent controls
            policy_network: Neural network for action priors (optional)
            value_network: Neural network for position evaluation (optional)
            simulations: Number of MCTS simulations per move
            simulation_time: Time limit for MCTS (seconds)
            c_puct: Exploration parameter for PUCT algorithm
        """
        super().__init__(player_id)
        self.policy_network = policy_network
        self.value_network = value_network
        self.simulations = simulations
        self.simulation_time = simulation_time
        self.c_puct = c_puct
        self.action_space = ActionSpace()

    def select_action(self, game_state: GameState) -> Action:
        """Select the best action using MCTS.

        Args:
            game_state: Current game state

        Returns:
            The selected action
        """
        root = MCTSNode(game_state)

        # Set prior probabilities from policy network if available
        if self.policy_network:
            self._set_prior_probabilities(root)

        start_time = time.time()
        simulation_count = 0

        # Run MCTS simulations
        while (
            simulation_count < self.simulations
            and time.time() - start_time < self.simulation_time
        ):

            # Selection phase - traverse tree to leaf
            node = root
            path = [node]

            while not node.is_terminal() and node.is_fully_expanded():
                node = node.select_child()
                path.append(node)

            # Expansion phase - add new child if possible
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
                path.append(node)

            # Simulation phase - evaluate position
            value = self._evaluate_position(node.game_state)

            # Backpropagation phase - update statistics
            for node in reversed(path):
                node.backup(value)
                value = -value  # Flip for opponent

            simulation_count += 1

        # Select the most visited child as the best move
        if not root.children:
            # No expansions happened, return random action
            legal_actions = self.action_space.get_legal_actions(game_state)
            return random.choice(legal_actions)

        best_child = max(
            (child for _, child in root.children), key=lambda child: child.visits
        )
        if best_child.action:
            return best_child.action

        # Fallback if no action found
        legal_actions = self.action_space.get_legal_actions(game_state)
        return (
            random.choice(legal_actions)
            if legal_actions
            else Action(ActionType.PASS_PRIORITY, self.player_id)
        )

    def _set_prior_probabilities(self, node: MCTSNode) -> None:
        """Set prior probabilities for actions using the policy network."""
        if not self.policy_network:
            return

        # TODO: Implement policy network evaluation
        # For now, set uniform priors
        num_actions = len(node.untried_actions)
        if num_actions > 0:
            for action in node.untried_actions:
                # This would be set from policy network output
                pass

    def _evaluate_position(self, game_state: GameState) -> float:
        """Evaluate a game position.

        Args:
            game_state: Game state to evaluate

        Returns:
            Value from current player's perspective (-1 to 1)
        """
        # Check for terminal states
        if game_state.is_game_over():
            winner = game_state.winner()
            if winner == self.player_id:
                return 1.0
            elif winner is not None:
                return -1.0
            else:
                return 0.0  # Draw

        # Use value network if available
        if self.value_network:
            return self._evaluate_with_network(game_state)

        # Fallback to simple heuristic
        return self._heuristic_evaluation(game_state)

    def _evaluate_with_network(self, game_state: GameState) -> float:
        """Evaluate position using neural network.

        Args:
            game_state: Game state to evaluate

        Returns:
            Network evaluation (-1 to 1)
        """
        # TODO: Implement network evaluation
        # This requires the game state encoder and value network
        return 0.0

    def _heuristic_evaluation(self, game_state: GameState) -> float:
        """Simple heuristic evaluation of the position.

        Args:
            game_state: Game state to evaluate

        Returns:
            Heuristic value (-1 to 1)
        """
        # Simple life difference heuristic
        my_life = game_state.players[self.player_id].life
        opp_life = game_state.players[1 - self.player_id].life

        life_diff = my_life - opp_life
        # Normalize to roughly [-1, 1]
        return max(-1.0, min(1.0, life_diff / 20.0))

    def update_from_game(
        self, game_history: List[Tuple[GameState, Action, float]]
    ) -> None:
        """Update agent from game history.

        For MCTS agent, this could be used to update neural networks.
        """
        # TODO: Implement training data collection for neural networks
        pass


class NeuralAgent(Agent):
    """Agent using neural networks for policy and value estimation."""

    def __init__(
        self,
        player_id: int,
        policy_value_network: Any,
        action_space: Optional[ActionSpace] = None,
        temperature: float = 1.0,
    ) -> None:
        """Initialize neural agent.

        Args:
            player_id: Player ID this agent controls
            policy_value_network: Combined policy/value network
            action_space: Action space for the game
            temperature: Temperature for action selection
        """
        super().__init__(player_id)
        self.policy_value_network = policy_value_network
        self.action_space = action_space or ActionSpace()
        self.temperature = temperature

    def select_action(self, game_state: GameState) -> Action:
        """Select action using neural network policy.

        Args:
            game_state: Current game state

        Returns:
            Selected action
        """
        legal_actions = self.action_space.get_legal_actions(game_state)
        if not legal_actions:
            raise ValueError("No legal actions available")

        # Get policy and value from network
        with torch.no_grad():
            policy_logits, value = self.policy_value_network(game_state)

        # Apply softmax with temperature
        if self.temperature > 0:
            torch.softmax(policy_logits / self.temperature, dim=-1)
            # TODO: Map probabilities to legal actions and sample

        # For now, return random action
        return random.choice(legal_actions)

    def update_from_game(
        self, game_history: List[Tuple[GameState, Action, float]]
    ) -> None:
        """Collect training data from game history."""
        # TODO: Store training examples for network updates
        pass
