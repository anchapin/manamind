"""Base agent interface and Monte Carlo Tree Search implementation.

This module defines the core agent interface and implements MCTS for decisions.
"""

from __future__ import annotations

import logging
import math
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from numpy import ndarray

from manamind.core.action import Action, ActionSpace, ActionType
from manamind.core.game_state import GameState

logger = logging.getLogger(__name__)

# Share of prior mass spread uniformly over a node's legal actions, so
# no legal move is left with a prior of exactly zero.
PRIOR_UNIFORM_FLOOR = 0.25


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

        # Priors for this node's legal actions, keyed by action identity.
        # Populated by MCTSAgent._set_prior_probabilities so that a
        # child created later by expand() inherits the right prior.
        self.action_priors: Dict[int, float] = {}

        # First-play urgency. None keeps the old behaviour (an unvisited
        # child scores 0 for exploitation). A float means an unvisited child
        # starts at the mean value of its already-visited siblings minus this
        # reduction, so the search judges an untried move against what this
        # position is actually worth instead of against a flat 0.
        self.fpu_reduction: Optional[float] = None

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
        """Score a child for selection.

        With a policy prior attached this is PUCT, the AlphaZero selection
        rule: the exploration term is scaled by the network's prior for the
        action, so promising moves are searched first. Without a prior
        (uniform 1.0 across children) it degrades to UCB1-style behaviour.

        Args:
            child_node: Child node to calculate score for
            c: Exploration parameter

        Returns:
            Selection score; higher is more worth searching
        """
        if child_node.visits > 0:
            exploitation = child_node.total_value / child_node.visits
        else:
            exploitation = self.first_play_value()

        # PUCT exploration: prior * sqrt(parent visits) / (1 + child visits).
        # Unlike UCB1 this is finite for an unvisited child, so the prior
        # decides which unexplored action is tried first instead of the
        # arbitrary order they were generated in.
        exploration = (
            c
            * child_node.prior_prob
            * math.sqrt(max(self.visits, 1))
            / (1 + child_node.visits)
        )

        return exploitation + exploration

    def first_play_value(self) -> float:
        """Exploitation score for a child that has not been visited yet."""
        if self.fpu_reduction is None:
            return 0.0
        visited = [
            child.total_value / child.visits
            for _, child in self.children
            if child.visits > 0
        ]
        if not visited:
            return 0.0
        return sum(visited) / len(visited) - self.fpu_reduction

    def select_child(self) -> MCTSNode:
        """Select the child with the highest UCB1 score."""
        return max(
            (child for _, child in self.children),
            key=lambda child: self.ucb1_score(child),
        )

    def expand(self) -> MCTSNode:
        """Expand the tree by adding a new child node."""
        if not self.untried_actions:
            raise ValueError("No untried actions to expand")

        action = self.untried_actions.pop()
        new_state = action.execute(self.game_state)
        child_node = MCTSNode(new_state, action, self)
        child_node.fpu_reduction = self.fpu_reduction

        # Carry over the policy prior for this action, if one was set.
        # The default is uniform rather than zero: a zero prior removes a
        # child from PUCT exploration permanently.
        if self.action_priors:
            child_node.prior_prob = self.action_priors.get(
                id(action), 1.0 / max(len(self.action_priors), 1)
            )

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
        root_dirichlet_alpha: Optional[float] = None,
        root_noise_fraction: float = 0.25,
        temperature: float = 0.0,
        fpu_reduction: Optional[float] = None,
        search: str = "puct",
        gumbel_k: int = 16,
        gumbel_noise: bool = False,
        c_visit: float = 50.0,
        c_scale: float = 1.0,
    ) -> None:
        """Initialize MCTS agent.

        Args:
            player_id: Player ID this agent controls
            policy_network: Neural network for action priors (optional)
            value_network: Neural network for position evaluation (optional)
            simulations: Number of MCTS simulations per move
            simulation_time: Time limit for MCTS (seconds)
            c_puct: Exploration parameter for PUCT algorithm
            root_dirichlet_alpha: When set, mix Dirichlet(alpha) noise into
                the root priors (AlphaZero self-play exploration). Leave
                None for evaluation and real play.
            root_noise_fraction: Weight of that noise in the mixed prior.
            temperature: 0 plays the most-visited move. Above 0, sample the
                move with probability proportional to visits ** (1 / T),
                as AlphaZero does in self-play so games stay varied.
            fpu_reduction: First-play urgency for unvisited children.
            search: "puct" (AlphaZero) or "gumbel" (Gumbel AlphaZero root
                search: Gumbel-Top-k plus sequential halving, with a
                completed-Q policy target). Gumbel ignores Dirichlet noise
                and temperature; its exploration is the Gumbel noise.
            gumbel_k: Actions considered at the root by Gumbel search.
            gumbel_noise: Add Gumbel noise at the root. On for self-play,
                off for evaluation, which then picks deterministically.
            c_visit, c_scale: The sigma(q) scaling from Danihelka et al.
                (2022): sigma(q) = (c_visit + max_b N(b)) * c_scale * q,
                with q rescaled to [0, 1].
        """
        if search not in ("puct", "gumbel"):
            raise ValueError(f"unknown search {search!r}")
        super().__init__(player_id)
        self.policy_network = policy_network
        self.value_network = value_network

        # Root of the most recent search, kept so callers (self-play) can
        # read the visit distribution that produced the chosen action.
        self._last_root: Optional[MCTSNode] = None
        self.simulations = simulations
        self.simulation_time = simulation_time
        self.c_puct = c_puct
        self.root_dirichlet_alpha = root_dirichlet_alpha
        self.root_noise_fraction = root_noise_fraction
        self.temperature = temperature
        self.fpu_reduction = fpu_reduction
        self._last_forced: Optional[Action] = None
        self.search = search
        self.gumbel_k = gumbel_k
        self.gumbel_noise = gumbel_noise
        self.c_visit = c_visit
        self.c_scale = c_scale
        # (action, probability) pairs from the last Gumbel search: the
        # completed-Q policy target. None after a PUCT search.
        self._last_gumbel_policy: Optional[List[Tuple[Action, float]]] = None
        self.action_space = ActionSpace()

    def select_action(self, game_state: GameState) -> Action:
        """Select the best action using MCTS.

        Args:
            game_state: Current game state

        Returns:
            The selected action
        """
        # A forced move (one legal action) needs no search. Searching it
        # anyway wastes the budget, and recording its trivial visit
        # distribution floods the policy target with that action: about
        # half of all simple-mode decisions are forced passes.
        self._last_gumbel_policy = None
        legal = self.action_space.get_legal_actions(game_state)
        if len(legal) == 1:
            self._last_root = None
            self._last_forced = legal[0]
            return legal[0]
        self._last_forced = None

        if self.search == "gumbel":
            return self._gumbel_select(game_state)

        root = MCTSNode(game_state)
        root.fpu_reduction = self.fpu_reduction
        self._last_root = root

        # Priors shape which moves PUCT explores first. Set them even
        # without a network: a uniform prior keeps selection well defined.
        self._set_prior_probabilities(root)
        self._add_root_noise(root)

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

            # Backpropagation phase - update statistics.
            # The leaf value is from this agent's perspective. Each node
            # stores value from the perspective of the player who chose the
            # action leading to it, so selection can maximise Q at every
            # level. Flipping the sign once per tree level instead assumes
            # players strictly alternate, which MTG does not: one player
            # often takes several actions in a row (land, spell, spell),
            # and per-level flipping scrambles the sign by depth.
            self._backup_path(path, value)

            simulation_count += 1

        # Select the most visited child as the best move
        if not root.children:
            # No expansions happened, return random action
            legal_actions = self.action_space.get_legal_actions(game_state)
            return random.choice(legal_actions)

        # Most visits wins; ties (common at low simulation counts, where
        # PUCT spreads visits almost evenly) go to the better mean value,
        # then the higher prior. Breaking ties by list order instead always
        # picked the first-expanded child, which is pass_priority because
        # legal actions list it last and expand() pops from the end.
        if self.temperature > 0 and root.children:
            weights = [
                child.visits ** (1.0 / self.temperature)
                for _, child in root.children
            ]
            if sum(weights) > 0:
                return random.choices(
                    [action for action, _ in root.children], weights=weights
                )[0]

        best_child = max(
            (child for _, child in root.children),
            key=lambda child: (
                child.visits,
                child.total_value / child.visits if child.visits else 0.0,
                child.prior_prob,
            ),
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

    def _simulate_from(self, root: MCTSNode, child: MCTSNode) -> None:
        """Run one simulation that starts by taking ``child`` at the root."""
        node = child
        path = [root, child]
        while (
            not node.is_terminal()
            and node.is_fully_expanded()
            and node.children
        ):
            node = node.select_child()
            path.append(node)
        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()
            path.append(node)
        value = self._evaluate_position(node.game_state)
        self._backup_path(path, value)

    def _gumbel_select(self, game_state: GameState) -> Action:
        """Gumbel AlphaZero root search (Danihelka et al., ICLR 2022).

        Sample the top-k root actions by logits + Gumbel noise, split the
        simulation budget over them with sequential halving, and play the
        survivor with the best g + logits + sigma(q). The policy target is
        softmax(logits + sigma(completed q)): every legal action gets a
        value (unvisited ones the mixed estimate v_mix), so the target can
        move away from the prior even with a handful of simulations. Visit
        counts at that budget mostly copy the prior, which is how earlier
        runs collapsed onto passing.
        """
        root = MCTSNode(game_state)
        root.fpu_reduction = self.fpu_reduction
        self._last_root = root
        self._set_prior_probabilities(root)
        while root.untried_actions:
            root.expand()
        children = [child for _, child in root.children]
        count = len(children)

        priors = np.array(
            [max(child.prior_prob, 1e-12) for child in children], dtype=float
        )
        priors /= priors.sum()
        logits = np.log(priors)
        if self.gumbel_noise:
            uniform = np.array(
                [max(random.random(), 1e-12) for _ in children], dtype=float
            )
            gumbel = -np.log(-np.log(uniform))
        else:
            gumbel = np.zeros(count)

        # Child values are stored from the root mover's perspective; the
        # leaf evaluator speaks for this agent.
        sign = 1.0 if game_state.priority_player == self.player_id else -1.0
        root_value = sign * self._evaluate_position(game_state)

        def completed_q() -> np.ndarray:
            visits = np.array([c.visits for c in children], dtype=float)
            q = np.array(
                [
                    c.total_value / c.visits if c.visits else 0.0
                    for c in children
                ]
            )
            visited = visits > 0
            if visited.any():
                weight = priors[visited].sum()
                mixed = (
                    root_value
                    + visits.sum()
                    / weight
                    * float((priors * q)[visited].sum())
                ) / (1.0 + visits.sum())
            else:
                mixed = root_value
            completed = np.where(visited, q, mixed)
            return np.clip((completed + 1.0) / 2.0, 0.0, 1.0)

        def sigma(q: np.ndarray) -> np.ndarray:
            max_visits = max((c.visits for c in children), default=0)
            return (self.c_visit + max_visits) * self.c_scale * q

        considered = min(max(self.gumbel_k, 1), count)
        remaining = list(np.argsort(-(gumbel + logits))[:considered])
        phases = (
            max(1, math.ceil(math.log2(considered))) if considered > 1 else 1
        )
        budget = max(self.simulations, 0)
        used = 0
        start = time.time()

        while used < budget and time.time() - start < self.simulation_time:
            per_action = max(1, budget // (phases * len(remaining)))
            for index in remaining:
                for _ in range(per_action):
                    if used >= budget:
                        break
                    self._simulate_from(root, children[index])
                    used += 1
            if len(remaining) == 1:
                break
            score = gumbel + logits + sigma(completed_q())
            remaining = sorted(remaining, key=lambda i: -score[i])[
                : max(1, len(remaining) // 2)
            ]

        score = gumbel + logits + sigma(completed_q())
        best = max(remaining, key=lambda i: score[i])

        target = logits + sigma(completed_q())
        target = np.exp(target - target.max())
        target /= target.sum()
        self._last_gumbel_policy = [
            (action, float(p)) for (action, _), p in zip(root.children, target)
        ]
        return root.children[best][0]

    def _backup_path(self, path: List[MCTSNode], value: float) -> None:
        """Propagate a leaf value up the path with per-node perspective.

        Args:
            path: Nodes from root to leaf, in order
            value: Leaf evaluation from this agent's perspective
        """
        for node in path:
            node.visits += 1
            if node.parent is None:
                node.total_value += value
                continue
            mover = node.parent.game_state.priority_player
            node.total_value += value if mover == self.player_id else -value

    @property
    def last_was_forced(self) -> bool:
        """True when the last select_action had exactly one legal move."""
        return getattr(self, "_last_forced", None) is not None

    def last_search_policy(self, width: int) -> ndarray[Any, Any]:
        """Visit-count distribution from the most recent search.

        This is the AlphaZero policy target: MCTS acts as a policy
        improvement operator over the network's raw output, so training
        toward the search's visit counts is what makes the loop improve.

        Args:
            width: Length of the returned vector, normally the network's
                action space size

        Returns:
            A probability vector of length `width`. Uniform if no search
            has run yet or the search produced no children.
        """
        root = self._last_root
        policy = np.zeros(width, dtype=np.float32)

        forced = getattr(self, "_last_forced", None)
        if root is None and forced is not None:
            index = self.action_space.action_to_id.get(
                forced.action_type.value
            )
            if index is not None and index < width:
                policy[index] = 1.0
                return policy

        if root is None or not root.children:
            policy[:] = 1.0 / width
            return policy

        gumbel = getattr(self, "_last_gumbel_policy", None)
        if gumbel:
            for action, probability in gumbel:
                index = self.action_space.action_to_id.get(
                    action.action_type.value
                )
                if index is not None and index < width:
                    policy[index] += probability
            total = policy.sum()
            if total > 0:
                return policy / total
            policy[:] = 1.0 / width
            return policy

        for action, child in root.children:
            index = self.action_space.action_to_id.get(
                action.action_type.value
            )
            if index is None or index >= width:
                continue
            policy[index] += float(child.visits)

        total = policy.sum()
        if total <= 0:
            policy[:] = 1.0 / width
        else:
            policy /= total

        return policy

    def _policy_priors(self, node: MCTSNode) -> Dict[int, float]:
        """Ask the policy network for a prior over this node's legal actions.

        Returns a mapping from the index of a legal action (in the node's
        own ordering) to its prior probability, normalised over the legal
        actions only. An empty mapping means the caller should fall back to
        uniform priors.
        """
        if self.policy_network is None:
            return {}

        actions = self._node_actions(node)
        if not actions:
            return {}

        try:
            with torch.no_grad():
                output = self.policy_network(node.game_state)
                logits = output[0] if isinstance(output, tuple) else output
                probs = torch.softmax(logits.view(-1), dim=-1)
        except Exception as error:  # pragma: no cover - defensive
            logger.warning(f"Policy network evaluation failed: {error}")
            return {}

        # Mask to legal actions, then renormalise. The action space is a
        # coarse action-type mapping today (see ActionSpace TODOs), so
        # several legal actions can share one index; they split that mass
        # evenly rather than each claiming it.
        width = probs.numel()
        indices: List[Optional[int]] = []
        for action in actions:
            index = self.action_space.action_to_id.get(
                action.action_type.value
            )
            if index is None or index >= width:
                indices.append(None)
            else:
                indices.append(index)

        shared: Dict[int, int] = {}
        for index in indices:
            if index is not None:
                shared[index] = shared.get(index, 0) + 1

        raw: List[float] = []
        for index in indices:
            if index is None:
                raw.append(0.0)
            else:
                raw.append(float(probs[index]) / shared[index])

        total = sum(raw)
        count = len(actions)
        if total <= 0:
            return {}

        # Mix in a uniform floor. A prior of exactly zero is absorbing
        # under PUCT: the exploration term vanishes, so once such a child
        # has been visited and scored badly it can never be revisited, and
        # one action with all the mass takes every remaining simulation.
        # Today's action space gives whole classes of legal moves a zero
        # prior, so the floor is what keeps the search looking at them.
        floor = PRIOR_UNIFORM_FLOOR
        return {
            position: (1.0 - floor) * (value / total) + floor / count
            for position, value in enumerate(raw)
        }

    def _node_actions(self, node: MCTSNode) -> List[Action]:
        """All legal actions at a node, expanded or not, in a stable order."""
        return list(node.untried_actions) + [
            action for action, _ in node.children
        ]

    def _set_prior_probabilities(self, node: MCTSNode) -> None:
        """Attach policy priors to a node for PUCT selection.

        Priors are cached on the node so a child created later by expand()
        picks up the prior for the action that produced it, instead of the
        default 1.0 that would make PUCT behave as if every move were
        equally promising.
        """
        actions = self._node_actions(node)
        if not actions:
            return

        priors = self._policy_priors(node)
        if not priors:
            uniform = 1.0 / len(actions)
            priors = {position: uniform for position in range(len(actions))}

        fallback = 1.0 / len(actions)
        node.action_priors = {
            id(action): priors.get(position, fallback)
            for position, action in enumerate(actions)
        }

        # Children that already exist get their prior now.
        for action, child in node.children:
            child.prior_prob = node.action_priors.get(id(action), fallback)

    def _add_root_noise(self, root: MCTSNode) -> None:
        """Mix Dirichlet noise into the root priors for self-play.

        Without it an untrained policy head is self-reinforcing: search
        follows the prior, the visit counts it produces become the training
        target, and the network learns its own starting bias. With the
        simple-mode network that bias was pass_priority, so both self-play
        seats passed every turn until someone drew from an empty library.
        """
        alpha = self.root_dirichlet_alpha
        if not alpha or len(root.action_priors) < 2:
            return
        keys = list(root.action_priors)
        draws = [random.gammavariate(alpha, 1.0) for _ in keys]
        total = sum(draws) or 1.0
        frac = self.root_noise_fraction
        for key, draw in zip(keys, draws):
            root.action_priors[key] = (1.0 - frac) * root.action_priors[
                key
            ] + frac * (draw / total)

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
        """Evaluate a position with the value head.

        Args:
            game_state: Game state to evaluate

        Returns:
            Network evaluation in [-1, 1], from the perspective of the
            player this agent controls. Falls back to the heuristic if the
            network cannot evaluate the state.
        """
        if self.value_network is None:
            return self._heuristic_evaluation(game_state)

        try:
            with torch.no_grad():
                output = self.value_network(game_state)
                value = output[1] if isinstance(output, tuple) else output
                scalar = float(value.view(-1)[0])
        except Exception as error:  # pragma: no cover - defensive
            logger.warning(f"Value network evaluation failed: {error}")
            return self._heuristic_evaluation(game_state)

        if not math.isfinite(scalar):
            return self._heuristic_evaluation(game_state)

        # Self-play labels each position from the point of view of the
        # player holding priority (the one choosing the move), so read the
        # value head the same way. Using active_player here disagreed with
        # the labels whenever the defender acts, e.g. every block decision.
        if game_state.priority_player != self.player_id:
            scalar = -scalar

        return max(-1.0, min(1.0, scalar))

    def _heuristic_evaluation(self, game_state: GameState) -> float:
        """Simple heuristic evaluation of the position.

        Args:
            game_state: Game state to evaluate

        Returns:
            Heuristic value (-1 to 1)
        """

        # Life alone is flat for the first several turns, so every early
        # move scored 0.0 and search had nothing to rank. Count material on
        # the battlefield and lands in play too.
        def material(player: Any) -> float:
            total = 0.0
            for card in player.battlefield.cards:
                if card.is_creature():
                    total += (card.current_power() or 0) + 0.5 * (
                        card.current_toughness() or 0
                    )
                elif card.is_land():
                    total += 0.5
            return total

        me = game_state.players[self.player_id]
        opp = game_state.players[1 - self.player_id]
        score = (me.life - opp.life) / 20.0 + (
            material(me) - material(opp)
        ) / 10.0
        return math.tanh(score)

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
