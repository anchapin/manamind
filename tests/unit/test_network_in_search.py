"""Tests that the network is actually consulted during MCTS.

Priors, position evaluation, and the policy target used for training were
all placeholders: uniform priors, a hardcoded 0.0 value, and a dummy
training target. These tests fail if any of them regress.
"""

from typing import Any, Tuple

import numpy as np
import torch

from manamind.core.agent import MCTSAgent, MCTSNode
from manamind.core.game_state import create_standard_game_start


class FakeNetwork:
    """Minimal stand-in that records calls and returns fixed outputs."""

    def __init__(self, value: float = 0.5, width: int = 16):
        self.value = value
        self.width = width
        self.calls = 0

    def __call__(self, game_state: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        self.calls += 1
        logits = torch.zeros(self.width)
        # Favour index 0 strongly so the prior is clearly non-uniform.
        logits[0] = 10.0
        return logits, torch.tensor([[self.value]])


def test_value_network_is_consulted() -> None:
    network = FakeNetwork(value=0.5)
    agent = MCTSAgent(0, policy_network=None, value_network=network)
    state = create_standard_game_start()

    result = agent._evaluate_with_network(state)

    assert network.calls == 1
    assert result != 0.0, "placeholder 0.0 is back"
    assert -1.0 <= result <= 1.0


def test_value_is_flipped_for_the_opponent() -> None:
    """The value head speaks for the active player, not for us."""
    network = FakeNetwork(value=0.5)
    state = create_standard_game_start()
    state.active_player = 0

    ours = MCTSAgent(0, value_network=network)._evaluate_with_network(state)
    theirs = MCTSAgent(1, value_network=network)._evaluate_with_network(state)

    assert ours == -theirs


def test_value_network_failure_falls_back_to_heuristic() -> None:
    class Broken:
        def __call__(self, game_state: Any) -> Any:
            raise RuntimeError("no")

    agent = MCTSAgent(0, value_network=Broken())
    state = create_standard_game_start()

    assert agent._evaluate_with_network(state) == agent._heuristic_evaluation(
        state
    )


def test_policy_priors_are_normalised_over_legal_actions() -> None:
    network = FakeNetwork()
    agent = MCTSAgent(0, policy_network=network)
    node = MCTSNode(create_standard_game_start())

    priors = agent._policy_priors(node)

    assert network.calls == 1
    if priors:
        assert (
            sum(priors.values()) == float(1.0)
            or abs(sum(priors.values()) - 1.0) < 1e-5
        )


def test_priors_default_to_uniform_without_a_network() -> None:
    agent = MCTSAgent(0)
    node = MCTSNode(create_standard_game_start())

    agent._set_prior_probabilities(node)

    assert node.action_priors
    values = list(node.action_priors.values())
    assert all(abs(value - values[0]) < 1e-9 for value in values)
    assert abs(sum(values) - 1.0) < 1e-6


def test_expanded_child_inherits_its_prior() -> None:
    agent = MCTSAgent(0)
    node = MCTSNode(create_standard_game_start())
    agent._set_prior_probabilities(node)

    if not node.untried_actions:
        return

    expected = node.action_priors[id(node.untried_actions[-1])]
    child = node.expand()

    assert child.prior_prob == expected


def test_puct_prefers_the_higher_prior_among_unvisited_children() -> None:
    """UCB1 returned inf for every unvisited child; PUCT must not."""
    parent = MCTSNode(create_standard_game_start())
    parent.visits = 10

    low = MCTSNode(create_standard_game_start())
    high = MCTSNode(create_standard_game_start())
    low.prior_prob, high.prior_prob = 0.01, 0.9

    assert parent.ucb1_score(high) > parent.ucb1_score(low)
    assert np.isfinite(parent.ucb1_score(high))


def test_search_policy_follows_visit_counts() -> None:
    agent = MCTSAgent(0, simulations=8, simulation_time=30.0)
    state = create_standard_game_start()
    agent.select_action(state)

    policy = agent.last_search_policy(16)

    assert policy.shape == (16,)
    assert abs(float(policy.sum()) - 1.0) < 1e-5
    # A real search put its mass somewhere specific, not spread uniformly
    # across the whole head the way the old dummy target did.
    assert float(policy.max()) > 1.0 / 16


def test_search_policy_is_uniform_before_any_search() -> None:
    agent = MCTSAgent(0)

    policy = agent.last_search_policy(10)

    assert np.allclose(policy, 0.1)
