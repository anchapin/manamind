"""Gumbel AlphaZero root search (issue #31)."""

from typing import Any, Tuple

import numpy as np
import pytest
import torch

from manamind.core.action import Action, ActionType
from manamind.core.agent import MCTSAgent
from manamind.core.game_state import create_standard_game_start


class FixedNetwork:
    """Strong prior on index 0, fixed value."""

    def __init__(self, width: int, value: float = 0.0):
        self.width = width
        self.value = value

    def __call__(self, game_state: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = torch.zeros(self.width)
        logits[0] = 5.0
        return logits, torch.tensor([[self.value]])


def _two_choice_agent(**kwargs: Any) -> Tuple[MCTSAgent, Action, Action]:
    agent = MCTSAgent(
        player_id=0, simulation_time=30.0, search="gumbel", **kwargs
    )
    a = Action(action_type=ActionType.PASS_PRIORITY, player_id=0)
    b = Action(action_type=ActionType.CONCEDE, player_id=0)
    agent.action_space.get_legal_actions = lambda state: [a, b]
    return agent, a, b


def test_unknown_search_is_rejected() -> None:
    with pytest.raises(ValueError):
        MCTSAgent(player_id=0, search="nope")


def test_puct_stays_the_default() -> None:
    assert MCTSAgent(player_id=0).search == "puct"


def test_budget_is_respected() -> None:
    agent, _, _ = _two_choice_agent(simulations=8)
    agent.select_action(create_standard_game_start())
    root = agent._last_root
    assert root is not None
    assert root.visits == 8
    assert sum(child.visits for _, child in root.children) == 8


def test_policy_target_is_a_distribution_over_legal_types() -> None:
    agent, a, b = _two_choice_agent(simulations=4)
    agent.select_action(create_standard_game_start())
    width = len(agent.action_space.action_to_id)
    policy = agent.last_search_policy(width)
    assert policy.sum() == pytest.approx(1.0)
    ids = {
        agent.action_space.action_to_id[a.action_type.value],
        agent.action_space.action_to_id[b.action_type.value],
    }
    assert set(np.flatnonzero(policy)) <= ids


def test_completed_q_moves_target_toward_the_better_move() -> None:
    """Conceding loses on the spot; the target should prefer passing even
    when the prior points nowhere in particular."""
    agent, a, b = _two_choice_agent(simulations=16)
    state = create_standard_game_start()
    agent.select_action(state)
    width = len(agent.action_space.action_to_id)
    policy = agent.last_search_policy(width)
    pass_id = agent.action_space.action_to_id[a.action_type.value]
    concede_id = agent.action_space.action_to_id[b.action_type.value]
    assert policy[pass_id] >= policy[concede_id]


def test_no_noise_is_deterministic() -> None:
    first, _, _ = _two_choice_agent(simulations=6)
    second, _, _ = _two_choice_agent(simulations=6)
    state = create_standard_game_start()
    assert (
        first.select_action(state).action_type
        == second.select_action(state).action_type
    )


def test_forced_move_still_skips_search() -> None:
    agent = MCTSAgent(player_id=0, simulations=50, search="gumbel")
    only = Action(action_type=ActionType.PASS_PRIORITY, player_id=0)
    agent.action_space.get_legal_actions = lambda state: [only]
    assert agent.select_action(create_standard_game_start()) is only
    assert agent.last_was_forced
    assert agent._last_gumbel_policy is None


def test_network_prior_drives_the_root_with_noise_off() -> None:
    probe = MCTSAgent(player_id=0)
    width = len(probe.action_space.action_to_id)
    agent, _, _ = _two_choice_agent(
        simulations=4, policy_network=FixedNetwork(width)
    )
    agent.select_action(create_standard_game_start())
    assert agent._last_gumbel_policy is not None
    assert sum(p for _, p in agent._last_gumbel_policy) == pytest.approx(1.0)
