"""Soft-Z value targets: the result blended with the root search value."""

import importlib.util
import sys
from pathlib import Path

import pytest

from manamind.core.agent import MCTSAgent, MCTSNode
from manamind.core.game_state import create_standard_game_start

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train_simple.py"
_spec = importlib.util.spec_from_file_location("train_simple", _SCRIPT)
assert _spec is not None and _spec.loader is not None
train_simple = importlib.util.module_from_spec(_spec)
sys.modules["train_simple"] = train_simple
_spec.loader.exec_module(train_simple)


def test_zero_mix_is_the_game_result() -> None:
    assert train_simple.value_target(1.0, -0.6, 0.0) == 1.0
    assert train_simple.value_target(-1.0, 0.9, 0.0) == -1.0


def test_full_mix_is_the_search_value() -> None:
    assert train_simple.value_target(1.0, -0.6, 1.0) == pytest.approx(-0.6)


def test_half_mix_averages() -> None:
    assert train_simple.value_target(-1.0, 0.5, 0.5) == pytest.approx(-0.25)


def test_missing_search_value_falls_back_to_result() -> None:
    assert train_simple.value_target(1.0, None, 0.5) == 1.0


def test_root_value_is_none_before_any_search() -> None:
    agent = MCTSAgent(player_id=0, simulations=1, simulation_time=0.01)
    assert agent.last_root_value() is None


def test_root_value_is_the_root_mean() -> None:
    agent = MCTSAgent(player_id=0, simulations=1, simulation_time=0.01)
    root = MCTSNode(create_standard_game_start())
    root.visits = 4
    root.total_value = -1.0
    agent._last_root = root
    assert agent.last_root_value() == pytest.approx(-0.25)


def test_unvisited_root_has_no_value() -> None:
    agent = MCTSAgent(player_id=0, simulations=1, simulation_time=0.01)
    agent._last_root = MCTSNode(create_standard_game_start())
    assert agent.last_root_value() is None


@pytest.mark.parametrize("search", ["puct", "gumbel"])
def test_search_leaves_a_bounded_root_value(search: str) -> None:
    from manamind.rules.simple import create_simple_game_start

    state = create_simple_game_start(0)
    agent = MCTSAgent(
        player_id=state.priority_player,
        simulations=8,
        simulation_time=5.0,
        search=search,
    )
    agent.select_action(state)
    value = agent.last_root_value()
    if agent.last_was_forced:
        assert value is None
    else:
        assert value is not None and -1.0 <= value <= 1.0
