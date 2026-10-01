"""Standing check that agents cannot see hidden information (#22)."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

from manamind.core.action import ActionSpace
from manamind.core.agent import Agent, MCTSAgent
from manamind.core.observation import (
    contains_hidden,
    determinize,
    is_hidden,
    observe,
)
from manamind.rules.simple import build_simple_deck, create_simple_game_start


def test_observation_hides_opponent_hand_and_both_libraries() -> None:
    state = create_simple_game_start(3)
    obs = observe(state, viewer=0)

    me, opp = obs.players
    assert all(not is_hidden(c) for c in me.hand.cards)
    assert [c.name for c in me.hand.cards] == [
        c.name for c in state.players[0].hand.cards
    ]
    assert all(is_hidden(c) for c in opp.hand.cards)
    assert all(is_hidden(c) for c in me.library.cards)
    assert all(is_hidden(c) for c in opp.library.cards)


def test_observation_keeps_zone_sizes() -> None:
    state = create_simple_game_start(3)
    obs = observe(state, viewer=1)
    for real, seen in zip(state.players, obs.players):
        assert seen.hand.size() == real.hand.size()
        assert seen.library.size() == real.library.size()


def test_observation_does_not_touch_the_engine_state() -> None:
    state = create_simple_game_start(3)
    observe(state, viewer=0)
    assert not contains_hidden(state)


def test_determinize_is_consistent_with_what_the_viewer_sees() -> None:
    state = create_simple_game_start(4)
    obs = observe(state, viewer=0)
    decks = {0: build_simple_deck(), 1: build_simple_deck()}
    world = determinize(obs, decks, random.Random(0))

    assert not contains_hidden(world)
    # The viewer's own hand is untouched.
    assert [c.name for c in world.players[0].hand.cards] == [
        c.name for c in state.players[0].hand.cards
    ]
    # Each player's cards across all zones are exactly their deck list.
    for pid in (0, 1):
        p = world.players[pid]
        cards = p.hand.cards + p.library.cards + p.battlefield.cards
        assert Counter(c.name for c in cards) == Counter(
            c.name for c in decks[pid]
        )


class _Spy(Agent):
    """Records every state it is handed."""

    def __init__(self, player_id: int) -> None:
        super().__init__(player_id)
        self.seen: list = []

    def select_action(self, game_state: Any) -> Any:
        self.seen.append(game_state)
        return ActionSpace().get_legal_actions(game_state)[0]

    def update_from_game(self, game_result: Any) -> None:
        pass


def test_engine_only_ever_hands_agents_observations() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import train_simple

    spies = {0: _Spy(0), 1: _Spy(1)}
    train_simple.play_game(spies, seed=5)
    for pid, spy in spies.items():
        assert spy.seen
        for state in spy.seen:
            opp = state.players[1 - pid]
            assert all(is_hidden(c) for c in opp.hand.cards)
            for player in state.players:
                assert all(is_hidden(c) for c in player.library.cards)


def test_mcts_agent_plays_from_an_observation() -> None:
    state = create_simple_game_start(6)
    decks = {0: build_simple_deck(), 1: build_simple_deck()}
    agent = MCTSAgent(0, simulations=4, simulation_time=5.0, deck_lists=decks)
    action = agent.select_action(observe(state, 0))
    # The chosen action is legal in the real game.
    legal = ActionSpace().get_legal_actions(state)
    assert action.action_type in {a.action_type for a in legal}
    action.execute(state)
