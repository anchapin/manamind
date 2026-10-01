"""The standard self-play loop hands agents observations (#22)."""

from __future__ import annotations

from manamind.core import agent as agent_module
from manamind.core.action import ActionSpace
from manamind.core.observation import contains_hidden
from manamind.rules.simple import build_simple_deck, build_simple_network
from manamind.training.self_play import SelfPlayTrainer


def _trainer(seed=5):
    network = build_simple_network(
        action_space_size=len(ActionSpace().action_to_id)
    )
    config = SelfPlayTrainer(network)._default_config()
    config.update(
        mcts_simulations=1,
        mcts_time_limit=0.01,
        max_game_length=6,
        deck_lists={0: build_simple_deck(), 1: build_simple_deck()},
        seed=seed,
    )
    return SelfPlayTrainer(network, config=config)


def test_agents_never_see_the_opponents_hand(monkeypatch) -> None:
    seen_states = []

    def spy_select(self, game_state):
        seen_states.append((self.player_id, game_state))
        return ActionSpace().get_legal_actions(game_state)[0]

    monkeypatch.setattr(agent_module.MCTSAgent, "select_action", spy_select)
    monkeypatch.setattr(
        agent_module.MCTSAgent,
        "last_search_policy",
        lambda self, size: __import__("numpy").zeros(size),
    )

    game = _trainer()._play_simulation_game()

    assert game is not None and seen_states
    for viewer, state in seen_states:
        opponent = state.players[1 - viewer]
        assert opponent.hand.size() > 0
        assert all(c.name == "<hidden>" for c in opponent.hand.cards)
        assert all(
            c.name != "<hidden>" for c in state.players[viewer].hand.cards
        )
        assert contains_hidden(state)


def test_games_are_seeded_from_the_config(monkeypatch) -> None:
    hands = []

    def spy_select(self, game_state):
        hands.append([c.name for c in game_state.players[0].hand.cards])
        return ActionSpace().get_legal_actions(game_state)[0]

    monkeypatch.setattr(agent_module.MCTSAgent, "select_action", spy_select)
    monkeypatch.setattr(
        agent_module.MCTSAgent,
        "last_search_policy",
        lambda self, size: __import__("numpy").zeros(size),
    )

    _trainer(seed=11)._play_simulation_game()
    first = hands[0]
    hands.clear()
    _trainer(seed=11)._play_simulation_game()
    assert hands[0] == first
