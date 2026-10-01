"""Game setup: seeded shuffle, opening hands, London mulligan (#22)."""

from __future__ import annotations

import random
from collections import Counter

from manamind.core.game_setup import (
    OPENING_HAND_SIZE,
    default_bottom,
    london_mulligan,
    setup_game,
)
from manamind.core.game_state import Player, create_standard_game_start
from manamind.rules.simple import build_simple_deck


def _names(cards):
    return [c.name for c in cards]


def test_setup_deals_seven_and_keeps_every_card() -> None:
    decks = [build_simple_deck(), build_simple_deck()]
    state = setup_game(decks, seed=1, keep=lambda hand, n: True)
    for player, deck in zip(state.players, decks):
        assert player.hand.size() == OPENING_HAND_SIZE
        assert Counter(_names(player.hand.cards + player.library.cards)) == (
            Counter(_names(deck))
        )


def test_setup_is_reproducible_from_a_seed() -> None:
    decks = [build_simple_deck(), build_simple_deck()]
    a = setup_game(decks, seed=42)
    b = setup_game(decks, seed=42)
    c = setup_game(decks, seed=43)
    for pa, pb in zip(a.players, b.players):
        assert _names(pa.hand.cards) == _names(pb.hand.cards)
        assert _names(pa.library.cards) == _names(pb.library.cards)
    assert any(
        _names(pa.library.cards) != _names(pc.library.cards)
        for pa, pc in zip(a.players, c.players)
    )


def test_london_mulligan_bottoms_one_card_per_mulligan() -> None:
    player = Player(player_id=0)
    player.library.cards = build_simple_deck()
    rng = random.Random(0)
    rng.shuffle(player.library.cards)

    taken = london_mulligan(player, rng, keep=lambda hand, n: n >= 2)

    assert taken == 2
    assert player.hand.size() == OPENING_HAND_SIZE - 2
    assert player.hand.size() + player.library.size() == 40


def test_mulligan_draws_a_fresh_seven_each_time() -> None:
    player = Player(player_id=0)
    player.library.cards = build_simple_deck()
    rng = random.Random(3)
    rng.shuffle(player.library.cards)
    sizes = []

    def keep(hand, n):
        sizes.append(len(hand))
        return n >= 3

    london_mulligan(player, rng, keep=keep)
    assert sizes == [7, 7, 7, 7]


def test_default_bottom_trims_the_overrepresented_side() -> None:
    deck = build_simple_deck()
    lands = [c for c in deck if c.is_land()][:6]
    spell = [c for c in deck if not c.is_land()][:1]
    chosen = default_bottom(lands + spell, 2)
    assert len(chosen) == 2
    assert all(c.is_land() for c in chosen)


def test_standard_start_without_decks_is_still_an_empty_state() -> None:
    state = create_standard_game_start()
    assert all(p.hand.size() == 0 for p in state.players)
    assert all(p.library.size() == 0 for p in state.players)


def test_standard_start_with_decks_deals_hands() -> None:
    decks = [build_simple_deck(), build_simple_deck()]
    state = create_standard_game_start(decks, seed=9)
    assert all(5 <= p.hand.size() <= 7 for p in state.players)
