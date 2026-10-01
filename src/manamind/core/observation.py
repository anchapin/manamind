"""What one player is allowed to see of a game.

The engine keeps the full ``GameState``. Agents receive ``observe(state,
viewer)``: a copy in which every card the viewer cannot see is replaced by a
placeholder. Zone sizes are preserved, identities and order are not.

Visible to the viewer: their own hand, both battlefields, both graveyards,
both exiles, the stack. Hidden: the opponent's hand and both libraries.

Search needs a complete state to simulate, so ``determinize`` fills the
placeholders with one sample consistent with what the viewer can see, drawn
from the known deck lists. That is the PIMC baseline; #16 replaces it with
a learned belief model.
"""

from __future__ import annotations

import copy
import random
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence

from manamind.core.game_state import Card, GameState, Player

HIDDEN_CARD_NAME = "<hidden>"


def hidden_card() -> Card:
    """A placeholder with no identity, cost, types or stats."""
    return Card(name=HIDDEN_CARD_NAME)


def is_hidden(card: Card) -> bool:
    return card.name == HIDDEN_CARD_NAME


def _redact(cards: List[Card]) -> List[Card]:
    return [hidden_card() for _ in cards]


def observe(game_state: GameState, viewer: int) -> GameState:
    """Return the viewer's observation of ``game_state``."""
    obs = game_state.copy()
    for player in obs.players:
        player.library.cards = _redact(player.library.cards)
        if player.player_id != viewer:
            player.hand.cards = _redact(player.hand.cards)
    return obs


def contains_hidden(game_state: GameState) -> bool:
    return any(
        is_hidden(card)
        for player in game_state.players
        for zone in (player.hand, player.library)
        for card in zone.cards
    )


def _visible_names(player: Player) -> Iterable[str]:
    for zone in (
        player.hand,
        player.battlefield,
        player.graveyard,
        player.exile,
    ):
        for card in zone.cards:
            if not is_hidden(card):
                yield card.name


def determinize(
    observation: GameState,
    deck_lists: Optional[Dict[int, Sequence[Card]]],
    rng: Optional[random.Random] = None,
) -> GameState:
    """Fill hidden slots with one world consistent with the observation.

    For each player the unseen pool is their deck list minus every card of
    theirs the viewer can see. The pool is shuffled and dealt into that
    player's hidden hand slots first, then their library. Without a deck
    list for a player, their placeholders stay as they are.
    """
    rng = rng or random.Random()
    world = observation.copy()
    for player in world.players:
        deck = (deck_lists or {}).get(player.player_id)
        if not deck:
            continue
        seen = Counter(_visible_names(player))
        pool: List[Card] = []
        for card in deck:
            if seen[card.name] > 0:
                seen[card.name] -= 1
            else:
                pool.append(copy.deepcopy(card))
        rng.shuffle(pool)
        for zone in (player.hand, player.library):
            for i, card in enumerate(zone.cards):
                if is_hidden(card) and pool:
                    zone.cards[i] = pool.pop()
    return world
