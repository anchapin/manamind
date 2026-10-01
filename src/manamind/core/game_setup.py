"""Game setup: seeded shuffle, opening hands, London mulligan.

London mulligan (current rule 103.5): draw seven. To mulligan, shuffle the
hand back and draw seven again. Once a player keeps after N mulligans, they
put N cards from that hand on the bottom of their library in any order.
Players decide in turn order, starting with the player who goes first.
"""

from __future__ import annotations

import copy
import random
from typing import Callable, List, Optional, Sequence

from manamind.core.game_state import Card, GameState, Player

OPENING_HAND_SIZE = 7

# keep(hand, mulligans_taken) -> True to keep this hand.
KeepPolicy = Callable[[List[Card], int], bool]
# bottom(hand, count) -> the cards to put on the bottom, len == count.
BottomPolicy = Callable[[List[Card], int], List[Card]]


def default_keep(hand: List[Card], mulligans: int) -> bool:
    """Keep two to five lands; take what you get after two mulligans."""
    if mulligans >= 2:
        return True
    lands = sum(1 for card in hand if card.is_land())
    return 2 <= lands <= 5


def default_bottom(hand: List[Card], count: int) -> List[Card]:
    """Bottom whatever the hand has most of: excess lands or top-end spells."""
    lands = [c for c in hand if c.is_land()]
    spells = sorted(
        (c for c in hand if not c.is_land()),
        key=lambda c: c.converted_mana_cost,
        reverse=True,
    )
    chosen: List[Card] = []
    while len(chosen) < count:
        target = len(hand) - count  # final hand size
        if lands and (len(lands) > (target + 1) // 2 or not spells):
            chosen.append(lands.pop())
        else:
            chosen.append(spells.pop(0))
    return chosen


def _draw(player: Player, count: int) -> None:
    for _ in range(min(count, player.library.size())):
        player.hand.add_card(player.library.cards.pop(0))


def london_mulligan(
    player: Player,
    rng: random.Random,
    keep: KeepPolicy = default_keep,
    bottom: BottomPolicy = default_bottom,
) -> int:
    """Deal an opening hand with London mulligans. Returns mulligans taken.

    Assumes the library is already shuffled and the hand is empty.
    """
    _draw(player, OPENING_HAND_SIZE)
    mulligans = 0
    while mulligans < OPENING_HAND_SIZE and not keep(
        list(player.hand.cards), mulligans
    ):
        player.library.cards.extend(player.hand.cards)
        player.hand.cards = []
        rng.shuffle(player.library.cards)
        _draw(player, OPENING_HAND_SIZE)
        mulligans += 1

    if mulligans:
        to_bottom = bottom(list(player.hand.cards), mulligans)
        if len(to_bottom) != mulligans:
            raise ValueError(
                f"bottom policy returned {len(to_bottom)} cards, "
                f"expected {mulligans}"
            )
        for card in to_bottom:
            player.hand.remove_card(card)
            player.library.add_card(card)
    return mulligans


def setup_game(
    decks: Sequence[Sequence[Card]],
    seed: Optional[int] = None,
    keep: KeepPolicy = default_keep,
    bottom: BottomPolicy = default_bottom,
    starting_player: int = 0,
) -> GameState:
    """Load both decks, shuffle with one seeded RNG, deal opening hands."""
    from manamind.core.game_state import create_empty_game_state

    if len(decks) != 2:
        raise ValueError("setup_game expects exactly two decks")
    rng = random.Random(seed)
    state = create_empty_game_state()
    for player, deck in zip(state.players, decks):
        player.library.cards = [copy.deepcopy(card) for card in deck]
        rng.shuffle(player.library.cards)

    for offset in range(2):
        pid = (starting_player + offset) % 2
        london_mulligan(state.players[pid], rng, keep, bottom)

    state.active_player = starting_player
    state.priority_player = starting_player
    return state
