"""Rules engines for manamind.

`simple` implements a deliberately small, fully-specified subset of Magic
used to validate the training loop end to end (see issue #20).
"""

from manamind.rules.simple import (
    SIMPLE_CARD_POOL,
    SimpleRules,
    build_simple_deck,
    create_simple_game_start,
)

__all__ = [
    "SIMPLE_CARD_POOL",
    "SimpleRules",
    "build_simple_deck",
    "create_simple_game_start",
]
