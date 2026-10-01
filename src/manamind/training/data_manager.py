"""Training data management for ManaMind.

NOT YET IMPLEMENTED. This module defines the interface only; every method
raises :class:`NotImplementedError`. See issue #19.
"""

from pathlib import Path
from typing import Any, Iterator, Sequence, Tuple


class TrainingDataManager:
    """Owns persistence and batching of self-play training examples.

    A training example is a ``(state, policy_target, value_target)`` triple
    produced by one decision point in a self-play game.
    """

    def __init__(self, data_dir: Path, buffer_size: int = 100_000) -> None:
        self.data_dir = Path(data_dir)
        self.buffer_size = buffer_size

    def add_examples(self, examples: Sequence[Any]) -> None:
        """Append examples to the replay buffer, evicting oldest if full."""
        raise NotImplementedError("TrainingDataManager is not implemented yet")

    def sample_batch(self, batch_size: int) -> Tuple[Any, Any, Any]:
        """Draw a random batch of
        ``(states, policy_targets, value_targets)``.
        """
        raise NotImplementedError("TrainingDataManager is not implemented yet")

    def iter_batches(self, batch_size: int) -> Iterator[Tuple[Any, Any, Any]]:
        """Iterate over the buffer once in shuffled batches."""
        raise NotImplementedError("TrainingDataManager is not implemented yet")

    def save(self, path: Path) -> None:
        """Persist the buffer to disk."""
        raise NotImplementedError("TrainingDataManager is not implemented yet")

    def load(self, path: Path) -> None:
        """Restore a buffer previously written by :meth:`save`."""
        raise NotImplementedError("TrainingDataManager is not implemented yet")

    def __len__(self) -> int:
        raise NotImplementedError("TrainingDataManager is not implemented yet")
