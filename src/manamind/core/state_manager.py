"""Optimized state management for efficient MCTS and training.

This module provides memory-efficient game state management with copy-on-write
semantics, incremental updates, and caching for high-performance training.
"""

import copy
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from manamind.core.action import Action
from manamind.core.game_state import GameState


@dataclass
class StateHash:
    """Lightweight hash representation of game state."""

    turn_number: int
    phase: str
    active_player: int
    priority_player: int
    player_lives: Tuple[int, int]
    zone_sizes: Tuple[int, ...]  # Sizes of all zones
    stack_size: int

    def __post_init__(self):
        self._hash = hash(
            (
                self.turn_number,
                self.phase,
                self.active_player,
                self.priority_player,
                self.player_lives,
                self.zone_sizes,
                self.stack_size,
            )
        )

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other) -> bool:
        if not isinstance(other, StateHash):
            return False
        return self._hash == other._hash


@dataclass
class StateDelta:
    """Represents changes between game states for incremental updates."""

    action: Action
    timestamp: float

    # Changed components
    player_changes: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    zone_changes: Dict[Tuple[int, str], Dict[str, Any]] = field(
        default_factory=dict
    )
    global_changes: Dict[str, Any] = field(default_factory=dict)

    # For efficient rollback
    reverse_delta: Optional["StateDelta"] = None


class CopyOnWriteGameState:
    """Game state with copy-on-write semantics for memory efficiency."""

    def __init__(self, base_state: Optional["CopyOnWriteGameState"] = None):
        self._refs_lock = threading.Lock()

        if base_state is None:
            # Create new state
            self._data = GameState.create_empty_game_state()
            self._ref_count = 1
            self._is_cow = False
            self._parent = None
        else:
            # Share data with parent
            self._data = base_state._data
            with base_state._refs_lock:
                base_state._ref_count += 1
            self._ref_count = 1
            self._is_cow = True
            self._parent = weakref.ref(base_state)

    def _ensure_writable(self):
        """Ensure this state is writable (copy-on-write)."""
        if self._is_cow:
            # Make a deep copy
            self._data = copy.deepcopy(self._data)
            self._is_cow = False
            self._parent = None

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to underlying state."""
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        return getattr(self._data, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Handle attribute setting with COW semantics."""
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._ensure_writable()
            setattr(self._data, name, value)

    def copy(self) -> "CopyOnWriteGameState":
        """Create a COW copy."""
        return CopyOnWriteGameState(self)

    def get_hash(self) -> StateHash:
        """Get lightweight hash of current state."""
        return StateHash(
            turn_number=self._data.turn_number,
            phase=self._data.phase,
            active_player=self._data.active_player,
            priority_player=self._data.priority_player,
            player_lives=(
                self._data.players[0].life,
                self._data.players[1].life,
            ),
            zone_sizes=tuple(
                len(getattr(player, zone).cards)
                for player in self._data.players
                for zone in [
                    "hand",
                    "battlefield",
                    "graveyard",
                    "library",
                    "exile",
                ]
            ),
            stack_size=len(self._data.stack),
        )


class IncrementalStateManager:
    """Manages incremental state updates for efficient MCTS rollouts."""

    def __init__(self, base_state: GameState):
        self.base_state = base_state
        self.delta_stack: List[StateDelta] = []
        self._current_state: Optional[GameState] = None
        self._state_cache: Dict[int, GameState] = {}
        # Cache states at different depths
        self._max_cache_size = 100

    def push_action(self, action: Action) -> GameState:
        """Apply action and return new state."""
        # Compute delta
        old_state = self.current_state()
        new_state = action.execute(old_state)

        delta = self._compute_delta(action, old_state, new_state)
        self.delta_stack.append(delta)

        # Cache state if stack not too deep
        if len(self.delta_stack) < self._max_cache_size:
            self._state_cache[len(self.delta_stack)] = new_state

        self._current_state = new_state
        return new_state

    def pop_action(self) -> Optional[GameState]:
        """Rollback last action."""
        if not self.delta_stack:
            return None

        delta = self.delta_stack.pop()

        # Check cache first
        cache_key = len(self.delta_stack)
        if cache_key in self._state_cache:
            self._current_state = self._state_cache[cache_key]
            return self._current_state

        # Apply reverse delta
        if delta.reverse_delta:
            current = self.current_state()
            self._current_state = self._apply_reverse_delta(
                current, delta.reverse_delta
            )
            return self._current_state

        # Fallback: recompute from base
        self._current_state = None
        return self.current_state()

    def current_state(self) -> GameState:
        """Get current state by applying all deltas."""
        if self._current_state is not None:
            return self._current_state

        # Check cache
        cache_key = len(self.delta_stack)
        if cache_key in self._state_cache:
            self._current_state = self._state_cache[cache_key]
            return self._current_state

        # Recompute from base
        state = copy.deepcopy(self.base_state)
        for delta in self.delta_stack:
            state = delta.action.execute(state)

        self._current_state = state
        return state

    def _compute_delta(
        self, action: Action, old_state: GameState, new_state: GameState
    ) -> StateDelta:
        """Compute delta between states."""
        delta = StateDelta(action=action, timestamp=action.timestamp)

        # Compare players
        for i, (old_player, new_player) in enumerate(
            zip(old_state.players, new_state.players)
        ):
            changes = {}
            if old_player.life != new_player.life:
                changes["life"] = (old_player.life, new_player.life)
            if old_player.mana_pool != new_player.mana_pool:
                changes["mana_pool"] = (
                    old_player.mana_pool,
                    new_player.mana_pool,
                )
            if changes:
                delta.player_changes[i] = changes

        # Compare zones
        zone_names = ["hand", "battlefield", "graveyard", "library", "exile"]
        for i, (old_player, new_player) in enumerate(
            zip(old_state.players, new_state.players)
        ):
            for zone_name in zone_names:
                old_zone = getattr(old_player, zone_name)
                new_zone = getattr(new_player, zone_name)

                if len(old_zone.cards) != len(new_zone.cards):
                    delta.zone_changes[(i, zone_name)] = {
                        "size_change": len(new_zone.cards)
                        - len(old_zone.cards)
                    }

        # Global changes
        if old_state.phase != new_state.phase:
            delta.global_changes["phase"] = (old_state.phase, new_state.phase)
        if old_state.priority_player != new_state.priority_player:
            delta.global_changes["priority"] = (
                old_state.priority_player,
                new_state.priority_player,
            )

        return delta

    def _apply_reverse_delta(
        self, state: GameState, reverse_delta: StateDelta
    ) -> GameState:
        """Apply reverse delta (not implemented - use recomputation)."""
        # This is complex to implement correctly for all changes
        # For now, fall back to recomputation
        return self.current_state()

    def get_depth(self) -> int:
        """Get current depth in the delta stack."""
        return len(self.delta_stack)

    def clear_cache(self):
        """Clear state cache to free memory."""
        self._state_cache.clear()
        self._current_state = None


class StatePool:
    """Pool of reusable game state objects to reduce allocations."""

    def __init__(self, initial_size: int = 100):
        self._available_states: List[GameState] = []
        self._in_use: Set[int] = set()
        self._lock = threading.Lock()

        # Pre-allocate states
        for _ in range(initial_size):
            state = GameState.create_empty_game_state()
            self._available_states.append(state)

    def acquire(self) -> GameState:
        """Get a state from the pool."""
        with self._lock:
            if self._available_states:
                state = self._available_states.pop()
                self._in_use.add(id(state))
                return state

        # Pool exhausted, create new state
        state = GameState.create_empty_game_state()
        with self._lock:
            self._in_use.add(id(state))
        return state

    def release(self, state: GameState):
        """Return a state to the pool."""
        state_id = id(state)
        with self._lock:
            if state_id in self._in_use:
                self._in_use.remove(state_id)
                # Reset state to clean condition
                self._reset_state(state)
                self._available_states.append(state)

    def _reset_state(self, state: GameState):
        """Reset state to initial condition."""
        state.turn_number = 1
        state.phase = "main"
        state.active_player = 0
        state.priority_player = 0

        for player in state.players:
            player.life = 20
            player.mana_pool.clear()
            player.lands_played_this_turn = 0

            # Clear zones
            for zone_name in [
                "hand",
                "battlefield",
                "graveyard",
                "library",
                "exile",
            ]:
                zone = getattr(player, zone_name)
                zone.cards.clear()

        state.stack.clear()
        state.history.clear()


class BatchStateProcessor:
    """Process multiple states in parallel for training efficiency."""

    def __init__(self, max_workers: int = 4, batch_size: int = 64):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.state_pool = StatePool()

    def process_states_parallel(
        self, states: List[GameState], operation: str, **kwargs
    ) -> List[Any]:
        """Process multiple states in parallel."""

        if operation == "encode":
            return self._batch_encode(states, **kwargs)
        elif operation == "legal_actions":
            return self._batch_legal_actions(states, **kwargs)
        elif operation == "evaluate":
            return self._batch_evaluate(states, **kwargs)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _batch_encode(
        self, states: List[GameState], encoder=None
    ) -> List[torch.Tensor]:
        """Encode multiple states in parallel."""
        if encoder is None:
            raise ValueError("Encoder required for batch encoding")

        # Group states by structure for efficient batching
        batches = self._create_batches(states)

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for batch in batches:
                future = executor.submit(encoder.encode_batch, batch)
                futures.append(future)

            for future in futures:
                batch_encodings = future.result()
                results.extend(batch_encodings.unbind(0))

        return results

    def _batch_legal_actions(
        self, states: List[GameState], action_space=None
    ) -> List[List[Action]]:
        """Generate legal actions for multiple states in parallel."""
        if action_space is None:
            raise ValueError(
                "ActionSpace required for legal action generation"
            )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(action_space.get_legal_actions, state)
                for state in states
            ]
            return [future.result() for future in futures]

    def _batch_evaluate(
        self, states: List[GameState], evaluator=None
    ) -> List[float]:
        """Evaluate multiple states in parallel."""
        if evaluator is None:
            raise ValueError("Evaluator required for batch evaluation")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(evaluator.evaluate_state, state)
                for state in states
            ]
            return [future.result() for future in futures]

    def _create_batches(
        self, states: List[GameState]
    ) -> List[List[GameState]]:
        """Group states into batches for processing."""
        batches = []
        for i in range(0, len(states), self.batch_size):
            batch = states[i: i + self.batch_size]
            batches.append(batch)
        return batches


class TranspositionTable:
    """Transposition table for caching game state evaluations in MCTS."""

    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self._table: Dict[StateHash, Dict[str, Any]] = {}
        self._access_order: List[StateHash] = []
        self._lock = threading.RLock()

    def get(self, state_hash: StateHash, key: str) -> Optional[Any]:
        """Get cached value for state and key."""
        with self._lock:
            if state_hash in self._table and key in self._table[state_hash]:
                # Update access order (LRU)
                if state_hash in self._access_order:
                    self._access_order.remove(state_hash)
                self._access_order.append(state_hash)

                return self._table[state_hash][key]
        return None

    def put(self, state_hash: StateHash, key: str, value: Any):
        """Store value for state and key."""
        with self._lock:
            if state_hash not in self._table:
                self._table[state_hash] = {}

                # Evict oldest entry if table is full
                if len(self._table) > self.max_size:
                    oldest = self._access_order.pop(0)
                    del self._table[oldest]

            self._table[state_hash][key] = value

            # Update access order
            if state_hash in self._access_order:
                self._access_order.remove(state_hash)
            self._access_order.append(state_hash)

    def get_visit_count(self, state_hash: StateHash) -> int:
        """Get MCTS visit count for state."""
        return self.get(state_hash, "visit_count") or 0

    def get_value_estimate(self, state_hash: StateHash) -> Optional[float]:
        """Get value estimate for state."""
        return self.get(state_hash, "value_estimate")

    def get_action_values(
        self, state_hash: StateHash
    ) -> Optional[Dict[str, float]]:
        """Get action value estimates for state."""
        return self.get(state_hash, "action_values")

    def update_mcts_data(
        self,
        state_hash: StateHash,
        visit_count: int,
        value_estimate: float,
        action_values: Dict[str, float],
    ):
        """Update MCTS data for state."""
        self.put(state_hash, "visit_count", visit_count)
        self.put(state_hash, "value_estimate", value_estimate)
        self.put(state_hash, "action_values", action_values)

    def clear(self):
        """Clear all cached data."""
        with self._lock:
            self._table.clear()
            self._access_order.clear()

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self._table),
                "max_size": self.max_size,
                "hit_rate": 0.0,  # TODO: Track hit rate
            }


class StateCompressionManager:
    """Manages state compression for memory efficiency during training."""

    def __init__(self):
        self.compressor = StateCompressor()
        self._compressed_cache: Dict[StateHash, bytes] = {}

    def compress_state(self, state: GameState) -> bytes:
        """Compress state to bytes."""
        return self.compressor.compress(state)

    def decompress_state(self, data: bytes) -> GameState:
        """Decompress bytes to state."""
        return self.compressor.decompress(data)

    def cache_compressed_state(self, state: GameState) -> StateHash:
        """Cache compressed state and return hash."""
        state_hash = StateHash(
            turn_number=state.turn_number,
            phase=state.phase,
            active_player=state.active_player,
            priority_player=state.priority_player,
            player_lives=(state.players[0].life, state.players[1].life),
            zone_sizes=tuple(
                len(getattr(player, zone).cards)
                for player in state.players
                for zone in [
                    "hand",
                    "battlefield",
                    "graveyard",
                    "library",
                    "exile",
                ]
            ),
            stack_size=len(state.stack),
        )

        compressed_data = self.compress_state(state)
        self._compressed_cache[state_hash] = compressed_data

        return state_hash

    def retrieve_state(self, state_hash: StateHash) -> Optional[GameState]:
        """Retrieve state from compressed cache."""
        if state_hash in self._compressed_cache:
            compressed_data = self._compressed_cache[state_hash]
            return self.decompress_state(compressed_data)
        return None


class StateCompressor:
    """Handles compression/decompression of game states."""

    def compress(self, state: GameState) -> bytes:
        """Compress game state to bytes."""
        # Simple serialization for now
        # TODO: Implement more sophisticated compression
        import gzip
        import pickle

        data = pickle.dumps(state)
        return gzip.compress(data)

    def decompress(self, data: bytes) -> GameState:
        """Decompress bytes to game state."""
        import gzip
        import pickle

        decompressed = gzip.decompress(data)
        return pickle.loads(decompressed)
