"""Game state representation and encoding for Magic: The Gathering.

This module defines how MTG game states are represented internally and encoded
into neural network inputs.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, Field


class Card(BaseModel):
    """Represents a Magic: The Gathering card with enhanced state tracking."""

    # Core card data (from MTGJSON)
    name: str
    mana_cost: str = ""
    converted_mana_cost: int = 0
    card_types: List[str] = Field(default_factory=list)  # ["Creature", "Artifact"]
    subtypes: List[str] = Field(default_factory=list)  # ["Human", "Soldier"]
    supertypes: List[str] = Field(default_factory=list)  # ["Legendary", "Basic"]

    # Creature/Planeswalker stats
    power: Optional[int] = None
    toughness: Optional[int] = None
    base_power: Optional[int] = None
    base_toughness: Optional[int] = None
    loyalty: Optional[int] = None
    starting_loyalty: Optional[int] = None

    # Game state
    tapped: bool = False
    summoning_sick: bool = False
    counters: Dict[str, int] = Field(default_factory=dict)

    # Text and abilities
    oracle_text: str = ""
    oracle_id: str = ""
    abilities: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)

    # Combat state
    attacking: bool = False
    blocking: Optional[int] = None
    blocked_by: List[int] = Field(default_factory=list)

    # Ownership and control
    controller: int = 0
    owner: int = 0

    # Zone information
    zone: str = "unknown"
    zone_position: Optional[int] = None

    # Internal encoding IDs (assigned during preprocessing)
    card_id: Optional[int] = None
    instance_id: Optional[int] = None

    @property
    def card_type(self) -> str:
        """Backward compatibility - return joined card types."""
        return " ".join(self.card_types) if self.card_types else ""

    @card_type.setter
    def card_type(self, value: str) -> None:
        """Backward compatibility - parse card type string."""
        if value:
            self.card_types = value.split()

    def is_creature(self) -> bool:
        """Check if this card is a creature."""
        return "Creature" in self.card_types

    def is_land(self) -> bool:
        """Check if this card is a land."""
        return "Land" in self.card_types

    def is_instant_or_sorcery(self) -> bool:
        """Check if this card is an instant or sorcery."""
        return "Instant" in self.card_types or "Sorcery" in self.card_types

    def current_power(self) -> Optional[int]:
        """Get current power including modifications."""
        if self.power is None:
            return None
        return (
            self.power + self.counters.get("+1/+1", 0) - self.counters.get("-1/-1", 0)
        )

    def current_toughness(self) -> Optional[int]:
        """Get current toughness including modifications."""
        if self.toughness is None:
            return None
        return (
            self.toughness
            + self.counters.get("+1/+1", 0)
            - self.counters.get("-1/-1", 0)
        )


class Zone(BaseModel):
    """Represents a game zone (hand, battlefield, graveyard, etc.)."""

    cards: List[Card] = Field(default_factory=list)
    name: str
    owner: int  # Player ID (0 or 1)

    def add_card(self, card: Card) -> None:
        """Add a card to this zone."""
        self.cards.append(card)

    def remove_card(self, card: Card) -> bool:
        """Remove a card from this zone. Returns True if successful."""
        try:
            self.cards.remove(card)
            return True
        except ValueError:
            return False

    def size(self) -> int:
        """Return the number of cards in this zone."""
        return len(self.cards)


class Player(BaseModel):
    """Represents a player in the game."""

    player_id: int
    life: int = 20
    mana_pool: Dict[str, int] = Field(default_factory=dict)
    lands_played_this_turn: int = 0

    # Zones
    hand: Zone
    battlefield: Zone
    graveyard: Zone
    library: Zone
    exile: Zone
    command_zone: Zone

    def __init__(self, player_id: int, **data):
        # Initialize zones with proper player ownership
        zones = {
            "hand": Zone(name="hand", owner=player_id),
            "battlefield": Zone(name="battlefield", owner=player_id),
            "graveyard": Zone(name="graveyard", owner=player_id),
            "library": Zone(name="library", owner=player_id),
            "exile": Zone(name="exile", owner=player_id),
            "command_zone": Zone(name="command_zone", owner=player_id),
        }
        super().__init__(player_id=player_id, **zones, **data)

    def can_play_land(self) -> bool:
        """Check if the player can play a land this turn."""
        return self.lands_played_this_turn == 0  # Simplified rule

    def total_mana(self) -> int:
        """Calculate total available mana."""
        return sum(self.mana_pool.values())


@dataclass
class GameState:
    """Represents the complete state of a Magic: The Gathering game.

    This is the main data structure that captures all relevant information
    about the current game state that the AI agent needs to make decisions.
    """

    # Players (required field)
    players: Tuple[Player, Player]

    # Basic game info
    turn_number: int = 1
    phase: str = "main"  # untap, upkeep, draw, main, combat, main2, end
    priority_player: int = 0  # Which player has priority (0 or 1)
    active_player: int = 0  # Whose turn it is

    # Stack (spells and abilities waiting to resolve)
    stack: List[Dict[str, Any]] = field(default_factory=list)

    # Game history for neural network context
    history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def current_player(self) -> Player:
        """Get the player whose turn it is."""
        return self.players[self.active_player]

    @property
    def opponent(self) -> Player:
        """Get the opponent of the active player."""
        return self.players[1 - self.active_player]

    def is_game_over(self) -> bool:
        """Check if the game has ended."""
        return any(player.life <= 0 for player in self.players)

    def winner(self) -> Optional[int]:
        """Return the winner's player ID, or None if game is ongoing."""
        for i, player in enumerate(self.players):
            if player.life <= 0:
                return 1 - i  # The other player wins
        return None

    def copy(self) -> GameState:
        """Create a deep copy of the game state for simulation."""
        # Deep copy is expensive but necessary for correctness
        # TODO: Optimize with copy-on-write or incremental updates
        return copy.deepcopy(self)

    def compute_hash(self) -> int:
        """Compute a hash for transposition tables and state caching."""
        # Create hash from key game state components
        hash_components = [
            self.turn_number,
            self.phase,
            self.active_player,
            self.priority_player,
            tuple(p.life for p in self.players),
            tuple(
                len(getattr(p, zone).cards)
                for p in self.players
                for zone in ["hand", "battlefield", "graveyard", "library", "exile"]
            ),
            len(self.stack),
        ]
        return hash(tuple(hash_components))

    def get_features_for_encoding(self) -> Dict[str, Any]:
        """Extract features for neural network encoding."""
        return {
            "turn_number": self.turn_number,
            "phase": self.phase,
            "active_player": self.active_player,
            "priority_player": self.priority_player,
            "players": [
                {
                    "life": p.life,
                    "mana_pool": p.mana_pool,
                    "lands_played": p.lands_played_this_turn,
                    "zones": {
                        zone_name: [
                            {
                                "card_id": card.card_id,
                                "tapped": getattr(card, "tapped", False),
                                "counters": getattr(card, "counters", {}),
                                "power": card.current_power(),
                                "toughness": card.current_toughness(),
                            }
                            for card in getattr(p, zone_name).cards
                        ]
                        for zone_name in [
                            "hand",
                            "battlefield",
                            "graveyard",
                            "library",
                            "exile",
                        ]
                    },
                }
                for p in self.players
            ],
            "stack_size": len(self.stack),
        }


class GameStateEncoder(nn.Module):
    """Neural network module to encode game states into fixed-size tensors.

    This is a critical component that converts the complex, variable-size
    game state into a fixed-size numerical representation that can be
    processed by the policy and value networks.
    """

    def __init__(
        self,
        vocab_size: int = 50000,  # Number of unique cards/tokens
        embed_dim: int = 512,
        hidden_dim: int = 1024,
        num_zones: int = 6,  # hand, battlefield, graveyard, library, exile, command
        max_cards_per_zone: int = 200,
        output_dim: int = 2048,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_zones = num_zones
        self.max_cards_per_zone = max_cards_per_zone
        self.output_dim = output_dim

        # Card embedding layer
        self.card_embedding = nn.Embedding(vocab_size, embed_dim)

        # Zone encoders (one for each zone type)
        self.zone_encoders = nn.ModuleList(
            [
                nn.LSTM(
                    embed_dim, hidden_dim // 2, batch_first=True, bidirectional=True
                )
                for _ in range(num_zones)
            ]
        )

        # Player state encoder
        self.player_encoder = nn.Linear(20, hidden_dim)  # life, mana, etc.

        # Global state encoder (turn, phase, priority, etc.)
        self.global_encoder = nn.Linear(10, hidden_dim)

        # Final combination layer
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim * (2 * num_zones + 2 + 1), hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    def encode_zone(self, zone: Zone, zone_idx: int) -> torch.Tensor:
        """Encode a single zone into a fixed-size representation.

        Args:
            zone: The zone to encode
            zone_idx: Index of the zone type (for selecting the right encoder)

        Returns:
            Fixed-size tensor representing the zone
        """
        # Convert cards to IDs and pad/truncate to fixed size
        card_ids = []
        for card in zone.cards[: self.max_cards_per_zone]:
            card_ids.append(card.card_id or 0)  # Use 0 for unknown cards

        # Pad if necessary
        while len(card_ids) < self.max_cards_per_zone:
            card_ids.append(0)  # Padding token

        # Convert to tensor and embed
        card_tensor = torch.tensor(card_ids, dtype=torch.long).unsqueeze(0)
        embedded_cards = self.card_embedding(card_tensor)

        # Encode with LSTM
        lstm_out, (hidden, _) = self.zone_encoders[zone_idx](embedded_cards)

        # Use final hidden state as zone representation
        return hidden.view(-1)  # Flatten

    def encode_player(self, player: Player) -> torch.Tensor:
        """Encode player state (life, mana, etc.) into a tensor."""
        features = [
            float(player.life) / 20.0,  # Normalized life
            float(player.lands_played_this_turn),
            float(player.hand.size()) / 10.0,  # Normalized hand size
            float(player.battlefield.size()) / 20.0,  # Normalized board size
            float(player.graveyard.size()) / 50.0,  # Normalized graveyard size
        ]

        # Add mana pool features (WUBRG + colorless)
        mana_colors = ["W", "U", "B", "R", "G", "C"]
        for color in mana_colors:
            features.append(float(player.mana_pool.get(color, 0)) / 10.0)

        # Pad to expected size
        while len(features) < 20:
            features.append(0.0)

        return self.player_encoder(torch.tensor(features, dtype=torch.float32))

    def forward(self, game_state: GameState) -> torch.Tensor:
        """Encode a complete game state into a fixed-size tensor.

        Args:
            game_state: The game state to encode

        Returns:
            Fixed-size tensor representation of the game state
        """
        encoded_parts = []

        # Encode zones for both players
        zone_types = [
            "hand",
            "battlefield",
            "graveyard",
            "library",
            "exile",
            "command_zone",
        ]
        for player in game_state.players:
            for zone_idx, zone_name in enumerate(zone_types):
                zone = getattr(player, zone_name)
                zone_encoding = self.encode_zone(zone, zone_idx)
                encoded_parts.append(zone_encoding)

        # Encode player states
        for player in game_state.players:
            player_encoding = self.encode_player(player)
            encoded_parts.append(player_encoding)

        # Encode global game state
        global_features = [
            float(game_state.turn_number) / 20.0,  # Normalized turn
            float(game_state.active_player),
            float(game_state.priority_player),
            float(len(game_state.stack)) / 10.0,  # Normalized stack size
        ]

        # Encode phase as one-hot
        phases = ["untap", "upkeep", "draw", "main", "combat", "main2", "end"]
        for phase in phases:
            global_features.append(1.0 if game_state.phase == phase else 0.0)

        global_encoding = self.global_encoder(
            torch.tensor(global_features, dtype=torch.float32)
        )
        encoded_parts.append(global_encoding)

        # Combine all encodings
        combined = torch.cat(encoded_parts, dim=0)
        return self.combiner(combined.unsqueeze(0)).squeeze(0)


# Factory functions for creating common game states


def create_empty_game_state() -> GameState:
    """Create an empty game state for testing."""
    player0 = Player(player_id=0)
    player1 = Player(player_id=1)
    return GameState(players=(player0, player1))


def create_standard_game_start() -> GameState:
    """Create a game state representing the start of a standard game."""
    # TODO: Implement proper game start with shuffled libraries, opening hands, etc.
    return create_empty_game_state()
