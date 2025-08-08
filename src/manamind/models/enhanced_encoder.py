"""Enhanced neural network encoder for comprehensive MTG game states.

This module provides advanced encoding capabilities for the full complexity of
Magic: The Gathering, including multi-modal encoding, attention mechanisms,
and optimized representations.
"""

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn

from manamind.core.game_state import Card, GameState, Player, Zone


@dataclass
class EncoderConfig:
    """Configuration for the enhanced game state encoder."""

    # Card vocabulary and embeddings
    card_vocab_size: int = 50000
    embed_dim: int = 512

    # Architecture dimensions
    hidden_dim: int = 1024
    output_dim: int = 2048

    # Attention settings
    num_heads: int = 8
    num_layers: int = 4

    # Zone settings
    max_cards_per_zone: int = 200
    num_zones: int = 6

    # Optimization
    dropout: float = 0.1
    use_attention: bool = True
    use_layer_norm: bool = True


class CardEmbeddingSystem(nn.Module):
    """Advanced card embedding with structural and semantic features."""

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Core embeddings
        self.card_embedding = nn.Embedding(vocab_size, embed_dim // 4)
        self.type_embedding = nn.Embedding(100, embed_dim // 8)  # Card types
        self.cost_embedding = nn.Embedding(50, embed_dim // 8)  # Mana costs

        # Structural feature encoding
        self.power_embedding = nn.Linear(1, embed_dim // 16)
        self.toughness_embedding = nn.Linear(1, embed_dim // 16)
        self.loyalty_embedding = nn.Linear(1, embed_dim // 16)

        # State encoding
        self.state_encoder = nn.Linear(
            10, embed_dim // 8
        )  # Tapped, counters, etc.

        # Final projection
        self.projector = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, card: Card) -> torch.Tensor:
        """Encode a single card into a dense representation."""
        features = []

        # Card ID embedding
        card_id = card.card_id or 0
        card_emb = self.card_embedding(torch.tensor(card_id, dtype=torch.long))
        features.append(card_emb)

        # Card type embedding (simplified)
        type_id = (
            hash(" ".join(card.card_types)) % 100 if card.card_types else 0
        )
        type_emb = self.type_embedding(torch.tensor(type_id, dtype=torch.long))
        features.append(type_emb)

        # Mana cost embedding
        cmc_id = min(card.converted_mana_cost, 49)
        cost_emb = self.cost_embedding(torch.tensor(cmc_id, dtype=torch.long))
        features.append(cost_emb)

        # Power/Toughness/Loyalty
        if hasattr(card, "current_power") and card.current_power() is not None:
            power_emb = self.power_embedding(
                torch.tensor([float(card.current_power())])
            )
            features.append(power_emb)
        else:
            features.append(torch.zeros(self.embed_dim // 16))

        if (
            hasattr(card, "current_toughness")
            and card.current_toughness() is not None
        ):
            toughness_emb = self.toughness_embedding(
                torch.tensor([float(card.current_toughness())])
            )
            features.append(toughness_emb)
        else:
            features.append(torch.zeros(self.embed_dim // 16))

        if hasattr(card, "loyalty") and card.loyalty is not None:
            loyalty_emb = self.loyalty_embedding(
                torch.tensor([float(card.loyalty)])
            )
            features.append(loyalty_emb)
        else:
            features.append(torch.zeros(self.embed_dim // 16))

        # State features
        state_features = torch.tensor(
            [
                float(getattr(card, "tapped", False)),
                float(getattr(card, "summoning_sick", False)),
                float(getattr(card, "attacking", False)),
                float(len(getattr(card, "counters", {}))),
                float(card.controller if hasattr(card, "controller") else 0),
                # Pad to 10 features
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ][:10]
        )

        state_emb = self.state_encoder(state_features)
        features.append(state_emb)

        # Combine all features
        combined = torch.cat(features, dim=-1)

        # Project to final dimension and normalize
        output = self.projector(combined)
        return self.layer_norm(output)


class ZoneEncoder(nn.Module):
    """Base class for encoding different types of zones."""

    def __init__(self, config: EncoderConfig, zone_type: str):
        super().__init__()
        self.config = config
        self.zone_type = zone_type
        self.card_embedder = CardEmbeddingSystem(
            config.card_vocab_size, config.embed_dim
        )

    def forward(self, zone: Zone, player_id: int) -> torch.Tensor:
        """Encode a zone into a fixed-size representation."""
        if not zone.cards:
            return torch.zeros(self.config.hidden_dim)

        # Encode all cards in the zone
        card_embeddings = []
        for card in zone.cards[: self.config.max_cards_per_zone]:
            card_emb = self.card_embedder(card)
            card_embeddings.append(card_emb)

        if not card_embeddings:
            return torch.zeros(self.config.hidden_dim)

        # Stack embeddings
        embeddings_tensor = torch.stack(card_embeddings)

        # Zone-specific aggregation
        return self._aggregate_embeddings(embeddings_tensor, zone, player_id)

    def _aggregate_embeddings(
        self, embeddings: torch.Tensor, zone: Zone, player_id: int
    ) -> torch.Tensor:
        """Override in subclasses for zone-specific aggregation."""
        return embeddings.mean(dim=0)


class HandEncoder(ZoneEncoder):
    """Specialized encoder for hand zone with hidden information modeling."""

    def __init__(self, config: EncoderConfig):
        super().__init__(config, "hand")
        self.attention = nn.MultiheadAttention(
            config.embed_dim, config.num_heads
        )
        self.output_proj = nn.Linear(config.embed_dim, config.hidden_dim)

    def _aggregate_embeddings(
        self, embeddings: torch.Tensor, zone: Zone, player_id: int
    ) -> torch.Tensor:
        """Use attention to weight hand cards by importance."""
        if len(embeddings.shape) == 1:
            embeddings = embeddings.unsqueeze(0)

        # Self-attention over hand cards
        attn_output, _ = self.attention(embeddings, embeddings, embeddings)

        # Aggregate with weighted average
        hand_encoding = attn_output.mean(dim=0)

        return self.output_proj(hand_encoding)


class BattlefieldEncoder(ZoneEncoder):
    """Specialized encoder for battlefield with creature interactions."""

    def __init__(self, config: EncoderConfig):
        super().__init__(config, "battlefield")
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.embed_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim,
                dropout=config.dropout,
            ),
            num_layers=2,
        )
        self.output_proj = nn.Linear(config.embed_dim, config.hidden_dim)

    def _aggregate_embeddings(
        self, embeddings: torch.Tensor, zone: Zone, player_id: int
    ) -> torch.Tensor:
        """Model battlefield interactions with transformer."""
        if len(embeddings.shape) == 1:
            embeddings = embeddings.unsqueeze(0)

        # Add positional encoding for battlefield position
        seq_len = embeddings.shape[0]
        pos_encoding = (
            torch.arange(seq_len, dtype=torch.float).unsqueeze(1) / 100.0
        )
        pos_encoding = pos_encoding.expand(-1, embeddings.shape[1])
        embeddings = embeddings + pos_encoding

        # Apply transformer to model interactions
        battlefield_encoding = self.transformer(embeddings)

        # Aggregate battlefield state
        aggregated = battlefield_encoding.mean(dim=0)

        return self.output_proj(aggregated)


class SequentialZoneEncoder(ZoneEncoder):
    """Encoder for zones where order matters (graveyard, library, exile)."""

    def __init__(self, config: EncoderConfig):
        super().__init__(config, "sequential")
        self.lstm = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout,
        )

    def _aggregate_embeddings(
        self, embeddings: torch.Tensor, zone: Zone, player_id: int
    ) -> torch.Tensor:
        """Use LSTM to preserve order information."""
        if len(embeddings.shape) == 1:
            embeddings = embeddings.unsqueeze(0)

        # Add batch dimension
        embeddings = embeddings.unsqueeze(0)

        # LSTM encoding
        lstm_out, (hidden, _) = self.lstm(embeddings)

        # Use final hidden state as zone representation
        return hidden.view(-1)


class PlayerStateEncoder(nn.Module):
    """Encode individual player state (life, mana, etc.)."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        # Player feature encoding
        self.life_encoder = nn.Linear(1, 32)
        self.mana_encoder = nn.Linear(6, 64)  # WUBRG + colorless
        self.misc_encoder = nn.Linear(10, 64)  # Other features

        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(32 + 64 + 64, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

    def forward(self, player: Player, player_id: int) -> torch.Tensor:
        """Encode player state features."""
        # Life encoding (normalized)
        life_feature = torch.tensor([float(player.life) / 20.0])
        life_emb = self.life_encoder(life_feature)

        # Mana pool encoding
        mana_colors = ["W", "U", "B", "R", "G", "C"]
        mana_features = torch.tensor(
            [
                float(player.mana_pool.get(color, 0)) / 10.0
                for color in mana_colors
            ]
        )
        mana_emb = self.mana_encoder(mana_features)

        # Miscellaneous features
        misc_features = torch.tensor(
            [
                float(player.lands_played_this_turn),
                float(player.hand.size()) / 10.0,
                float(player.battlefield.size()) / 20.0,
                float(player.graveyard.size()) / 50.0,
                float(player.library.size()) / 60.0,
                float(player.exile.size()) / 20.0,
                float(player_id),  # Player identity
                0.0,
                0.0,
                0.0,  # Reserved for future features
            ]
        )
        misc_emb = self.misc_encoder(misc_features)

        # Combine all features
        combined = torch.cat([life_emb, mana_emb, misc_emb], dim=0)
        return self.output_proj(combined)


class GlobalStateEncoder(nn.Module):
    """Encode global game state (turn, phase, stack, etc.)."""

    def __init__(self, config: EncoderConfig):
        super().__init__()

        # Phase/step encoding
        self.phase_embedding = nn.Embedding(10, 64)

        # Turn and priority encoding
        self.turn_encoder = nn.Linear(4, 64)

        # Stack encoding
        self.stack_encoder = nn.Linear(5, 64)

        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(64 + 64 + 64, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

    def forward(self, game_state: GameState) -> torch.Tensor:
        """Encode global game state."""
        # Phase encoding
        phases = ["untap", "upkeep", "draw", "main", "combat", "main2", "end"]
        phase_id = (
            phases.index(game_state.phase) if game_state.phase in phases else 0
        )
        phase_emb = self.phase_embedding(
            torch.tensor(phase_id, dtype=torch.long)
        )

        # Turn and priority features
        turn_features = torch.tensor(
            [
                float(game_state.turn_number) / 20.0,
                float(game_state.active_player),
                float(game_state.priority_player),
                (
                    1.0
                    if game_state.active_player == game_state.priority_player
                    else 0.0
                ),
            ]
        )
        turn_emb = self.turn_encoder(turn_features)

        # Stack features
        stack_features = torch.tensor(
            [
                float(len(game_state.stack)) / 10.0,
                0.0,
                0.0,
                0.0,
                0.0,  # Reserved for stack content analysis
            ]
        )
        stack_emb = self.stack_encoder(stack_features)

        # Combine features
        combined = torch.cat([phase_emb, turn_emb, stack_emb], dim=0)
        return self.output_proj(combined)


class StateFusionNetwork(nn.Module):
    """Fuse all encoded components into final game state representation."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        # Attention fusion
        if config.use_attention:
            self.cross_attention = nn.MultiheadAttention(
                config.hidden_dim, config.num_heads
            )

        # Final fusion layers
        fusion_input_dim = config.hidden_dim * (
            2 * 6 + 2 + 1
        )  # zones + players + global

        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, config.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(config.output_dim)
        else:
            self.layer_norm = None

    def forward(
        self,
        zone_encodings: Dict[int, Dict[str, torch.Tensor]],
        player_encodings: List[torch.Tensor],
        global_encoding: torch.Tensor,
        stack_encoding: torch.Tensor = None,
        combat_encoding: torch.Tensor = None,
    ) -> torch.Tensor:
        """Fuse all encodings into final representation."""

        # Collect all encodings
        all_encodings = []

        # Zone encodings for both players
        zone_names = [
            "hand",
            "battlefield",
            "graveyard",
            "library",
            "exile",
            "command_zone",
        ]
        for player_id in [0, 1]:
            for zone_name in zone_names:
                if (
                    player_id in zone_encodings
                    and zone_name in zone_encodings[player_id]
                ):
                    all_encodings.append(zone_encodings[player_id][zone_name])
                else:
                    # Add zero encoding for missing zones
                    all_encodings.append(torch.zeros(self.config.hidden_dim))

        # Player encodings
        all_encodings.extend(player_encodings)

        # Global encoding
        all_encodings.append(global_encoding)

        # Stack encoding (if provided)
        if stack_encoding is not None:
            all_encodings.append(stack_encoding)
        else:
            all_encodings.append(torch.zeros(self.config.hidden_dim))

        # Ensure we have the expected number of encodings
        expected_count = 2 * 6 + 2 + 1 + 1  # zones + players + global + stack
        while len(all_encodings) < expected_count:
            all_encodings.append(torch.zeros(self.config.hidden_dim))

        # Concatenate all encodings
        fused_representation = torch.cat(all_encodings[:expected_count], dim=0)

        # Apply fusion network
        output = self.fusion_network(fused_representation)

        # Apply layer norm if configured
        if self.layer_norm is not None:
            output = self.layer_norm(output)

        return output


class EnhancedGameStateEncoder(nn.Module):
    """Complete enhanced game state encoder integrating all components."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        # Component encoders
        self.zone_encoders = nn.ModuleDict(
            {
                "hand": HandEncoder(config),
                "battlefield": BattlefieldEncoder(config),
                "graveyard": SequentialZoneEncoder(config),
                "library": SequentialZoneEncoder(config),
                "exile": SequentialZoneEncoder(config),
                "command_zone": SequentialZoneEncoder(config),
            }
        )

        self.player_encoder = PlayerStateEncoder(config)
        self.global_encoder = GlobalStateEncoder(config)
        self.state_fusion = StateFusionNetwork(config)

    def forward(self, game_state: GameState) -> torch.Tensor:
        """Encode complete game state into fixed-size tensor."""
        # Encode zones for both players
        zone_encodings = {}
        for player_id, player in enumerate(game_state.players):
            player_zones = {}
            for zone_name in [
                "hand",
                "battlefield",
                "graveyard",
                "library",
                "exile",
                "command_zone",
            ]:
                zone = getattr(player, zone_name)
                encoder = self.zone_encoders[zone_name]
                player_zones[zone_name] = encoder(zone, player_id)
            zone_encodings[player_id] = player_zones

        # Encode players
        player_encodings = [
            self.player_encoder(player, player_id)
            for player_id, player in enumerate(game_state.players)
        ]

        # Encode global state
        global_encoding = self.global_encoder(game_state)

        # Fuse all components
        return self.state_fusion(
            zone_encodings, player_encodings, global_encoding
        )

    def encode_batch(self, game_states: List[GameState]) -> torch.Tensor:
        """Encode multiple game states in batch."""
        encodings = []
        for state in game_states:
            encoding = self.forward(state)
            encodings.append(encoding)
        return torch.stack(encodings)
