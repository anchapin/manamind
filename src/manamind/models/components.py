"""Neural network components and building blocks.

This module contains reusable neural network components used throughout
the ManaMind architecture.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with layer normalization and dropout.
    
    This is a key building block for the deep networks, helping with
    gradient flow and training stability.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        dropout_rate: float = 0.1,
        activation: str = "relu"
    ):
        """Initialize the residual block.
        
        Args:
            hidden_dim: Hidden dimension
            dropout_rate: Dropout rate for regularization
            activation: Activation function ('relu', 'gelu', 'swish')
        """
        super().__init__()
        
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)  # Expand
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.layer_norm2 = nn.LayerNorm(hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)  # Contract
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "swish":
            self.activation = nn.SiLU()  # Swish = SiLU in PyTorch
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.
        
        Args:
            x: Input tensor [batch_size, hidden_dim]
            
        Returns:
            Output tensor with same shape as input
        """
        residual = x
        
        # First transformation
        x = self.layer_norm1(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        
        # Second transformation
        x = self.layer_norm2(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        
        # Residual connection
        return x + residual


class AttentionLayer(nn.Module):
    """Multi-head attention layer for processing sequences.
    
    This can be useful for attending to different parts of the game state,
    such as cards in hand, battlefield, etc.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        use_bias: bool = True
    ):
        """Initialize the attention layer.
        
        Args:
            hidden_dim: Hidden dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout_rate: Dropout rate for attention weights
            use_bias: Whether to use bias in linear layers
        """
        super().__init__()
        
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        self.query_projection = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        
        # Dropout and layer norm
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through multi-head attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            mask: Optional attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        residual = x
        
        # Apply layer norm first (pre-norm architecture)
        x = self.layer_norm(x)
        
        # Compute Q, K, V
        queries = self.query_projection(x)
        keys = self.key_projection(x)
        values = self.value_projection(x)
        
        # Reshape for multi-head attention
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for multi-head attention
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attention_scores.masked_fill_(mask == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, values)
        
        # Reshape back
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )
        
        # Output projection
        output = self.output_projection(attention_output)
        output = self.output_dropout(output)
        
        # Residual connection
        return output + residual


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence models.
    
    Adds positional information to embeddings, useful for processing
    sequences like game history or card sequences.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        max_seq_len: int = 5000,
        dropout_rate: float = 0.1
    ):
        """Initialize positional encoding.
        
        Args:
            hidden_dim: Hidden dimension
            max_seq_len: Maximum sequence length
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, hidden_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * 
                           (-math.log(10000.0) / hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            
        Returns:
            Input with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CardEmbedding(nn.Module):
    """Specialized embedding layer for Magic: The Gathering cards.
    
    This embedding layer combines multiple aspects of a card:
    - Card identity (name/ID)
    - Mana cost
    - Card type
    - Power/toughness (for creatures)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_mana_cost: int = 20,
        num_card_types: int = 100,
        max_power_toughness: int = 20
    ):
        """Initialize card embedding.
        
        Args:
            vocab_size: Size of card vocabulary
            embed_dim: Embedding dimension
            max_mana_cost: Maximum mana cost to handle
            num_card_types: Number of different card types
            max_power_toughness: Maximum power/toughness value
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Main card embedding
        self.card_embedding = nn.Embedding(vocab_size, embed_dim // 2)
        
        # Mana cost embedding
        self.mana_embedding = nn.Embedding(max_mana_cost + 1, embed_dim // 8)
        
        # Card type embedding
        self.type_embedding = nn.Embedding(num_card_types, embed_dim // 8)
        
        # Power/toughness embedding (for creatures)
        self.power_embedding = nn.Embedding(max_power_toughness + 1, embed_dim // 16)
        self.toughness_embedding = nn.Embedding(max_power_toughness + 1, embed_dim // 16)
        
        # Combine embeddings
        self.combiner = nn.Linear(embed_dim // 2 + embed_dim // 8 + embed_dim // 8 + 
                                embed_dim // 16 + embed_dim // 16, embed_dim)
    
    def forward(
        self,
        card_ids: torch.Tensor,
        mana_costs: torch.Tensor,
        card_types: torch.Tensor,
        powers: torch.Tensor,
        toughnesses: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through card embedding.
        
        Args:
            card_ids: Card ID tensor [batch_size, num_cards]
            mana_costs: Mana cost tensor [batch_size, num_cards]
            card_types: Card type tensor [batch_size, num_cards]
            powers: Power tensor [batch_size, num_cards]
            toughnesses: Toughness tensor [batch_size, num_cards]
            
        Returns:
            Combined card embeddings [batch_size, num_cards, embed_dim]
        """
        # Get individual embeddings
        card_emb = self.card_embedding(card_ids)
        mana_emb = self.mana_embedding(mana_costs)
        type_emb = self.type_embedding(card_types)
        power_emb = self.power_embedding(powers)
        tough_emb = self.toughness_embedding(toughnesses)
        
        # Concatenate and combine
        combined = torch.cat([card_emb, mana_emb, type_emb, power_emb, tough_emb], dim=-1)
        return self.combiner(combined)


class GatingMechanism(nn.Module):
    """Gating mechanism for controlling information flow.
    
    This can be useful for deciding which parts of the game state
    are most relevant for the current decision.
    """
    
    def __init__(self, hidden_dim: int):
        """Initialize gating mechanism.
        
        Args:
            hidden_dim: Hidden dimension
        """
        super().__init__()
        
        self.gate_projection = nn.Linear(hidden_dim, hidden_dim)
        self.content_projection = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gating mechanism.
        
        Args:
            x: Input tensor [batch_size, hidden_dim]
            
        Returns:
            Gated output tensor [batch_size, hidden_dim]
        """
        gate = torch.sigmoid(self.gate_projection(x))
        content = self.content_projection(x)
        return gate * content