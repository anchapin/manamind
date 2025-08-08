"""Combined policy-value network for ManaMind agent.

This module implements the core neural network architecture that combines
both policy (action prediction) and value (position evaluation) estimation
in a single network, similar to AlphaZero.
"""

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from manamind.core.game_state import GameStateEncoder
from manamind.models.components import AttentionLayer, ResidualBlock


class PolicyValueNetwork(nn.Module):
    """Combined policy-value network for Magic: The Gathering.

    This network takes a game state as input and outputs:
    1. Policy: Probability distribution over possible actions
    2. Value: Estimated probability of winning from this position

    Architecture is inspired by AlphaZero but adapted for MTG's complexity.
    """

    def __init__(
        self,
        state_dim: int = 2048,  # From GameStateEncoder output
        hidden_dim: int = 1024,
        num_residual_blocks: int = 8,
        num_attention_heads: int = 8,
        action_space_size: int = 10000,  # Maximum number of possible actions
        dropout_rate: float = 0.1,
        use_attention: bool = True,
    ):
        """Initialize the policy-value network.

        Args:
            state_dim: Dimension of encoded game state
            hidden_dim: Hidden dimension for residual blocks
            num_residual_blocks: Number of residual blocks in the backbone
            num_attention_heads: Number of attention heads (if using attention)
            action_space_size: Size of the action space
            dropout_rate: Dropout rate for regularization
            use_attention: Whether to use attention mechanisms
        """
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_space_size = action_space_size
        self.use_attention = use_attention

        # Game state encoder
        self.state_encoder = GameStateEncoder(output_dim=state_dim)

        # Input projection
        self.input_projection = nn.Linear(state_dim, hidden_dim)

        # Backbone network - stack of residual blocks
        self.backbone = nn.ModuleList(
            [
                ResidualBlock(hidden_dim, dropout_rate)
                for _ in range(num_residual_blocks)
            ]
        )

        # Optional attention layer
        if use_attention:
            self.attention = AttentionLayer(
                hidden_dim, num_attention_heads, dropout_rate
            )

        # Policy head - predicts action probabilities
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, action_space_size),
        )

        # Value head - predicts win probability
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),  # Output in [-1, 1] range
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, game_state: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            game_state: Either a GameState object or pre-encoded tensor

        Returns:
            Tuple of (policy_logits, value):
                - policy_logits: Raw logits for action probabilities
                  [batch_size, action_space_size]
                - value: Estimated win probability [-1, 1] [batch_size, 1]
        """
        # Encode game state if needed
        if hasattr(game_state, "players"):  # GameState object
            x = self.state_encoder(game_state)
        else:  # Already encoded tensor
            x = game_state

        # Handle batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Input projection
        x = self.input_projection(x)
        x = F.relu(x)

        # Pass through residual blocks
        for block in self.backbone:
            x = block(x)

        # Optional attention
        if self.use_attention:
            # For attention, we need sequence dimension
            # Reshape to [batch, seq_len, hidden_dim] if needed
            if x.dim() == 2:
                x = x.unsqueeze(1)  # Add sequence dimension
            x = self.attention(x)
            x = x.squeeze(1)  # Remove sequence dimension

        # Policy and value heads
        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value

    def predict_action_probs(
        self, game_state: Any, temperature: float = 1.0
    ) -> torch.Tensor:
        """Get action probabilities from the policy head.

        Args:
            game_state: Game state to evaluate
            temperature: Temperature for softmax (higher = more exploration)

        Returns:
            Action probabilities [batch_size, action_space_size]
        """
        policy_logits, _ = self.forward(game_state)

        if temperature > 0:
            probs = F.softmax(policy_logits / temperature, dim=-1)
        else:
            # Deterministic - pick highest probability action
            probs = torch.zeros_like(policy_logits)
            probs.scatter_(-1, policy_logits.argmax(dim=-1, keepdim=True), 1.0)

        return probs

    def evaluate_position(self, game_state: Any) -> torch.Tensor:
        """Get position evaluation from the value head.

        Args:
            game_state: Game state to evaluate

        Returns:
            Position value [-1, 1] [batch_size, 1]
        """
        _, value = self.forward(game_state)
        return value

    def get_action_value_pairs(
        self, game_state: Any, legal_actions: Any
    ) -> List[Tuple[Any, float]]:
        """Get (action, value) pairs for all legal actions.

        This is useful for MCTS to get both policy priors and value estimates.

        Args:
            game_state: Current game state
            legal_actions: List of legal Action objects

        Returns:
            List of (action, prior_prob, value) tuples
        """
        policy_logits, value = self.forward(game_state)
        F.softmax(policy_logits, dim=-1)

        # TODO: Map legal_actions to network output indices
        # This requires the ActionSpace to provide action->index mapping

        action_values = []
        for action in legal_actions:
            # Placeholder - need to implement action encoding
            prior_prob = 1.0 / len(legal_actions)  # Uniform for now
            action_values.append((action, prior_prob))

        return action_values


class PolicyValueLoss(nn.Module):
    """Loss function for training the policy-value network.

    Combines:
    1. Cross-entropy loss for policy (action prediction)
    2. Mean squared error for value (outcome prediction)
    3. L2 regularization for weights
    """

    def __init__(self, value_weight: float = 1.0, l2_reg: float = 1e-4):
        """Initialize the loss function.

        Args:
            value_weight: Weight for value loss relative to policy loss
            l2_reg: L2 regularization coefficient
        """
        super().__init__()
        self.value_weight = value_weight
        self.l2_reg = l2_reg

    def forward(
        self,
        policy_logits: torch.Tensor,
        value_pred: torch.Tensor,
        target_policy: torch.Tensor,
        target_value: torch.Tensor,
        model: nn.Module,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute the combined loss.

        Args:
            policy_logits: Predicted policy logits
                [batch_size, action_space_size]
            value_pred: Predicted values [batch_size, 1]
            target_policy: Target policy distribution
                [batch_size, action_space_size]
            target_value: Target values [batch_size, 1]
            model: The model (for L2 regularization)

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains
                individual components
        """
        # Policy loss - cross entropy between predicted and target
        policy_loss = -torch.sum(
            target_policy * F.log_softmax(policy_logits, dim=-1), dim=-1
        )
        policy_loss = policy_loss.mean()

        # Value loss - MSE between predicted and target values
        value_loss = F.mse_loss(value_pred.squeeze(), target_value.squeeze())

        # L2 regularization
        l2_loss: torch.Tensor = torch.tensor(0.0)
        for param in model.parameters():
            l2_loss += torch.sum(param**2)
        l2_loss = l2_loss * self.l2_reg

        # Combined loss
        total_loss = policy_loss + self.value_weight * value_loss + l2_loss

        loss_dict = {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "l2_loss": float(l2_loss),
        }

        return total_loss, loss_dict


def create_policy_value_network(**kwargs: Any) -> PolicyValueNetwork:
    """Factory function to create a policy-value network with default settings.

    Args:
        **kwargs: Keyword arguments to override defaults

    Returns:
        Initialized PolicyValueNetwork
    """
    defaults = {
        "state_dim": 2048,
        "hidden_dim": 1024,
        "num_residual_blocks": 8,
        "num_attention_heads": 8,
        "action_space_size": 10000,
        "dropout_rate": 0.1,
        "use_attention": True,
    }
    defaults.update(kwargs)

    return PolicyValueNetwork(
        state_dim=int(defaults["state_dim"]),
        hidden_dim=int(defaults["hidden_dim"]),
        num_residual_blocks=int(defaults["num_residual_blocks"]),
        num_attention_heads=int(defaults["num_attention_heads"]),
        action_space_size=int(defaults["action_space_size"]),
        dropout_rate=float(defaults["dropout_rate"]),
        use_attention=bool(defaults["use_attention"]),
    )
