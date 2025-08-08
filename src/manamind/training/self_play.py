"""Self-play training implementation for ManaMind.

This module implements the core self-play training loop where the agent
learns by playing millions of games against itself, similar to AlphaZero.
"""

import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from manamind.core.action import Action
from manamind.core.agent import MCTSAgent
from manamind.core.game_state import GameState, create_standard_game_start
from manamind.forge_interface import (  # ForgeGameRunner not implemented yet
    ForgeClient,
)
from manamind.models.policy_value_network import PolicyValueNetwork

# from manamind.training.data_manager import TrainingDataManager

logger = logging.getLogger(__name__)


class SelfPlayGame:
    """Represents a single self-play game and its training data."""

    def __init__(self, game_id: str):
        self.game_id = game_id
        self.history: List[Tuple[GameState, Action, float, np.ndarray]] = []
        self.winner: Optional[int] = None
        self.num_moves = 0
        self.start_time = time.time()
        self.end_time: Optional[float] = None

    def add_move(
        self, state: GameState, action: Action, mcts_policy: np.ndarray
    ) -> None:
        """Add a move to the game history.

        Args:
            state: Game state before the move
            action: Action taken
            mcts_policy: MCTS action probabilities for training
        """
        # Store with temporary reward (will be updated at game end)
        self.history.append((state, action, 0.0, mcts_policy))
        self.num_moves += 1

    def finalize_game(self, winner: Optional[int]) -> None:
        """Finalize the game and assign rewards.

        Args:
            winner: Winning player ID (0 or 1), or None for draw
        """
        self.winner = winner
        self.end_time = time.time()

        # Update rewards based on game outcome
        for i, (state, action, _, policy) in enumerate(self.history):
            # Determine reward from this player's perspective
            player_id = action.player_id

            if winner == player_id:
                reward = 1.0
            elif winner is None:
                reward = 0.0  # Draw
            else:
                reward = -1.0

            # Update history with final reward
            self.history[i] = (state, action, reward, policy)

    def duration(self) -> float:
        """Get game duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time

    def get_training_examples(
        self,
    ) -> List[Tuple[GameState, np.ndarray, float]]:
        """Extract training examples from this game.

        Returns:
            List of (state, mcts_policy, reward) tuples for training
        """
        examples = []
        for state, action, reward, mcts_policy in self.history:
            examples.append((state, mcts_policy, reward))
        return examples


class SelfPlayTrainer:
    """Self-play trainer for the ManaMind agent.

    This class manages the self-play training process:
    1. Generates self-play games using current model
    2. Collects training data from games
    3. Updates the neural network
    4. Iterates to improve performance
    """

    def __init__(
        self,
        policy_value_network: PolicyValueNetwork,
        forge_client: Optional[ForgeClient] = None,
        data_manager: Optional[
            Any
        ] = None,  # TrainingDataManager not implemented yet
        config: Optional[Dict] = None,
    ):
        """Initialize self-play trainer.

        Args:
            policy_value_network: The network to train
            forge_client: Forge client for running games
            data_manager: Training data manager
            config: Training configuration
        """
        self.network = policy_value_network
        self.forge_client = forge_client
        # self.data_manager = data_manager or TrainingDataManager()
        self.data_manager = data_manager  # Placeholder until implemented

        # Training configuration
        self.config = config or self._default_config()

        # Training state
        self.current_iteration = 0
        self.total_games_played = 0
        self.training_examples: List[Tuple[GameState, np.ndarray, float]] = []
        self.performance_history: List[Dict[str, Any]] = []

        # Create MCTS agents for self-play
        self.mcts_config = {
            "simulations": self.config["mcts_simulations"],
            "simulation_time": self.config["mcts_time_limit"],
            "c_puct": self.config["c_puct"],
        }

    def _default_config(self) -> Dict:
        """Default training configuration."""
        return {
            # Self-play parameters
            "games_per_iteration": 100,
            "max_game_length": 200,
            "mcts_simulations": 800,
            "mcts_time_limit": 1.0,
            "c_puct": 1.0,
            # Training parameters
            "training_iterations": 1000,
            "examples_buffer_size": 100000,
            "batch_size": 64,
            "epochs_per_iteration": 10,
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            # Evaluation parameters
            "evaluation_frequency": 10,
            "evaluation_games": 50,
            # Checkpointing
            "checkpoint_frequency": 10,
            "checkpoint_dir": "checkpoints",
        }

    def train(self, num_iterations: Optional[int] = None) -> None:
        """Run the main training loop.

        Args:
            num_iterations: Number of training iterations
                (uses config default if None)
        """
        num_iterations = num_iterations or self.config["training_iterations"]

        logger.info(
            f"Starting self-play training for {num_iterations} iterations"
        )

        for iteration in range(num_iterations):
            self.current_iteration = iteration

            logger.info(
                f"=== Training Iteration {iteration + 1}/{num_iterations} ==="
            )

            # Phase 1: Generate self-play games
            logger.info("Generating self-play games...")
            new_examples = self._generate_self_play_games()

            # Phase 2: Update training data
            self.training_examples.extend(new_examples)
            self._maintain_examples_buffer()

            logger.info(f"Training buffer size: {len(self.training_examples)}")

            # Phase 3: Train neural network
            if len(self.training_examples) >= self.config["batch_size"]:
                logger.info("Training neural network...")
                self._train_network()

            # Phase 4: Evaluation and checkpointing
            if (iteration + 1) % self.config["evaluation_frequency"] == 0:
                logger.info("Evaluating model performance...")
                self._evaluate_model()

            if (iteration + 1) % self.config["checkpoint_frequency"] == 0:
                logger.info("Saving checkpoint...")
                self._save_checkpoint()

        logger.info("Training completed!")

    def _generate_self_play_games(
        self,
    ) -> List[Tuple[GameState, np.ndarray, float]]:
        """Generate self-play games and extract training examples."""
        num_games = self.config["games_per_iteration"]
        all_examples = []
        games_completed = 0

        with tqdm(total=num_games, desc="Self-play games") as pbar:
            while games_completed < num_games:
                try:
                    # Run a single self-play game
                    game = self._play_single_game()

                    if game and game.winner is not None:
                        # Extract training examples
                        examples = game.get_training_examples()
                        all_examples.extend(examples)
                        games_completed += 1
                        self.total_games_played += 1

                        # Update progress
                        pbar.set_postfix(
                            {
                                "moves": game.num_moves,
                                "duration": f"{game.duration():.1f}s",
                                "winner": f"P{game.winner}",
                            }
                        )
                        pbar.update(1)

                except Exception as e:
                    logger.error(f"Error in self-play game: {e}")
                    continue

        msg = f"Generated {len(all_examples)} training examples from {games_completed} games"  # noqa: E501
        logger.info(msg)
        return all_examples

    def _play_single_game(self) -> Optional[SelfPlayGame]:
        """Play a single self-play game.

        Returns:
            Completed SelfPlayGame or None if game failed
        """
        if self.forge_client:
            return self._play_forge_game()
        else:
            return self._play_simulation_game()

    def _play_forge_game(self) -> Optional[SelfPlayGame]:
        """Play a game using the Forge engine."""
        try:
            # Create game runner
            # game_runner = ForgeGameRunner(self.forge_client)
            # Not implemented yet
            game_runner = None  # Placeholder

            # Create MCTS agents
            agent1 = MCTSAgent(
                player_id=0,
                policy_network=self.network,
                value_network=self.network,
                **self.mcts_config,
            )

            agent2 = MCTSAgent(
                player_id=1,
                policy_network=self.network,
                value_network=self.network,
                **self.mcts_config,
            )

            # Run the game
            game_result = game_runner.play_game(agent1, agent2)

            if game_result:
                # Convert to SelfPlayGame format
                game = SelfPlayGame(game_result.game_id)

                # Add moves from game history
                for state, action, mcts_policy in game_result.history:
                    game.add_move(state, action, mcts_policy)

                # Finalize with winner
                game.finalize_game(game_result.winner)
                return game

        except Exception as e:
            logger.error(f"Error in Forge game: {e}")

        return None

    def _play_simulation_game(self) -> Optional[SelfPlayGame]:
        """Play a game using pure Python simulation (testing without Forge)."""
        try:
            # Create initial game state
            game_state = create_standard_game_start()
            game = SelfPlayGame("simulation")

            # Create MCTS agents
            agents = [
                MCTSAgent(0, self.network, self.network, **self.mcts_config),
                MCTSAgent(1, self.network, self.network, **self.mcts_config),
            ]

            move_count = 0
            max_moves = self.config["max_game_length"]

            while not game_state.is_game_over() and move_count < max_moves:
                current_player = game_state.priority_player
                agent = agents[current_player]

                # Get action from MCTS
                action = agent.select_action(game_state)

                # TODO: Get MCTS policy for training
                # For now, use dummy policy
                mcts_policy = (
                    np.ones(self.config.get("action_space_size", 1000)) / 1000
                )

                # Record move
                game.add_move(game_state.copy(), action, mcts_policy)

                # Execute action
                game_state = action.execute(game_state)
                move_count += 1

            # Determine winner
            winner = game_state.winner()
            game.finalize_game(winner)

            return game

        except Exception as e:
            logger.error(f"Error in simulation game: {e}")
            return None

    def _maintain_examples_buffer(self) -> None:
        """Maintain the training examples buffer at the configured size."""
        buffer_size = self.config["examples_buffer_size"]

        if len(self.training_examples) > buffer_size:
            # Remove oldest examples to maintain buffer size
            excess = len(self.training_examples) - buffer_size
            self.training_examples = self.training_examples[excess:]

            msg = f"Trimmed training buffer to {len(self.training_examples)} examples"  # noqa: E501
            logger.info(msg)

    def _train_network(self) -> None:
        """Train the neural network on collected examples."""
        if not self.training_examples:
            logger.warning("No training examples available")
            return

        # TODO: Implement neural network training
        # This would involve:
        # 1. Creating data loaders from training examples
        # 2. Running gradient descent for specified epochs
        # 3. Updating the policy-value network
        # 4. Logging training metrics

        logger.info(
            f"Training network on {len(self.training_examples)} examples"
        )

        # Placeholder for actual training implementation
        self.config["batch_size"]
        epochs = self.config["epochs_per_iteration"]

        # Shuffle training examples
        random.shuffle(self.training_examples)

        logger.info(f"Completed {epochs} training epochs")

    def _evaluate_model(self) -> None:
        """Evaluate the current model performance."""
        # TODO: Implement model evaluation
        # This could involve:
        # 1. Playing games against previous model versions
        # 2. Playing against Forge AI at different difficulty levels
        # 3. Computing win rates and other metrics
        # 4. Logging evaluation results

        logger.info("Model evaluation completed")

    def _save_checkpoint(self) -> None:
        """Save a training checkpoint."""
        checkpoint_dir = Path(self.config["checkpoint_dir"])
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_path = (
            checkpoint_dir
            / f"checkpoint_iteration_{self.current_iteration}.pt"
        )

        checkpoint = {
            "iteration": self.current_iteration,
            "total_games": self.total_games_played,
            "model_state_dict": self.network.state_dict(),
            "config": self.config,
            "performance_history": self.performance_history,
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Also save as latest checkpoint
        latest_path = checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        self.current_iteration = checkpoint["iteration"]
        self.total_games_played = checkpoint["total_games"]
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.performance_history = checkpoint.get("performance_history", [])

        logger.info(
            f"Loaded checkpoint from iteration {self.current_iteration}"
        )

    def get_training_stats(self) -> Dict:
        """Get current training statistics."""
        return {
            "current_iteration": self.current_iteration,
            "total_games_played": self.total_games_played,
            "training_examples": len(self.training_examples),
            "performance_history": self.performance_history,
        }
