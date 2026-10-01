"""Regression tests for the self-play training step.

The training loop used to log success without touching the network. These
tests fail if `_train_network` ever stops performing a real update again.
"""

import numpy as np
import pytest
import torch

from manamind.core.game_state import create_standard_game_start
from manamind.models.policy_value_network import PolicyValueNetwork
from manamind.training.self_play import SelfPlayTrainer


def _tiny_network() -> PolicyValueNetwork:
    """A deliberately small network so tests stay fast on CPU."""
    return PolicyValueNetwork(
        state_dim=64,
        hidden_dim=32,
        num_residual_blocks=1,
        action_space_size=16,
        use_attention=False,
    )


def _trainer(network: PolicyValueNetwork, **overrides) -> SelfPlayTrainer:
    trainer = SelfPlayTrainer(network)
    trainer.config.update(
        {"batch_size": 2, "epochs_per_iteration": 2, "learning_rate": 1e-2}
    )
    trainer.config.update(overrides)
    return trainer


def _examples(count: int = 4):
    state = create_standard_game_start()
    policy = np.ones(16) / 16
    return [
        (state, policy, 1.0 if index % 2 == 0 else -1.0)
        for index in range(count)
    ]


def test_train_network_updates_parameters() -> None:
    """Gradient descent must actually move the weights."""
    network = _tiny_network()
    trainer = _trainer(network)
    trainer.training_examples = _examples()

    before = [param.detach().clone() for param in network.parameters()]
    trainer._train_network()

    changed = [
        not torch.equal(old, new.detach())
        for old, new in zip(before, network.parameters())
    ]
    assert any(changed), "no parameter changed during training"


def test_train_network_records_loss_history() -> None:
    """Loss history is how training progress becomes visible."""
    network = _tiny_network()
    trainer = _trainer(network)
    trainer.training_examples = _examples()

    trainer._train_network()

    assert len(trainer.training_losses) == 1
    record = trainer.training_losses[0]
    assert record["epochs"], "per-epoch metrics missing"
    assert np.isfinite(record["final_epoch_loss"])


def test_train_network_skips_when_buffer_too_small() -> None:
    """A partial batch is skipped rather than silently mis-trained."""
    network = _tiny_network()
    trainer = _trainer(network, batch_size=8)
    trainer.training_examples = _examples(2)

    before = [param.detach().clone() for param in network.parameters()]
    trainer._train_network()

    assert trainer.training_losses == []
    for old, new in zip(before, network.parameters()):
        assert torch.equal(old, new.detach())


def test_optimizer_is_reused_across_iterations() -> None:
    """Adam's moment estimates are useless if the optimizer is rebuilt."""
    network = _tiny_network()
    trainer = _trainer(network)
    trainer.training_examples = _examples()

    trainer._train_network()
    first = trainer._optimizer
    trainer._train_network()

    assert trainer._optimizer is first


def test_encode_batch_renormalises_truncated_targets() -> None:
    """A policy target wider than the action space is clipped and rescaled."""
    network = _tiny_network()
    trainer = _trainer(network)

    state = create_standard_game_start()
    oversized = np.ones(1000) / 1000
    states, policies, values = trainer._encode_batch(
        [(state, oversized, 0.5), (state, np.zeros(16), -0.5)]
    )

    assert states.shape == (2, 64)
    assert policies.shape == (2, 16)
    assert policies[0].sum() == pytest.approx(1.0, abs=1e-5)
    # An all-zero target falls back to uniform rather than a dead gradient.
    assert policies[1].sum() == pytest.approx(1.0, abs=1e-5)
    assert values.tolist() == [0.5, -0.5]
