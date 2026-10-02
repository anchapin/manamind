"""Reference host playing through the exported ONNX model (#46)."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import export_onnx  # noqa: E402
import onnx_host  # noqa: E402
import train_simple  # noqa: E402

from manamind.core.action import ActionSpace  # noqa: E402
from manamind.core.agent import RandomAgent  # noqa: E402
from manamind.rules.simple import (  # noqa: E402
    SimpleStateEncoder,
    create_simple_game_start,
)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    torch.manual_seed(0)
    net = train_simple.build_simple_network(
        action_space_size=len(export_onnx.action_names())
    ).eval()
    out = tmp_path_factory.mktemp("model")
    export_onnx.export(net, out)
    return net, out


def _states(count: int = 40):
    """Positions from a real random-vs-random game."""
    state = create_simple_game_start(seed=5)
    agents = {0: RandomAgent(0, seed=1), 1: RandomAgent(1, seed=2)}
    seen = []
    while not state.is_game_over() and len(seen) < count:
        seen.append(state)
        state = (
            agents[state.priority_player].select_action(state).execute(state)
        )
    return seen


def test_numpy_observation_matches_the_encoder(exported) -> None:
    _, model_dir = exported
    schema = onnx_host.load_schema(model_dir)
    encoder = SimpleStateEncoder()
    for state in _states():
        np.testing.assert_allclose(
            onnx_host.observation(state, schema),
            encoder.features(state).numpy(),
            rtol=0,
            atol=1e-6,
        )


def test_priors_match_the_pytorch_policy(exported) -> None:
    net, model_dir = exported
    agent = onnx_host.OnnxPolicyAgent(model_dir)
    space = ActionSpace()
    names = agent.schema["actions"]
    for state in _states(15):
        legal = space.get_legal_actions(state)
        logits, value = agent.evaluate(state)
        priors = onnx_host.legal_priors(logits, legal, names)
        with torch.no_grad():
            ref_logits, ref_value = net(state)
        ref = onnx_host.legal_priors(ref_logits[0].numpy(), legal, names)
        np.testing.assert_allclose(priors, ref, atol=1e-5)
        assert priors.sum() == pytest.approx(1.0)
        assert value == pytest.approx(float(ref_value), abs=1e-5)


def test_shared_action_ids_split_their_mass() -> None:
    space = ActionSpace()
    legal = space.get_legal_actions(create_simple_game_start(seed=0))
    names = export_onnx.action_names()
    priors = onnx_host.legal_priors(np.zeros(len(names)), legal, names)
    by_type = {}
    for action, p in zip(legal, priors):
        by_type.setdefault(action.action_type.value, []).append(p)
    for group in by_type.values():
        assert max(group) == pytest.approx(min(group))


def test_full_game_through_onnx_only(exported) -> None:
    _, model_dir = exported
    agent = onnx_host.OnnxPolicyAgent(model_dir, player_id=0, seed=0)
    agents = {0: agent, 1: RandomAgent(1, seed=0)}
    winner, turns, _ = train_simple.play_game(agents, seed=11)
    assert turns > 0
    assert agent.decisions > 0
    assert winner in (0, 1, None)


def _key(action):
    # Actions carry a creation timestamp, so compare what they do.
    card = getattr(action, "card", None)
    return (action.action_type.value, getattr(card, "name", None))


def test_blunder_rate_one_plays_randomly(exported) -> None:
    _, model_dir = exported
    state = create_simple_game_start(seed=0)
    legal = {_key(a) for a in ActionSpace().get_legal_actions(state)}
    greedy = onnx_host.OnnxPolicyAgent(model_dir)
    top = _key(greedy.select_action(state))
    agent = onnx_host.OnnxPolicyAgent(model_dir, blunder_rate=1.0, seed=3)
    picks = {_key(agent.select_action(state)) for _ in range(30)}
    assert picks <= legal
    assert len(picks) > 1 or legal == {top}


def test_unknown_schema_version_fails_loudly(exported, tmp_path) -> None:
    _, model_dir = exported
    bad = tmp_path / "bad"
    shutil.copytree(model_dir, bad)
    schema_path = bad / "model.schema.json"
    schema = json.loads(schema_path.read_text())
    schema["schema_version"] = "simple-v0"
    schema_path.write_text(json.dumps(schema))
    with pytest.raises(ValueError, match="schema_version"):
        onnx_host.load_schema(bad)
