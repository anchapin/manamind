"""BC-pass behaviour cloning: resume trigger and val memory (#96)."""

import json
from pathlib import Path

import pytest
import torch

import manamind.training.train_forge as tf
from manamind.models.forge_pointer import ImitateOutput


def _read_log(out_dir: Path) -> list:
    path = out_dir / "log.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _write_checkpoint(path: Path, games: int, bc_done: bool) -> None:
    net = tf.ForgePointerNet()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    torch.save(
        {
            "network": net.state_dict(),
            "optimizer": opt.state_dict(),
            "config": {"card_dim": net.card_dim, "state_dim": net.state_dim},
            "meta": {"games": games, "bc_done": bc_done},
        },
        path,
    )


@pytest.fixture()
def fake_env(monkeypatch):
    class FakeEnv:
        def __init__(self, command, cwd=None):
            self.command = list(command)
            self.summary = None

        def reset(self):
            from manamind.forge_interface.forge_env import StepResult

            return StepResult(None, True, 1.0, {"result": "win", "turns": 4})

        def step(self, reply):
            from manamind.forge_interface.forge_env import StepResult

            return StepResult(None, True, 1.0, {"result": "win", "turns": 4})

        def close(self):
            pass

    monkeypatch.setattr(tf, "ForgeEnv", FakeEnv)


def test_resume_at_expert_games_runs_bc(fake_env, tmp_path):
    """A resume whose checkpoint sits exactly at expert_games still runs BC.

    Regression: the BC block used to live inside the
    ``expert_games > start_game`` guard, so this case skipped BC entirely
    and burned the whole --games budget on RL (run 44939730233).
    """
    ckpt = tmp_path / "seed.pt"
    _write_checkpoint(ckpt, games=2, bc_done=False)
    out = tmp_path / "run"
    meta = tf.train(
        command=["rl"],
        cwd=None,
        games=0,
        out_dir=out,
        expert_command=["bridgeX"],
        expert_games=2,
        bc_epoch_count=1,
        resume=ckpt,
        save_every=0,
    )
    bc = [r for r in _read_log(out) if "bc_epoch" in r]
    assert len(bc) == 1
    assert meta["bc_done"] is True
    assert (out / "bc.pt").exists()


def test_resume_after_bc_skips_bc(fake_env, tmp_path):
    ckpt = tmp_path / "seed.pt"
    _write_checkpoint(ckpt, games=2, bc_done=True)
    out = tmp_path / "run"
    meta = tf.train(
        command=["rl"],
        cwd=None,
        games=0,
        out_dir=out,
        expert_command=["bridgeX"],
        expert_games=2,
        bc_epoch_count=1,
        resume=ckpt,
        save_every=0,
    )
    assert [r for r in _read_log(out) if "bc_epoch" in r] == []
    assert meta["bc_done"] is True


class FakeImitateNet:
    """Deterministic stand-in for ForgePointerNet.imitate."""

    def __init__(self) -> None:
        self.p = torch.nn.Parameter(torch.tensor(0.5))
        self.q = torch.nn.Parameter(torch.tensor(-0.2))

    def parameters(self):
        return [self.p, self.q]

    def train(self, mode: bool = True) -> None:
        pass

    def imitate(self, d):
        x = torch.tensor(float(d["x"]))
        return ImitateOutput(
            log_prob=self.p * x,
            value=self.q * x + 1.0,
            labelled=d["labelled"],
            choice=d["choice"],
            agree=d["agree"],
        )


def _fake_games(n_games: int, rows_per_game: int):
    games = []
    for g in range(n_games):
        decisions = []
        returns = []
        for k in range(rows_per_game):
            decisions.append(
                {
                    "x": 0.1 * ((g * 7 + k) % 13),
                    "labelled": k % 2 == 0,
                    "choice": k % 3 != 0,
                    "agree": k % 4 == 0,
                }
            )
            returns.append(0.25 * ((g + k) % 5) - 0.5)
        games.append({"decisions": decisions, "returns": returns})
    return games


def _run_bc_epochs(games, val_chunk):
    torch.manual_seed(0)
    net = FakeImitateNet()
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    recs = tf.bc_epochs(
        net, opt, games, epochs=1, batch=8, seed=0, val_chunk=val_chunk
    )
    return recs


def test_bc_epochs_chunked_val_matches_single_pass():
    games = _fake_games(30, 5)
    big = _run_bc_epochs(games, val_chunk=10_000)
    small = _run_bc_epochs(games, val_chunk=2)
    a, b = big[0], small[0]
    assert a["val_rows"] == b["val_rows"] == 15
    for key in ("val_nll", "val_acc", "val_value"):
        assert abs(a[key] - b[key]) < 1e-6, key


def _write_corpus(path, games):
    import gzip

    with gzip.open(path, "wt") as f:
        for g in games:
            f.write(
                json.dumps(
                    {
                        "decisions": g["decisions"],
                        "returns": g["returns"],
                        "result": "win",
                    }
                )
                + "\n"
            )


def test_expert_corpus_stores_raw_lines_only(tmp_path):
    games = _fake_games(3, 2)
    path = tmp_path / "expert_data.jsonl.gz"
    _write_corpus(path, games)
    corpus = tf.ExpertCorpus(path)
    assert len(corpus) == 3
    assert all(isinstance(x, bytes) for x in corpus.lines)
    assert corpus[0]["decisions"] == games[0]["decisions"]


def test_bc_epochs_corpus_stream_matches_list(tmp_path):
    games = _fake_games(30, 5)
    path = tmp_path / "expert_data.jsonl.gz"
    _write_corpus(path, games)
    corpus = tf.ExpertCorpus(path)
    torch.manual_seed(0)
    net_a = FakeImitateNet()
    opt_a = torch.optim.Adam(net_a.parameters(), lr=0.01)
    from_list = tf.bc_epochs(net_a, opt_a, games, epochs=2, batch=8, seed=0)
    torch.manual_seed(0)
    net_b = FakeImitateNet()
    opt_b = torch.optim.Adam(net_b.parameters(), lr=0.01)
    from_corpus = tf.bc_epochs(net_b, opt_b, corpus, epochs=2, batch=8, seed=0)
    assert from_list == from_corpus


def test_bc_epochs_small_buffer_covers_all_rows():
    games = _fake_games(30, 5)
    torch.manual_seed(0)
    net = FakeImitateNet()
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    recs = tf.bc_epochs(
        net, opt, games, epochs=1, batch=8, seed=0, buffer_rows=32
    )
    assert recs[0]["train_rows"] == 135
