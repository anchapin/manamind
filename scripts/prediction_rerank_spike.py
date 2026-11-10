"""Manamind #106 prediction-rerank spike: stage-1 analysis.

Reads the planar-nexus self-play probe records (selfplay-forge.ts
--probe collect; one JSON game per line, gzip or plain) and scores the
value-head re-rank signal:

For every net decision carrying a probe payload (the search's
per-candidate value-head reads: visit counts, mean leaf value over the
dealt worlds, and the search's own pick index):

- top-k candidates by visit count (k, ties by candidate order);
- valuePick = the highest mean-q candidate among the visited ones inside
  that top-k (ties: more visits, then candidate order);
- override = valuePick differs from the search's pick;
- regret = q[valuePick] - q[pick] (how much value the search pick left
  on the table, by the value head's own read of the successors).

Aggregates per seat and overall, plus the mover's realized game return
on override vs agree decisions, into `rerank_analysis.json`. Eval-only:
no training, no champion changes. Go/no-go and any h2h decision live on
manamind issue #96.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path


def load_games(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    games = []
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                games.append(json.loads(line))
    return games


def value_pick(visits: list[int], q: list[float | None], k: int) -> int | None:
    """Value-head favorite among the top-k visited candidates, or None."""
    order = sorted(
        range(len(visits)), key=lambda i: (-visits[i], i)
    )[:k]
    best: int | None = None
    best_q = -math.inf
    best_v = -1
    for i in order:
        if visits[i] <= 0 or q[i] is None:
            continue
        if q[i] > best_q or (q[i] == best_q and visits[i] > best_v):
            best = i
            best_q = q[i]
            best_v = visits[i]
    return best


def analyze_games(games: list[dict], k: int) -> dict:
    seats: dict[int, dict] = {}
    rows = []
    for game in games:
        returns = game.get("returns") or []
        for idx, d in enumerate(game.get("decisions") or []):
            probe = d.get("probe")
            if not probe:
                continue
            visits = probe["visits"]
            q = probe["q"]
            pick = probe["pick"]
            vp = value_pick(visits, q, k)
            if vp is None:
                continue
            seat = d.get("seat", 0)
            row = {
                "seed": game.get("seed"),
                "idx": idx,
                "seat": seat,
                "pick": pick,
                "value_pick": vp,
                "override": vp != pick,
                "q_pick": q[pick],
                "q_value_pick": q[vp],
                "visits_pick": visits[pick],
                "visits_value_pick": visits[vp],
                "return": returns[idx] if idx < len(returns) else None,
            }
            if row["override"] and row["q_pick"] is not None:
                row["regret"] = row["q_value_pick"] - row["q_pick"]
            rows.append(row)
            bucket = seats.setdefault(seat, {"n": 0, "overrides": 0})
            bucket["n"] += 1
            bucket["overrides"] += 1 if row["override"] else 0

    def mean(vals: list[float]) -> float | None:
        return sum(vals) / len(vals) if vals else None

    overrides = [r for r in rows if r["override"]]
    regrets = [r["regret"] for r in overrides if "regret" in r]
    agree_ret = [
        r["return"] for r in rows if not r["override"] and r["return"] is not None
    ]
    override_ret = [
        r["return"] for r in overrides if r["return"] is not None
    ]
    out = {
        "k": k,
        "n_games": len(games),
        "n_decisions": len(rows),
        "override_rate": len(overrides) / len(rows) if rows else None,
        "mean_regret": mean(regrets),
        "regret_positive_share": (
            sum(1 for r in regrets if r > 0) / len(regrets) if regrets else None
        ),
        "mean_q_pick": mean([r["q_pick"] for r in rows if r["q_pick"] is not None]),
        "mean_q_value_pick": mean(
            [r["q_value_pick"] for r in rows if r["q_value_pick"] is not None]
        ),
        "mean_visits_pick": mean([float(r["visits_pick"]) for r in rows]),
        "mean_visits_value_pick": mean(
            [float(r["visits_value_pick"]) for r in rows]
        ),
        "mover_return_agree": mean(agree_ret),
        "mover_return_override": mean(override_ret),
        "by_seat": {
            str(seat): {
                "n": b["n"],
                "overrides": b["overrides"],
                "override_rate": b["overrides"] / b["n"] if b["n"] else None,
                "mean_regret": mean(
                    [
                        r["regret"]
                        for r in rows
                        if r["seat"] == seat and "regret" in r
                    ]
                ),
            }
            for seat, b in sorted(seats.items())
        },
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    games = load_games(args.inp)
    out = analyze_games(games, args.top_k)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
