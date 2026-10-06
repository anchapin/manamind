# Embedding manamind in a host engine (#46)

This is the contract between a trained manamind network and a host engine
such as planar-nexus (anchapin/planar-nexus#2378). It covers schema
`simple-v1`, the simple ruleset only. Real-card play needs a new schema
(#67), and hosts must refuse any schema version they don't know.

## What ships

`scripts/export_onnx.py CHECKPOINT OUT_DIR` writes two files. You can also
run it on the home PC with the "Export ONNX on home PC" workflow, which
uploads the artifact `onnx-<run>-<ckpt>`.

| File | Contents |
|---|---|
| `model.onnx` | Opset 17. Input `observation` float32 `[batch, 25]`. Outputs `policy_logits` float32 `[batch, 23]` and `value` float32 `[batch, 1]`. |
| `model.schema.json` | `schema_version`, `observation_dim`, every observation field in order with its scaling, the perspective, and the action names in id order. |

The export checks parity against PyTorch on 64 real positions (max abs
difference 1e-4). `scripts/onnx_host.py` is the reference host: it builds
observations from the schema with numpy and runs inference in onnxruntime,
without loading PyTorch or the training code.

## Call shape

```
evaluate(observation, legal_actions) -> (priors over legal_actions, value)
```

1. **Perspective.** Build the observation for the player who holds
   priority, not the active player. `value` is that player's expected
   result, from -1 (loss) to 1 (win).
2. **Observation.** 25 floats, in the order listed in `model.schema.json`:

   | Index | Field | Value |
   |---|---|---|
   | 0–9 | `self.*` | life / 20, hand / 7, library / 40, graveyard / 10, lands / 10, untapped lands / 10, creatures / 10, total power / 20, total toughness / 20, untapped creatures / 10 |
   | 10–19 | `opponent.*` | same ten fields for the other player |
   | 20–22 | `phase.main`, `phase.combat`, `phase.end` | one-hot of the current phase |
   | 23 | `self_is_active` | 1 if it is the priority player's turn |
   | 24 | `turn` | min(turn number, 60) / 60 |

3. **Actions.** Logits cover 23 action types, named in the schema
   (`play_land`, `cast_spell`, `pass_priority`, `declare_attackers`, ...).
   Map each legal move to its action type, softmax over the types that
   appear among the legal moves, and split a type's probability evenly
   when several legal moves share it. `legal_priors` in
   `scripts/onnx_host.py` is the reference version, and it matches
   `MCTSAgent`.
4. **Versioning.** Check `schema_version` before loading. An unknown
   version is a hard error, never a best-effort read.

## Difficulty tiers

All four tiers are the same network (`seed0_settle` iter 45) turned down.
They are not separately tuned profiles. The numbers below come from 160
games per cell. Full strength is Gumbel search with 40 simulations, greedy.
Runs 37502476482 and 37502491955; decision recorded on #46.

| Tier | Setting | vs random | vs full strength | ms/decision (Python, 2 threads) |
|---|---|---|---|---|
| Easy | policy only, temperature 1.0 | .775 | .044 | 0.27 |
| Medium | Gumbel 10 sims, 25% blunder rate | .944 | .181 | 39.8 |
| Hard | Gumbel 10 sims | .944 | .406 | 53.3 |
| Expert | Gumbel 40 sims | .994 | .500 | 233.5 |

The blunder rate is the chance of playing a uniformly random legal move
instead of the chosen one.

## Where search runs

- **Easy ships now.** It's one ONNX call per decision with no search, so
  onnxruntime-web or `ort` can run it directly.
- **Medium, Hard and Expert need search outside Python.** The plan is to
  port Gumbel search (`MCTSAgent`, `search="gumbel"`) to TypeScript and run
  it in the engine worker (anchapin/planar-nexus#2417), so a search never
  blocks a frame. The port has to reproduce the determinized simulation
  and the `_settle` step (forced moves are played out before a leaf is
  evaluated), or its strength won't match this table.
- The latency figures above are Python numbers, where most of the cost is
  rules simulation rather than the network. Re-measure in the host before
  tuning simulation counts.

## Budget

The simple-mode network is small. The ONNX file is well under 1 MB, and
inference fits easily in planar-nexus's 4 GB minimum spec. The real-card
state encoder (#28) will be the first real memory and latency cost.
