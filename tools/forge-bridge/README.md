# Forge bridge (#74)

Drives headless [Forge](https://github.com/Card-Forge/forge) games with a
manamind-controlled seat. Follow-up to the #66 engine spike.

Forge itself is not vendored. Download the release (2.0.15 tested) and a
JDK 17+, then:

```bash
FORGE_DIR=/path/to/forge tools/forge-bridge/build.sh
cd "$FORGE_DIR"
java -Xmx1500m -Duser.home="$FORGE_DIR/home" \
  -cp /path/to/manamind/tools/forge-bridge/out:forge-gui-desktop-2.0.15-jar-with-dependencies.jar \
  manamind.forge.RandomBench 100 rg.dck ub.dck
```

`user.home` needs a `.forge/` directory Forge can write to.

## RandomBench

One seat is a `PlayerControllerAi` subclass that picks uniformly at random
(including "pass") at each priority decision, random attackers and random
blocks, and the other seat is stock Forge AI. Everything else (targets,
mana payment, mulligans, triggers) still falls back to Forge AI.

The candidate set for spells is every ability Forge reports as playable that
the Forge AI's `canPlaySa` accepts (`WillPlay`). That check also picks
targets, so this is a narrower action space than a learned policy will
eventually need; see #74.

### Results (M19 Welcome Decks RG vs UB, 100 games, seats and decks alternated)

| | |
|---|---|
| crashes / exceptions | 0 |
| random seat wins | 0 / 100 (as expected vs Forge AI) |
| mean turns | 15.8 (max 34) |
| random-seat decisions | 15,634 |
| wall time (excl. ~45 s JVM start) | 197.2 s → 79 decisions/s overall |
| time inside our controller | 27.0 s → ~580 decisions/s |

Most of the wall time is the Forge AI opponent thinking. In self-play both
seats are ours, so the controller-side number is the better estimate of the
ceiling. 450 "AI failed to play" log lines: `canPlaySa` accepted a spell
whose mana payment then failed (e.g. Gravewaker); the engine recovered each
time.

## PipeBench + Python client

`PipeBench` is the same controller, but every priority, attack and block
decision is sent to an external process as one line of JSON (prefixed
`@@MM ` on stdout, since Forge logs there too) and the answer comes back on
stdin. Each message carries only what that seat can see: turn, phase, both
life totals, own hand, opponent hand size, both battlefields, library sizes,
and the legal options. Cards come with name, type line, mana cost, mana
value, power/toughness, tapped and summoning-sick flags; graveyards are
listed by name. `python/forge_client.py` launches one JVM for many
games and answers with a random policy:

```bash
python tools/forge-bridge/python/forge_client.py \
  --forge-dir "$FORGE_DIR" --java "$JAVA_HOME/bin/java" --games 40 rg.dck ub.dck
```

Protocol replies: `priority` takes an option index (`len(options)` = pass);
`attack` takes space-separated creature indices; `block` takes
space-separated `blocker:attacker` pairs. Illegal combat replies fall back
to Forge AI and are counted as fallbacks.

### Results (same decks, 40 games, Python random policy)

| | |
|---|---|
| crashes / fallbacks / parse errors | 0 / 0 / 0 |
| mean turns | 15.6 |
| Python-side decisions | 6,187 |
| wall time (after JVM ready) | 93.5 s → 66 decisions/s, incl. Forge AI opponent |
| time spent in the Python policy | 0.10 s |

The pipe adds little over the in-JVM random seat (79/s with the opponent's
thinking included). Turning the options into manamind's encoder and action
space is the next step.

## ForgeEnv (Python package)

`manamind.forge_interface.ForgeEnv` wraps the pipe as a decision-driven
environment: `reset()` returns the first decision of the next game, and
`step(reply)` answers it and returns the next one, or `done` with reward
+1/-1/0 when the game ends. Build the command with `bridge_command(...)`
and the replies with `priority_reply`, `attack_reply`, `block_reply`.
Unit tests (`tests/unit/test_forge_env.py`) use a fake bridge, so CI
doesn't need Forge or Java. Smoke run against real Forge: 6 games,
964 decisions in 23 s, 0 fallbacks or errors.
