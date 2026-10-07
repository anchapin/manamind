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
