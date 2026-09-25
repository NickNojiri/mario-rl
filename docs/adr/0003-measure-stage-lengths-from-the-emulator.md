# ADR-0003: Stage lengths are measured from the emulator, never from published level data

Status: accepted 2026-09-23

## Context

Raw x_pos is not comparable across stages of different lengths, so averaging it over 5 held-out stages silently
weights the long ones. This sat in the README caveats from v2 onward. Fixing it needs each stage's flagpole
x_pos.

SMB level lengths are widely published. Copying them would be one lookup and would look identical in the
results table — which is exactly the problem: a wrong number from the internet is indistinguishable from a
right one until it corrupts every derived figure, and nothing in the repo would catch it.

## Decision

Measure the flagpole from the running game. When an episode ends with `flag_get`, the x_pos recorded at that
moment is read from the game's own RAM (0x6D page + 0x86 offset) and *is* the flagpole position.
`scripts/measure_stage_lengths.py` harvests these from recorded eval episodes, and `--verify` replays one from
its checkpoint and seed to confirm the value end to end.

Unmeasured stages are written as `null` with `"method": "not measured"`. They are placeholders for real runs.

## Alternatives rejected

- **Published level lengths.** Rejected as above. This is the whole point of the ADR.
- **Parsing level layout out of the ROM.** Would cover all 32 stages, but it is a second implementation of the
  game's own geometry, with its own bugs, to compute something the game will tell us directly.
- **Averaging disagreeing measurements.** Rejected: two different values cannot both be the flagpole. The
  script refuses the stage instead, since an average would silently hide the contradiction.

## Consequences

Good: 7 of 32 stages measured, all verified by replay. 140 recorded flag episodes agreed exactly per stage —
strong evidence the method is right. `progress_fraction` and a group mean are now reported, carrying
`stages_with_known_length` so partial coverage is never mistaken for full coverage.

Bad, and the honest cost of this choice: we can only measure stages the agent has actually finished. It has
never finished a held-out level, so `mean_progress_fraction` is `None` for the entire held-out group — the
normalization cannot be applied to the primary metric until the agent completes one. Copying the numbers from
a wiki would have "fixed" this today and produced figures nobody could defend.
