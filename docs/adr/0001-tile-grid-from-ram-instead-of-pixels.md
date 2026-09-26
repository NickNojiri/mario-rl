# ADR-0001: Observe the tile grid from RAM instead of pixels

Status: accepted (v2)

## Context

v1 was Double DQN on stacked grayscale frames of 1-1. Its dominant failure, measured over 30 eval episodes:
half of them stalled with the A button held for 100% of the last 150 steps. SMB requires A to be *released*
before it can jump again, and a stack of frames does not show that a button is currently held. The agent could
not see the one piece of state that was killing it.

A second problem: pixels are level-specific. World 1 and world 5 render the same wall with different colours,
so a pixel policy has to relearn "wall" per theme, which works against the project's actual question — whether
it learns mechanics or memorizes levels.

## Decision

Observe a 13x16 grid of the game's own metatile ids, read directly from RAM at 0x500-0x69F, plus explicit
extras: the previous action as a one-hot, Mario's horizontal and vertical speed, float/air state and powerup.

## Alternatives rejected

- **Pixels plus a frame stack.** Rejected: the held-button failure above, and ~4ms/step of emulation is already
  the bottleneck without adding a larger CNN on top.
- **Pixels plus an explicit previous-action input.** Would fix the held-A bug but not the per-theme relearning,
  and costs more compute for the same information.
- **A hand-written feature vector** (distance to next pit, nearest enemy, etc.). Rejected as too close to
  solving the game by hand; the grid keeps the spatial problem intact.

## Consequences

Good: the same vocabulary on every level, the previous action is visible by construction, and the observation
build costs ~5us against ~4.2ms for the emulator step, so it is free in practice.

Bad, and later measured: raw tile ids also encode *which* level this is (solid themes 0x54/0x52/0x62 differ by
world), which is a memorization channel. The train/held-out gap is roughly 2x and has never closed. Collapsing
ids to semantic classes is the open follow-up.

Also bad: enemy *motion* is invisible in a stack of tile frames — a moving goomba is just a different cell. This
was not anticipated and had to be patched in obs_version 3 with explicit per-enemy velocity features.
