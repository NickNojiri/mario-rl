# Architecture decision records

One file per decision that would otherwise only exist as a shape in the code. Each records what was chosen,
what was rejected and why, and what it cost — including the decisions that turned out badly, which are the
ones worth reading.

| # | Decision | Status |
|---|---|---|
| [0001](0001-tile-grid-from-ram-instead-of-pixels.md) | Observe the tile grid from RAM instead of pixels | accepted |
| [0002](0002-held-out-levels-as-the-primary-metric.md) | Held-out real levels are the primary metric, with two baselines | accepted |
| [0003](0003-measure-stage-lengths-from-the-emulator.md) | Stage lengths are measured from the emulator, never published data | accepted |
| [0004](0004-ablate-by-zeroing-not-removing.md) | Ablate the previous action by zeroing its slot, not removing it | accepted |
| [0005](0005-short-screening-runs-only-for-within-distribution-changes.md) | Screen short only for within-distribution changes | accepted |

Related, kept in the README rather than here because they are results rather than decisions:
[changelog](../../README.md#changelog), [known gaps](../../README.md#known-gaps),
[pre-registered predictions](../../README.md#pre-registered-predictions).
