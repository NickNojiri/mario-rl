# ADR-0005: Screen short only for within-distribution changes

Status: accepted 2026-09-24

## Context

A full run is 15M steps, about 6.6 hours on this machine. A 2M-step screen is 53 minutes. If a short run ranked
ideas the same way a long one does, experiment throughput would improve roughly 7x — so it was worth checking
whether it does, using checkpoints already on disk and no new training.

Three long runs have a snapshot at ~1.4M steps and a known final score:

| run      | @1.4M | final       |
|----------|-------|-------------|
| v3b      | 527   | 674 (7.8M)  |
| gen1     | 456   | 446 (7.8M)  |
| gen2     | 449   | 671 (15M)   |

gen1 and gen2 were 7 points apart at 1.4M — noise, SE is about 46 — and finished 225 apart. **A 2M screen
would have killed gen2**, which went on to tie the best run. Even at 7.8M the two were only ~1.1 SE apart; the
separation did not appear until 15M.

The mechanism is not random. A curriculum pays its cost up front: gen2 spent 40% of episodes on generated
levels that are in neither the training nor the test distribution, so its early progress on real levels lagged
by construction and the transfer benefit arrived late. Screening short systematically penalizes any method
whose payoff is deferred.

## Decision

Screen short only for changes that apply identically at train and test time:

| change type | screen at 2M? | examples |
|---|---|---|
| observation encoding / architecture | yes | semantic tile classes, global pooling |
| optimizer / reward weights | yes | lr, entropy, clip, epochs |
| reweighting existing training levels | yes | prioritized level replay |
| training data from outside the distribution | **no — budget 15M** | procedural generation, demonstrations |

Spend the saved compute on evaluation episodes, which cost minutes rather than hours: 10 to 40 episodes per
stage halves the standard error from about 46 to about 23.

## Alternatives rejected

- **Screen everything short.** Rejected on the evidence above; it would have discarded gen2.
- **Never screen short.** Rejected: it is safe but costs 6.6 hours to rule out a bad learning rate, and the
  within-distribution cases show no such late divergence.
- **Rule by "data change vs optimizer change".** Too coarse, and PLR is the counterexample: it is a data change
  that *did* show up at 480k. The distinction that survives is whether the training data comes from outside the
  evaluation distribution, not whether data is touched at all.

## Consequences

Good: cheap experiments stay cheap and the expensive ones are identified in advance rather than after a
misleading early read.

Bad: n = 3. The direction is clear and the mechanism is plausible, but this is a rule inferred from three runs,
not an established result. It should be revisited once the multi-seed data (README P1) exists, since
seed-to-seed spread may account for part of the early-checkpoint noise this rule attributes to curriculum cost.
