# ADR-0002: Held-out real levels are the primary metric, with two baselines

Status: accepted (v2), amended 2026-09-24 (scripted baseline)

## Context

The project's question is whether the agent learns mechanics or memorizes levels. Training-level score cannot
answer it: longer training moved training distance from 1434 to 1617 and training flag rate from 13.3% to 17.1%
while held-out distance moved 674 to 671. Optimizing the number we look at would have looked like progress.

A single primary metric also has to be fixed *before* experiments, or every run can be described as an
improvement on whichever metric happened to move.

## Decision

The primary number is **held-out mean x_pos**: 5 real stages never trained on (2-1, 3-3, 4-2, 5-4, 7-1), 10
episodes each, seeds 50000+, randomized start offsets, actions sampled from the policy. Flag rate and stall
rate are reported alongside it, always.

Two baselines under the identical protocol, because one weak floor proves nothing:

- uniform random: **382**
- scripted right-and-jump: **412**, with its jump cycle tuned on *training* stages only

## Alternatives rejected

- **Training-level score.** Rejected on the evidence above: it moves ~3x more than the held-out number.
- **Episode reward.** Rejected because the reward is ours and we keep changing it; a number that moves when we
  edit its own definition cannot compare runs across reward versions.
- **Flag rate as primary.** It is the thing we actually want, but it is 0.0% on held-out in every run at every
  checkpoint, so it has no resolution. Distance is the shaped proxy. This is a known substitution, not a
  claim that distance is the goal.
- **Tuning the scripted baseline on held-out stages.** Rejected: it would make the floor a function of the test
  set.

## Consequences

Good: the metric is fixed, cheap, has measured noise (SE about 46 at 10 episodes/stage) and two floors.

Bad: mean x_pos rewards dying far over surviving carefully, and is not normalized by level length — averaging
across stages of different lengths weights the long ones. ADR-0003 fixes the second; the first is still open
and is the reason "run right, never retreat" is optimal under our own specification.

Bad: the held-out set has been used to choose checkpoints and settings, so it is really a *validation* set and
671/674 are optimistically biased. Lost Levels (16 usable stages, RAM map verified) is the candidate true test
set.
