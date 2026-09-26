# ADR-0004: Ablate the previous action by zeroing its slot, not removing it

Status: accepted 2026-09-24

## Context

The previous action has been in the observation since v2, added to fix v1's held-A stall (ADR-0001). It went in
as part of a bundle, so there has never been a controlled comparison showing it matters. Review feedback asked
for exactly that: on/off, fixed budget, three seeds, reporting stall rate, completion rate and distance.

An ablation is only worth running if the two arms differ in the one variable under test.

## Decision

`obs_prev_action=False` feeds `-1` to `read_extras`, so the previous-action one-hot stays all zeros. The slot
is kept.

## Alternatives rejected

- **Remove the dimension.** The obvious reading of "ablate". Rejected because it changes `n_extras`, which
  changes every tensor shape and the first linear layer's width. The arms would then differ in *network
  capacity* as well as in the information available, and any measured gap could be attributed to either. It
  would also change the learning hash, so no existing checkpoint would resume.
- **Randomize the one-hot instead of zeroing it.** Rejected: that injects noise into the input rather than
  removing information, which is a different experiment.
- **Train one model and mask at eval.** Rejected: the policy would have been trained to rely on a feature that
  is then withdrawn, which measures brittleness, not whether the feature helps learning.

## Consequences

Good: the arms are byte-identical outside the ablated slice (asserted in a test), `n_extras` and the learning
hash are unchanged at the default so every existing run still resumes, and the new field is excluded from the
hash at its default value so no prior checkpoint's hash moves.

Bad: a zeroed one-hot is a state the network never sees under the default, and "all zeros" is also what the
first step of an episode looks like before any action has been taken. The ablated arm therefore reads every
step as "no previous action" rather than as a distinct "hidden" token. This is the standard way to do it and
keeps capacity fixed, but it is not information-theoretically clean, and if the result is null that is one
reason it might be.

The prediction was pre-registered in the README before the runs finished, with an explicitly weak prior: an
earlier 18-variant sweep moved the primary metric exactly once, so "no measurable difference" is a live
outcome and will be reported as the result rather than as a failed experiment.
