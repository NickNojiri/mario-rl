# ADR-0006: Tune by successive halving, then confirm the winner on unseen seeds

Status: accepted 2026-09-26

## Context

Finding better optimizer settings by full runs is expensive: 27 configs at 2.25M steps each is about 61M steps,
roughly 27 hours on this machine. Most configs are visibly worse long before the end, so most of that compute is
spent confirming what was already clear.

Two facts measured in this repo constrain how to go faster:

- **Short screens are valid only for some changes (ADR-0005).** They rank optimizer and objective settings
  faithfully, but kill methods whose payoff is deferred (two procedural-generation runs were 7 apart at 1.4M
  steps and 225 apart at the end).
- **Noise is large.** Seed-level SD is about 57 and eval noise about 46 (README, P1). The best of 27 noisy
  single-seed scores is partly the luckiest, not only the best.

## Decision

Successive halving: 27 configs at 250k steps, keep the top third, triple the budget, repeat — 27 → 9 → 3 → 1.
Promoted configs resume from their checkpoints. Then a separate confirmation: retrain the winner on seeds 0–2,
which the search never used, and compare against the default on the same seeds and budget with t ≥ 2 required.

Trial 0 is the unmodified default, so the search itself shows whether tuning beat doing nothing.

## Alternatives rejected

- **Grid or random search at full budget.** ~27 hours for the same 27 configs. Failing fast is the whole point.
- **Bayesian optimization (Optuna, etc.).** A new dependency, and with noise comparable to the effects being
  sought, a surrogate model mostly fits noise. Random sampling plus aggressive early stopping is the more honest
  tool at this signal-to-noise ratio.
- **Population-based training.** Changes hyperparameters mid-run, so the "winner" is a schedule rather than a
  setting, and it needs a population running in parallel, which 8 cores cannot give it.
- **Reporting the search winner's score as the result.** Rejected: it carries the winner's curse. The search picks
  a candidate; only the confirmation measures it.
- **Searching reward weights, observation or procgen too.** Out of scope: they change the problem rather than the
  optimizer, and procgen specifically is the case ADR-0005 says a short screen misjudges.

## Consequences

Good: ~7 hours instead of ~27, a built-in default comparison, a resumable state file, and a confirmation step
that separates skill from luck.

Bad, and accepted as the price of failing fast: at 250k steps the gaps between configs are small next to the
noise, so first-rung eliminations are partly random and a genuinely good config can die early. More eval
episodes at later rungs (10 / 15 / 20) reduce this where it matters most, but do not remove it.

Resuming from a checkpoint equals one longer run only while nothing in the config schedules on `total_steps`.
There is no LR annealing or entropy decay today, and `tests/test_sha_search.py` fails if a schedule field is ever
added, because at that point the search has to restart promoted configs instead.

A restart of the search itself must not re-promote a finished rung; that bug existed during development, was
caught by a smoke run, and is pinned by a test that fails without the fix.
