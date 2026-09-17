# mario-rl

Double-DQN on Super Mario Bros 1-1, CPU-only. Runs in WSL Ubuntu, because nes-py has to be compiled and gcc is already there.

## Setup (once, inside WSL)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.10 ~/.venvs/mario-rl
source ~/.venvs/mario-rl/bin/activate
cd /mnt/c/Users/17143/mario-rl
uv pip install -r requirements.txt --index-strategy unsafe-best-match
python scripts/probe_env.py   # import test + API behaviour check
```

From PowerShell, prefix any command with the venv wrapper:

```bash
wsl -d Ubuntu --cd /mnt/c/Users/17143/mario-rl -- bash scripts/wsl_run.sh python -m pytest -q tests
```

## Use

```bash
python train.py --preset smoke                            # ~20 s, exercises learn/sync/checkpoint
python train.py --preset full --run-name dqn_1_1          # ~105 steps/s on CPU -> 3M steps is about 8 h
python train.py --resume runs/dqn_1_1/latest.pt --total-steps 5000000
python eval.py runs/dqn_1_1/latest.pt --episodes 30
python eval.py runs/dqn_1_1/latest.pt --stage 2           # held-out stage
```

## Results: run 1 (`dqn_1_1`, 3M steps, ~9.7 h on a Ryzen 7 7800X3D)

![Run 1 DQN agent clearing World 1-1](docs/media/run1_flag_1-1.gif)

*Run 1's agent reaching the flag on 1-1. This is eval episode seed 10005 at epsilon 0.01, replayed exactly
with `python -m scripts.record_gif runs/dqn_1_1/latest.pt --seed 10005`. It is the best case, not the typical
one: 1 of 30 eval episodes reached the flag.*

| Eval (30 episodes, epsilon 0.01) | Mean x_pos | Flag rate |
|---|---|---|
| 1-1 | 1905 / ~3161 | 3% |
| 1-2 (held out) | 386 | 0% |

During training (epsilon 0.1), episodes getting past the 1-1 pipes rose from 24% to 80%, and the flag count went
from 0 to 64 per 250k steps. Main failure: half of eval episodes stall with A held for 100% of the last 150 steps.
SMB needs A released before it can jump again, and frames alone don't show that the button is held.
Next run: add the previous action as a network input.

Full write-up: [docs/mario_rl_report.pdf](docs/mario_rl_report.pdf)

## v2: PPO on tile grids across real stages

Goal: learn Mario's *mechanics*, measured on real levels the agent never trains on.

- **Observation:** a 13x16 grid of the game's own tile ids read from RAM, plus enemies, Mario, the previous
  action (so "A is already held" is visible), speed, and air/ground state. It's the same vocabulary on every
  level, and a future procedural generator can emit it too.
- **Stages:** 22 real stages for training, 5 held out (`2-1, 3-3, 4-2, 5-4, 7-1`). Water levels and looping
  maze castles are excluded.
- **Reward:** game reward + a small coin bonus (about one step of running) + a flag bonus. Both are config values,
  so their effect can be tested with an ablation.
- **Algorithm:** PPO with 12 parallel emulators; truncated episodes bootstrap from the real final observation.

```bash
python train_ppo.py --preset ppo_smoke                       # ~10 s, every code path
python train_ppo.py --preset ppo_1h --run-name ppo_1h_a      # ~1.7M steps, ~53 min at ~540 steps/s
python eval_stages.py runs/ppo_1h_a/latest.pt                # train + held-out stages
python eval_stages.py --random --config runs/ppo_1h_a/config.json
python -m scripts.compare_evals runs/ppo_1h_a/eval_final_all.json runs/ppo_1h_a/eval_random_all.json
python -m scripts.render_tiles --stage 1-1                   # check the RAM tile decoding by eye
```

### v2 results: a 1-hour run, not a finished agent

`ppo_1h_a`: 1.72M steps in 53 minutes (Ryzen 7 7800X3D, CPU only, ~540 steps/s). One training seed.

![PPO agent on held-out castle 5-4](docs/media/ppo_1h_heldout_5-4_best.gif)

*Best case, not typical: the single best of 50 held-out eval episodes. Stage 5-4 is a castle the agent never
trained on. It jumps past a fire bar and over lava, reaches x_pos 1763, then dies. On held-out stages the agent
reached the flag in **0 of 50** episodes, and its mean x_pos on 5-4 was 598. Replay:
`python -m scripts.record_gif_ppo runs/ppo_1h_a/latest.pt --stage 5-4 --seed 50008 --out <file>.gif`*

**Eval protocol:** 10 episodes per stage, seeds 50000-50009, random start delay of 0-30 steps, actions
sampled from the policy. The random baseline uses the same protocol. Mean x_pos is how far Mario got; ± is
the standard error; z is the difference divided by its standard error.

| Stages | Trained agent | Random policy | Ratio | Evidence | Flags reached |
|---|---|---|---|---|---|
| Training (22 stages) | **774** | 443 | 1.75x | z = 10.6; clearly ahead on 17 of 22 | 1 / 220 episodes |
| **Held out (5 stages)** | **582** | 382 | **1.52x** | z = 3.9 as a group; clearly ahead on 1 of 5 | **0 / 50** |

| Held-out stage | Trained agent | Random policy | Difference |
|---|---|---|---|
| 2-1 overworld | 666 ± 124 | 554 ± 91 | +112 (within noise) |
| 3-3 athletic | 828 ± 93 | 324 ± 17 | **+504 (z = 5.3)** |
| 4-2 underground | 251 ± 14 | 239 ± 16 | +12 (no difference) |
| 5-4 castle | 598 ± 132 | 387 ± 47 | +211 (within noise) |
| 7-1 overworld | 567 ± 95 | 405 ± 61 | +162 (within noise) |

**What the numbers show:**
- **On levels it trained on, it clearly learned.** It goes 1.75x as far as random and is clearly ahead on 17 of 22 stages.
  During training, mean episode distance rose from 547 to 868 and was still rising when the hour ended. It reached the
  flag 29 times in training (1-1, 3-2, 4-1, 6-1).
- **On levels it never saw, there is an early but weak sign of transfer.** As a group, held-out stages beat random
  (z = 3.9), but only one stage (3-3) is clearly better on its own. Underground 4-2 is no better than random, and no
  held-out level was completed.
- **Checkpoints during the hour were noisy.** Held-out mean x_pos at the 15-, 30- and 45-minute snapshots was 483,
  391 and 466 (5 episodes per stage), against 382 for random. Treat any single mid-run number with caution.
- **Caveats:** one seed; 10 episodes per stage; x_pos is not normalized by level length; the random start delay was
  0-8 steps in training but 0-30 in eval.

Next: longer runs (the training curve had not flattened), 3+ seeds, and progress measured as a fraction of each level.
Raw eval data: [docs/results/ppo_1h_a](docs/results/ppo_1h_a).

## Decisions that aren't obvious from the code

- **Truncation.** Timer expiry kills Mario, so it counts as `terminated`. gym's TimeLimit is 9,999,999 steps and never fires. The only truncation is the no-progress cutoff (`no_progress_steps`), and that is where bootstrapping through `truncated` matters.
- **Rewards.** The env returns the raw skip-frame sum, clipped to ±15 per frame, which gives ±60 per step. The agent stores `reward * reward_scale` (1/15). Logs and eval report raw reward.
- **Buffer.** Each frame is stored once and stacks are rebuilt at sample time: about 706 MB for 100k transitions.
- **Resume.** The checkpoint restores weights, target network, Adam state, step, episode and every RNG. The buffer is restored only if it was saved (`--save-buffer`). Otherwise learning waits until `burnin` new transitions arrive. Changing a learning field on resume needs `--allow-config-change`.
- **Eval.** Each episode gets its own seed, which varies the NOOP-start count and the epsilon RNG. `unique_trajectories` flags a policy that just replays one memorized action sequence.
