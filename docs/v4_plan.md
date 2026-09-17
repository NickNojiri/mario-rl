# v4 plan: from measured failures to techniques

## What the data says (1-hour v2 agent, `scripts/pit_forensics.py`, 162 episodes, 1,220 takeoffs)

| Finding | Number |
|---|---|
| Pit deaths where Mario **walked off without jumping** | 32 / 64 (50%) |
| Pit deaths from **releasing A too early** | 9 (14%) |
| Took off too early / too slow / gap out of reach / other | 6 / 3 / 6 / 8 |
| Pit-adjacent jumps that **succeed** | 147 / 169 (87%) |
| A held, successful vs failed pit jumps (median steps) | 5 vs 2 |
| Steps the pit edge was on screen before the fatal step (median) | 12 |
| Policy's jump probability in the 4 steps before walking off (median) | 0.48 |

Measured jump physics (`scripts/jump_physics.py`, identical on 8-1 and 3-2): max distance is 3.3 tiles standing,
5.2 walking and 9.8 running. A tap covers about half the distance of a full hold, and a full hold needs 6-8 agent
steps (24-32 frames) of A.

**Diagnosis.** The agent does see the danger (12 steps of warning), but it is indecisive at the edge (48% jump)
and cannot sustain long jumps. Part of that is simple probability: the policy re-samples every 4 frames, so
holding A for 5 steps at 80% per step happens only 0.8^5 = 33% of the time.

## Techniques, and where each one applies

| # | Technique (source) | The math or probability it uses | What it fixes here |
|---|---|---|---|
| 1 | **Macro actions / options** ([Sutton, Precup & Singh 1999](http://incompleteideas.net/papers/SPS-aij.pdf); [FiGAR, Sharma et al. 2017](https://arxiv.org/abs/1702.06054)) | Semi-MDP: an action lasting d steps is discounted by gamma^d, and advantages chain with (gamma*lambda)^d | "Run-jump long" becomes one decision instead of 8 lucky ones in a row (removes the 0.8^k problem) |
| 2 | **Physics hint features** (fair observation engineering) | Distance to the pit edge, pit width, and whether a walk/run tap/full jump can clear it, from the measured jump table; time-to-contact dx / relative speed for the nearest enemy; a stomp-opportunity flag | Turns "can I make this jump?" from something learned slowly out of pixels into something the agent reads directly |
| 3 | **Go-Explore style practice** ([Ecoffet et al., Nature 2021](https://www.nature.com/articles/s41586-020-03157-9)) | Remember states just before deaths, return to them deterministically (the emulator replays exactly), explore from there | Many more attempts at the exact pits and enemies it fails on. Eval still starts every level from the beginning |
| 4 | **Prioritized Level Replay** ([Jiang et al., ICML 2021](https://proceedings.mlr.press/v139/jiang21b/jiang21b.pdf)) | Sample stages by rank of learning potential (mean positive advantage), mixed 50/50 with uniform | Less time on stages it has solved or can't learn yet, more on the ones where it is improving |
| 5 | **Goal/mode-conditioned policy** ([UVFA, Schaul et al. 2015](https://proceedings.mlr.press/v37/schaul15.html)) | One network V(s, mode), pi(a given s, mode); the reward weights depend on the mode | **Safe mode** (death -300, slow clock penalty, points valued) and **insane mode** (speedrun: 3x clock penalty, 1.5x progress, big flag bonus) from one training run |
| 6 | **Potential-based shaping** ([Ng, Harada & Russell 1999](https://www.cs.utexas.edu/~shivaram/readings/b2hd-NgHR1999.html)) | F = gamma*Phi(s') - Phi(s) leaves the optimal policy unchanged | Reference for keeping shaping honest. v3's new-ground progress is already close to potential-based; points and hurt bonuses are not, so they are capped |

Considered for later: data augmentation (RAD/DrAC) matters mainly for pixel observations; recurrent PPO for
piranha-plant timing; risk-sensitive CVaR objectives as a principled "safe mode".

## v4 build

- `actions="simple_macro"`: the 7 simple buttons + left+A and left+B, plus 5 macros: run-jump short (2 steps),
  medium (5), long (8), walk-jump long (6), hop back (left+A, 4). Macros are expanded in the trainer as forced
  steps, so no env waits on another. Their returns use the semi-MDP discount.
- `obs_version=4`: v3 + 9 physics hints + a mode one-hot.
- `reward_version=4`: v3 components with mode-dependent weights (safe / insane), mode sampled 50/50 per episode.
- `practice_prob=0.25`: return to a remembered point 6-20 steps before a recent death on that stage.
- `plr=True`: prioritized stage sampling.
- Eval reports safe and insane modes separately; practice and prioritization are off during eval.
- Same length as `ppo_1h_a`, compared on the same held-out protocol, plus pit forensics before and after.
