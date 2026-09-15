"""Per-episode view of eval behaviour: where episodes end and what the agent was pressing.

    python -m scripts.diagnose_eval runs/dqn_1_1/latest.pt --episodes 12
"""
import argparse
import collections
from dataclasses import replace

import numpy as np
import torch

from agent.mario import Mario
from config import Config
from env.make_env import ACTION_SETS, make_from_config

p = argparse.ArgumentParser()
p.add_argument("checkpoint")
p.add_argument("--episodes", type=int, default=12)
p.add_argument("--epsilon", type=float, default=0.01)
p.add_argument("--seed", type=int, default=10_000)
p.add_argument("--stage", type=int)
args = p.parse_args()

ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
cfg = replace(Config.from_dict(ck["config"]), buffer_size=16, device="cpu")
if args.stage:
    cfg = replace(cfg, stage=args.stage)
mario = Mario(cfg, len(ACTION_SETS[cfg.actions]))
mario.net.load_state_dict(ck["model"])
names = [" ".join(b) for b in ACTION_SETS[cfg.actions]]
env = make_from_config(cfg)

for i in range(args.episodes):
    rng = np.random.default_rng(args.seed + i)
    state, _ = env.reset(seed=args.seed + i)
    actions = []
    while True:
        a = mario.act(state, epsilon=args.epsilon, rng=rng)
        actions.append(a)
        state, _, terminated, truncated, info = env.step(a)
        if terminated or truncated:
            break
    tail = actions[-150:]
    a_share = sum("A" in names[a] for a in tail) / len(tail)
    top = collections.Counter(names[a] for a in tail).most_common(2)
    end = "TRUNC" if truncated else ("FLAG " if info["flag_get"] else "death")
    print(f"ep{i:2d} {end} x={info['x_pos']:4d} y={info['y_pos']:3d} len={len(actions):4d} "
          f"A pressed in last 150: {a_share:4.0%}  top: {top}")
env.close()
