"""How does the PPO agent use its buttons, and what kills it?

    python -m scripts.diagnose_actions runs/ppo_1h_a/latest.pt --episodes 4
"""
import argparse
from collections import Counter

import numpy as np
import torch

from agent.ppo import PPOAgent
from config import PPOConfig
from env.tiles import TileMarioEnv

p = argparse.ArgumentParser()
p.add_argument("checkpoint")
p.add_argument("--episodes", type=int, default=4, help="per stage")
p.add_argument("--stages", default="all")
p.add_argument("--seed", type=int, default=90_000)
args = p.parse_args()

torch.set_num_threads(4)
ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
cfg = PPOConfig.from_dict({**ckpt["config"], "noop_max": 30})
train, test = cfg.resolved_stages()
stages = train + test if args.stages == "all" else args.stages.split(",")
env = TileMarioEnv(**cfg.env_kwargs(stages))
env.prebuild()
agent = PPOAgent(cfg, env.n_actions, env.n_extras)
agent.net.load_state_dict(ckpt["model"])
agent.net.eval()
names = [" ".join(b) for b in env.action_set]
LEFT = names.index("left")
NOOP = names.index("NOOP")

actions, pre_death_left, pre_death_noop, all_left, all_noop = Counter(), [], [], [], []
endings = Counter()
left_used_episodes = 0
n = 0
for stage in stages:
    for i in range(args.episodes):
        seed = args.seed + n
        n += 1
        gen = torch.Generator().manual_seed(seed)
        obs, _ = env.reset(seed=seed, stage=stage)
        probs_hist, used_left = [], False
        while True:
            probs, _ = agent.probs(obs)
            a = int(torch.multinomial(probs, 1, generator=gen))
            actions[names[a]] += 1
            used_left |= a == LEFT
            probs_hist.append(probs.numpy())
            obs, _, terminated, truncated, info = env.step(a)
            if terminated or truncated:
                break
        left_used_episodes += used_left
        ph = np.array(probs_hist)
        all_left.append(ph[:, LEFT].mean())
        all_noop.append(ph[:, NOOP].mean())
        if info["episode"]["flag_get"]:
            endings["flag"] += 1
        elif truncated:
            endings["stalled (cutoff)"] += 1
        else:
            ram = env.ram
            endings["fell in pit" if ram[0xB5] > 1 else ("time up" if info["time"] == 0 else "hit enemy/hazard")] += 1
            pre_death_left.append(ph[-8:, LEFT].mean())
            pre_death_noop.append(ph[-8:, NOOP].mean())
env.close()

total = sum(actions.values())
print(f"episodes={n}  steps={total}")
print("action share:", {k: f"{v / total:.1%}" for k, v in actions.most_common()})
print(f"episodes where LEFT was pressed at least once: {left_used_episodes}/{n}")
print(f"mean P(LEFT) overall {np.mean(all_left):.1%}, in last 8 steps before a death {np.mean(pre_death_left):.1%}")
print(f"mean P(NOOP) overall {np.mean(all_noop):.1%}, in last 8 steps before a death {np.mean(pre_death_noop):.1%}")
print("how episodes ended:", dict(endings))
