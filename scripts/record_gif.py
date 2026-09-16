"""Replay one DQN eval episode exactly (same seed protocol as eval.py) and save it as a GIF.

    python -m scripts.record_gif runs/dqn_1_1/latest.pt --seed 10005 --out docs/media/run1_flag_1-1.gif

eval.py episode i uses seed base_seed + i for both the NOOP-start count and the epsilon RNG, so
`--seed 10005` reproduces episode 5 of `eval.py --seed 10000`.
"""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from agent.mario import Mario
from config import Config
from env.make_env import ACTION_SETS, make_from_config


def find_smb(env):
    while not hasattr(env, "ram"):
        env = env.env
    return env


p = argparse.ArgumentParser()
p.add_argument("checkpoint", type=Path)
p.add_argument("--seed", type=int, default=10005)
p.add_argument("--epsilon", type=float, default=0.01)
p.add_argument("--out", type=Path, default=Path("docs/media/run1_flag_1-1.gif"))
p.add_argument("--every", type=int, default=1, help="keep every Nth agent step (each step = 4 game frames)")
p.add_argument("--colors", type=int, default=48)
args = p.parse_args()

ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
cfg = replace(Config.from_dict(ckpt["config"]), buffer_size=16, device="cpu")
mario = Mario(cfg, len(ACTION_SETS[cfg.actions]))
mario.net.load_state_dict(ckpt["model"])
env = make_from_config(cfg)
smb = find_smb(env._base)

rng = np.random.default_rng(args.seed)
state, _ = env.reset(seed=args.seed)
frames = [smb.screen.copy()]
steps = 0
while True:
    state, _, terminated, truncated, info = env.step(mario.act(state, epsilon=args.epsilon, rng=rng))
    steps += 1
    if steps % args.every == 0 or terminated or truncated:
        frames.append(smb.screen.copy())
    if terminated or truncated:
        break
env.close()
print(f"seed={args.seed} steps={steps} x_pos={info['x_pos']} flag_get={info['flag_get']} "
      f"terminated={terminated} truncated={truncated} frames={len(frames)}")

# One shared palette for every frame avoids color flicker between frames.
palette = Image.fromarray(np.concatenate(frames[:: max(1, len(frames) // 12)], axis=0)).quantize(
    colors=args.colors, method=Image.Quantize.MEDIANCUT)
images = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
images += [images[-1]] * 20  # hold the final frame
args.out.parent.mkdir(parents=True, exist_ok=True)
images[0].save(args.out, save_all=True, append_images=images[1:], duration=int(1000 / 15 * args.every),
               loop=0, optimize=True, disposal=1)
print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")
