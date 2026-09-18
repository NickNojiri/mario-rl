"""Replay one eval_stages.py episode exactly and save it as a GIF.

    python -m scripts.record_gif_ppo runs/ppo_1h_a/latest.pt --stage 3-3 --seed 50003 --out docs/media/x.gif

Uses the same protocol as eval_stages.py: env.reset(seed), torch.Generator(seed) for action sampling,
1 torch thread, noop_max from --noop-max (eval default 30). Prints the replayed result so it can be
checked against the eval JSON.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from agent.macros import MacroStepper
from agent.ppo import PPOAgent
from config import PPOConfig
from env.tiles import TileMarioEnv

p = argparse.ArgumentParser()
p.add_argument("checkpoint", type=Path)
p.add_argument("--stage", required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--noop-max", type=int, default=30)
p.add_argument("--greedy", action="store_true")
p.add_argument("--out", type=Path, required=True)
p.add_argument("--colors", type=int, default=48)
p.add_argument("--mode", choices=["safe", "insane"], default="safe")
args = p.parse_args()

torch.set_num_threads(1)
ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
cfg = PPOConfig.from_dict({**ckpt["config"], "noop_max": args.noop_max, "practice_prob": 0.0, "procgen_prob": 0.0})
env = TileMarioEnv(**cfg.env_kwargs([args.stage]))
stepper = MacroStepper(env, cfg.actions)
agent = PPOAgent(cfg, env.n_policy_actions, env.n_extras)
agent.net.load_state_dict(ckpt["model"])
agent.net.eval()

gen = torch.Generator().manual_seed(args.seed)
obs, _ = env.reset(seed=args.seed, stage=args.stage, mode=args.mode, practice=False)
frames = [env.screen.copy()]
env.frame_callback = lambda: frames.append(env.screen.copy()) if len(frames) % 1 == 0 else None
while True:
    a = int(agent.act({k: v[None] for k, v in obs.items()}, greedy=args.greedy, generator=gen)[0][0])
    obs, _, terminated, truncated, info = stepper.step(a)
    if terminated or truncated:
        break
frames = frames[::4]  # one frame per agent step (15 fps), same pacing as before
ep = info["episode"]
env.close()
print(f"stage={args.stage} seed={args.seed} x_pos={ep['x_pos']} flag={ep['flag_get']} coins={ep['coins']} "
      f"length={ep['length']} terminated={terminated} truncated={truncated}")

palette = Image.fromarray(np.concatenate(frames[:: max(1, len(frames) // 12)], axis=0)).quantize(
    colors=args.colors, method=Image.Quantize.MEDIANCUT)
images = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
images += [images[-1]] * 20
args.out.parent.mkdir(parents=True, exist_ok=True)
images[0].save(args.out, save_all=True, append_images=images[1:], duration=67, loop=0, optimize=True, disposal=1)
print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")
