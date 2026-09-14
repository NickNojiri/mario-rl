"""Fixed-protocol evaluation, separate from training noise.

    python eval.py runs/dqn_1_1/latest.pt --episodes 30
    python eval.py runs/dqn_1_1/latest.pt --world 1 --stage 2     # held-out stage

Each episode i uses seed base_seed + i for both the NOOP-start count and the epsilon RNG,
so episodes differ from each other but the whole eval is reproducible. The emulator is
deterministic, so `unique_trajectories` tells you whether the policy is just replaying
one memorized action sequence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from agent.mario import Mario
from config import Config
from env.make_env import ACTION_SETS, make_from_config


def evaluate(mario: Mario, cfg: Config, episodes: int, epsilon: float, base_seed: int, render: bool = False) -> dict:
    env = make_from_config(cfg, render_mode="human" if render else None)
    rewards, xs, flags, lengths, truncs, traj_hashes = [], [], [], [], [], set()
    mario.net.eval()
    for i in range(episodes):
        rng = np.random.default_rng(base_seed + i)
        state, info = env.reset(seed=base_seed + i)
        total, actions = 0.0, []
        while True:
            a = mario.act(state, epsilon=epsilon, rng=rng)
            actions.append(a)
            state, r, terminated, truncated, info = env.step(a)
            total += r
            if terminated or truncated:
                break
        rewards.append(total)
        xs.append(info["x_pos"])
        flags.append(bool(info["flag_get"]))
        lengths.append(len(actions))
        truncs.append(bool(truncated))
        traj_hashes.add(hashlib.sha1(np.asarray(actions, dtype=np.int8).tobytes()).hexdigest())
    env.close()
    return {
        "episodes": episodes,
        "epsilon": epsilon,
        "base_seed": base_seed,
        "world": cfg.world,
        "stage": cfg.stage,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_x_pos": float(np.mean(xs)),
        "max_x_pos": int(np.max(xs)),
        "flag_rate": float(np.mean(flags)),
        "no_progress_truncation_rate": float(np.mean(truncs)),
        "mean_length": float(np.mean(lengths)),
        "unique_trajectories": len(traj_hashes),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--world", type=int)
    p.add_argument("--stage", type=int)
    p.add_argument("--noop-max", type=int)
    p.add_argument("--render", action="store_true")
    p.add_argument("--out", type=Path, help="write results JSON here (default: next to checkpoint)")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    overrides = {k: v for k, v in {"world": args.world, "stage": args.stage, "noop_max": args.noop_max}.items()
                 if v is not None}
    cfg = replace(Config.from_dict(ckpt["config"]), device="cpu", **overrides)
    mario = Mario(replace(cfg, buffer_size=16), n_actions=len(ACTION_SETS[cfg.actions]))  # eval needs no buffer
    mario.net.load_state_dict(ckpt["model"])

    results = {"checkpoint": str(args.checkpoint), "train_step": ckpt["curr_step"],
               **evaluate(mario, cfg, args.episodes, args.epsilon, args.seed, args.render)}
    print(json.dumps(results, indent=2))
    out = args.out or args.checkpoint.with_name(
        f"eval_step{ckpt['curr_step']}_w{cfg.world}s{cfg.stage}.json")
    out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
