"""Train a Double-DQN Mario agent.

    python train.py --preset smoke
    python train.py --preset full --run-name dqn_1_1
    python train.py --resume runs/dqn_1_1/latest.pt --total-steps 5000000
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import signal
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from agent.mario import Mario
from config import BOOKKEEPING_FIELDS, Config, get_preset
from env.make_env import make_from_config

EPISODE_FIELDS = [
    "episode", "step", "epsilon", "reward", "length", "x_pos", "flag_get",
    "terminated", "truncated", "mean_loss", "mean_q", "steps_per_sec", "wall_time",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["smoke", "full"], default="smoke")
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", type=Path, default=None, help="checkpoint to resume; its config is reused")
    p.add_argument("--allow-config-change", action="store_true")
    # Bookkeeping overrides are always safe; learning overrides change the hash.
    p.add_argument("--total-steps", type=int)
    p.add_argument("--save-every", type=int)
    p.add_argument("--save-buffer", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--device")
    p.add_argument("--world", type=int)
    p.add_argument("--stage", type=int)
    p.add_argument("--seed", type=int)
    return p.parse_args()


def build_config(args) -> tuple[Config, Path]:
    overrides = {
        k: v for k, v in {
            "total_steps": args.total_steps, "save_every": args.save_every, "save_buffer": args.save_buffer,
            "device": args.device, "world": args.world, "stage": args.stage, "seed": args.seed,
        }.items() if v is not None
    }
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        cfg = replace(Config.from_dict(ckpt["config"]), **overrides)
        run_dir = args.resume.parent
    else:
        cfg = get_preset(args.preset, **overrides)
        run_dir = Path("runs") / (args.run_name or f"{args.preset}_{time.strftime('%Y%m%d_%H%M%S')}")
    changed = set(overrides) - BOOKKEEPING_FIELDS
    if args.resume and changed and not args.allow_config_change:
        raise SystemExit(f"overriding learning fields {sorted(changed)} on resume needs --allow-config-change")
    return cfg, run_dir


def main():
    args = parse_args()
    # kill/pkill send SIGTERM; route it through the same save-on-exit path as Ctrl+C.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    cfg, run_dir = build_config(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    if cfg.torch_threads:
        torch.set_num_threads(cfg.torch_threads)

    env = make_from_config(cfg)
    mario = Mario(cfg, env.n_actions)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if args.resume:
        restored_buffer = mario.load(args.resume, allow_config_change=args.allow_config_change)
        print(f"resumed step={mario.curr_step} episode={mario.episode} eps={mario.exploration_rate:.3f} "
              f"buffer={'restored' if restored_buffer else f'empty, refilling to burnin={cfg.burnin}'}")
    (run_dir / "config.json").write_text(json.dumps({**cfg.to_dict(), "learning_hash": cfg.learning_hash()}, indent=2))

    log_path = run_dir / "episodes.csv"
    new_log = not log_path.exists()
    log_file = open(log_path, "a", newline="")
    writer = csv.DictWriter(log_file, fieldnames=EPISODE_FIELDS)
    if new_log:
        writer.writeheader()

    ckpt_path = run_dir / "latest.pt"
    # monotonic: WSL2 resyncs its wall clock with Windows, which made time.time() deltas go negative.
    t_start = time.monotonic()
    try:
        while mario.curr_step < cfg.total_steps:
            mario.episode += 1
            state, _ = env.reset(seed=cfg.seed * 1_000_003 + mario.episode)
            ep_reward, ep_len, losses, qs = 0.0, 0, [], []
            ep_t0, ep_step0 = time.monotonic(), mario.curr_step
            while True:
                action = mario.act(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                mario.cache(state, action, reward, next_state, terminated, truncated)
                out = mario.learn()
                if out is not None:
                    qs.append(out[0])
                    losses.append(out[1])
                ep_reward += reward
                ep_len += 1
                state = next_state

                if mario.curr_step % cfg.save_every == 0:
                    mario.save(ckpt_path, save_buffer=cfg.save_buffer)
                    print(f"[ckpt] step={mario.curr_step} -> {ckpt_path}")
                if mario.curr_step % cfg.snapshot_every == 0:
                    snap = run_dir / "snapshots" / f"step_{mario.curr_step:08d}.pt"
                    mario.save(snap, save_buffer=False)  # weights + optimizer only, ~27 MB
                if terminated or truncated or mario.curr_step >= cfg.total_steps:
                    break

            dt = time.monotonic() - ep_t0
            row = {
                "episode": mario.episode, "step": mario.curr_step, "epsilon": round(mario.exploration_rate, 4),
                "reward": ep_reward, "length": ep_len, "x_pos": info["x_pos"], "flag_get": int(info["flag_get"]),
                "terminated": int(terminated), "truncated": int(truncated),
                "mean_loss": round(float(np.mean(losses)), 5) if losses else "",
                "mean_q": round(float(np.mean(qs)), 4) if qs else "",
                "steps_per_sec": round((mario.curr_step - ep_step0) / max(dt, 1e-9), 1),
                "wall_time": round(time.monotonic() - t_start, 1),
            }
            writer.writerow(row)
            log_file.flush()
            if mario.episode % cfg.log_every_episodes == 0:
                print(" ".join(f"{k}={v}" for k, v in row.items()))
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        mario.buffer.end_episode()
        mario.save(ckpt_path, save_buffer=cfg.save_buffer)
        print(f"[ckpt] final step={mario.curr_step} -> {ckpt_path}")
        log_file.close()
        env.close()


if __name__ == "__main__":
    main()
