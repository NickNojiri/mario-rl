"""Where does rollout time actually go, and how does it scale with n_envs?

    python -m scripts.bench_envs                              # the full sweep, writes docs/results/bench_envs.csv
    python -m scripts.bench_envs --n-envs 1,12 --steps 200    # a quick look

Four costs are separated:

  emulator          one skip-frame group of NES emulation, measured by calling the env's raw step directly
  obs_build         decoding RAM into the tile grid + extras, measured by calling the env's observation builder
  policy_forward    one batched agent.act() at this n_envs, in the main process
  ipc_contention    derived: the vec-env's measured per-step wall time minus (emulator + obs_build + policy)

The first three are measured in-process on a single env, so they are clean per-step costs with no worker
processes involved. The fourth is a residual, not a direct measurement: it absorbs pickling, pipe traffic,
scheduler queueing, and -- once n_envs exceeds the physical core count -- core contention. It can go negative
if the workers overlap better than the serial model assumes. Both the measured parts and the residual are
written to the CSV so the residual can be recomputed or ignored.

The sweep takes a while; nothing here is run automatically.
"""
from __future__ import annotations

import argparse
import csv
import os
import platform
import time
from pathlib import Path

import torch

from agent.ppo import PPOAgent
from agent.vec_env import SubprocVecEnv
from config import get_ppo_preset
from env.tiles import ACTION_SETS, TileMarioEnv, n_extras, n_policy_actions

FIELDS = ["n_envs", "steps_per_env", "agent_steps", "wall_s", "agent_steps_per_s", "vec_step_s", "emulator_s",
          "obs_build_s", "policy_forward_s", "ipc_contention_s", "preset", "cpu_model", "logical_cores",
          "torch_threads", "timestamp", "method"]
METHOD = ("emulator/obs_build measured in-process on one env; policy_forward batched at n_envs; "
          "ipc_contention = vec_step_s - (emulator_s + obs_build_s + policy_forward_s), a residual")


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def time_components(cfg, stages, steps: int, warmup: int) -> tuple[float, float]:
    """Seconds per agent step for (emulator, observation build), measured on one env, no subprocesses."""
    env = TileMarioEnv(**cfg.env_kwargs(stages))
    right = ACTION_SETS[cfg.actions].index(["right"]) if ["right"] in ACTION_SETS[cfg.actions] else 1
    try:
        env.reset(seed=0)
        for _ in range(warmup):
            _, terminated, truncated, _ = env._raw_skip(right, use_joypad=True)
            if terminated or truncated:
                env.reset(seed=0)

        emulator = 0.0
        done_count = 0
        for _ in range(steps):
            t0 = time.perf_counter()
            _, terminated, truncated, _ = env._raw_skip(right, use_joypad=True)
            emulator += time.perf_counter() - t0
            if terminated or truncated:
                env.reset(seed=done_count)  # resets are not counted; they are not part of a steady-state step
                done_count += 1

        obs_build = 0.0
        for _ in range(steps):
            t0 = time.perf_counter()
            env._obs()
            obs_build += time.perf_counter() - t0
        return emulator / steps, obs_build / steps
    finally:
        env.close()


def bench_one(n_envs: int, cfg, cfg_name: str, stages, steps: int, warmup: int) -> dict:
    agent = PPOAgent(cfg, n_policy_actions(cfg.actions),
                     n_extras(len(ACTION_SETS[cfg.actions]), cfg.obs_version, cfg.reward_version >= 4))
    envs = SubprocVecEnv(n_envs, cfg.env_kwargs(stages))
    try:
        obs = envs.reset(seed=0)
        for _ in range(warmup):
            action, _, _ = agent.act(obs)
            obs = envs.step(action)[0]

        policy_s = step_s = 0.0
        t_start = time.perf_counter()
        for _ in range(steps):
            t0 = time.perf_counter()
            action, _, _ = agent.act(obs)
            t1 = time.perf_counter()
            obs = envs.step(action)[0]
            t2 = time.perf_counter()
            policy_s += t1 - t0
            step_s += t2 - t1
        wall = time.perf_counter() - t_start
    finally:
        envs.close()

    emulator, obs_build = time_components(cfg, stages, steps, warmup)
    vec_step = step_s / steps
    policy = policy_s / steps
    return {
        "n_envs": n_envs, "steps_per_env": steps, "agent_steps": steps * n_envs,
        "wall_s": round(wall, 4), "agent_steps_per_s": round(steps * n_envs / wall, 1),
        "vec_step_s": round(vec_step, 6), "emulator_s": round(emulator, 6), "obs_build_s": round(obs_build, 6),
        "policy_forward_s": round(policy, 6),
        "ipc_contention_s": round(vec_step - (emulator + obs_build + policy), 6),
        "preset": cfg_name, "cpu_model": cpu_model(), "logical_cores": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "method": METHOD,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-envs", default="1,2,4,8,12,16", help="comma list")
    p.add_argument("--steps", type=int, default=300, help="timed steps per env, after warmup")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--preset", default="ppo_full")
    p.add_argument("--out", type=Path, default=Path("docs/results/bench_envs.csv"))
    args = p.parse_args()

    cfg = get_ppo_preset(args.preset)
    torch.set_num_threads(cfg.rollout_threads)
    stages, _ = cfg.resolved_stages()

    rows = []
    for n in [int(x) for x in args.n_envs.split(",") if x]:
        row = bench_one(n, cfg, args.preset, stages, args.steps, args.warmup)
        rows.append(row)
        print(f"n_envs={n:>3} {row['agent_steps_per_s']:>8.1f} steps/s   "
              f"emu={row['emulator_s'] * 1e3:.2f}ms obs={row['obs_build_s'] * 1e3:.2f}ms "
              f"policy={row['policy_forward_s'] * 1e3:.2f}ms ipc={row['ipc_contention_s'] * 1e3:+.2f}ms", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    new = not args.out.exists()
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")
