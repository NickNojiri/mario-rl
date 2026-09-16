"""Train PPO on tile-grid observations across many real SMB stages (held-out stages are never sampled).

    python train_ppo.py --preset ppo_smoke
    python train_ppo.py --preset ppo_full --run-name ppo_tiles_1
    python train_ppo.py --resume runs/ppo_tiles_1/latest.pt --total-steps 40000000
    python train_ppo.py --preset ppo_full --run-name no_coins --coin-reward 0   # coin-incentive ablation
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

from agent.ppo import PPOAgent, compute_gae
from agent.vec_env import SubprocVecEnv, stack_obs
from config import PPO_BOOKKEEPING_FIELDS, PPOConfig, get_ppo_preset
from env.tiles import ACTION_SETS, n_extras

UPDATE_FIELDS = ["update", "step", "steps_per_sec", "episodes", "mean_x_pos", "flag_rate", "mean_coins",
                 "mean_game_reward", "stall_rate", "policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac",
                 "explained_variance", "wall_time"]
EPISODE_FIELDS = ["step", "stage", "x_pos", "flag_get", "coins", "game_reward", "reward", "length", "terminated",
                  "truncated"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["ppo_smoke", "ppo_full"], default="ppo_smoke")
    p.add_argument("--run-name")
    p.add_argument("--resume", type=Path)
    p.add_argument("--allow-config-change", action="store_true")
    p.add_argument("--total-steps", type=int)
    p.add_argument("--n-envs", type=int)
    p.add_argument("--coin-reward", type=float)
    p.add_argument("--flag-reward", type=float)
    p.add_argument("--seed", type=int)
    return p.parse_args()


def main():
    args = parse_args()
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    overrides = {k: v for k, v in {"total_steps": args.total_steps, "n_envs": args.n_envs,
                                   "coin_reward": args.coin_reward, "flag_reward": args.flag_reward,
                                   "seed": args.seed}.items() if v is not None}
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        cfg = replace(PPOConfig.from_dict(ckpt["config"]), **overrides)
        run_dir = args.resume.parent
        changed = set(overrides) - PPO_BOOKKEEPING_FIELDS
        if changed and not args.allow_config_change:
            raise SystemExit(f"overriding learning fields {sorted(changed)} on resume needs --allow-config-change")
    else:
        cfg = get_ppo_preset(args.preset, **overrides)
        run_dir = Path("runs") / (args.run_name or f"{args.preset}_{time.strftime('%Y%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(cfg.torch_threads)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    train_stages, test_stages = cfg.resolved_stages()
    assert not set(train_stages) & set(test_stages), "held-out stages leaked into training"
    n_actions = len(ACTION_SETS[cfg.actions])
    agent = PPOAgent(cfg, n_actions, n_extras(n_actions))
    if args.resume:
        agent.load(args.resume, allow_config_change=args.allow_config_change)
        print(f"resumed step={agent.global_step} updates={agent.updates}")
    (run_dir / "config.json").write_text(json.dumps(
        {**cfg.to_dict(), "resolved_train_stages": train_stages, "resolved_test_stages": test_stages,
         "learning_hash": cfg.learning_hash()}, indent=2))

    envs = SubprocVecEnv(cfg.n_envs, cfg.env_kwargs(train_stages))
    obs = envs.reset(seed=cfg.seed * 10_007 + agent.updates * cfg.n_envs)

    def open_csv(name, fields):
        path = run_dir / name
        new = not path.exists()
        f = open(path, "a", newline="")
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        return f, w

    upd_file, upd_writer = open_csv("updates.csv", UPDATE_FIELDS)
    ep_file, ep_writer = open_csv("episodes.csv", EPISODE_FIELDS)

    T, N = cfg.rollout_len, cfg.n_envs
    grid_shape = obs["tiles"].shape[1:]
    buf = {
        "tiles": np.zeros((T, N, *grid_shape), np.int16),
        "extras": np.zeros((T, N, obs["extras"].shape[1]), np.float32),
        "actions": np.zeros((T, N), np.int64), "logp": np.zeros((T, N), np.float32),
        "values": np.zeros((T, N), np.float32), "rewards": np.zeros((T, N), np.float32),
        "terminated": np.zeros((T, N), bool), "truncated": np.zeros((T, N), bool),
        "trunc_values": np.zeros((T, N), np.float32),
    }
    t_start = time.monotonic()
    last_save = agent.global_step // cfg.save_every
    last_snap = agent.global_step // cfg.snapshot_every
    try:
        while agent.global_step < cfg.total_steps:
            t_upd = time.monotonic()
            episodes = []
            buf["trunc_values"][:] = 0.0
            for t in range(T):
                action, logp, value = agent.act(obs)
                next_obs, reward, terminated, truncated, infos = envs.step(action)
                buf["tiles"][t], buf["extras"][t] = obs["tiles"], obs["extras"]
                buf["actions"][t], buf["logp"][t], buf["values"][t] = action, logp, value
                buf["rewards"][t] = reward * cfg.reward_scale
                buf["terminated"][t], buf["truncated"][t] = terminated, truncated

                cut = [i for i in range(N) if truncated[i] and not terminated[i]]
                if cut:
                    final = stack_obs([infos[i]["final_obs"] for i in cut])
                    buf["trunc_values"][t, cut] = agent.value(final)
                for info in infos:
                    if "episode" in info:
                        ep = info["episode"]
                        episodes.append(ep)
                        ep_writer.writerow({"step": agent.global_step, **{k: ep[k] for k in EPISODE_FIELDS[1:]}})
                obs = next_obs
                agent.global_step += N

            adv, returns = compute_gae(buf["rewards"], buf["values"], agent.value(obs), buf["terminated"],
                                       buf["truncated"], buf["trunc_values"], cfg.gamma, cfg.gae_lambda)
            flat = lambda a: a.reshape(T * N, *a.shape[2:])
            stats = agent.update({"tiles": flat(buf["tiles"]), "extras": flat(buf["extras"]),
                                  "actions": flat(buf["actions"]), "logp": flat(buf["logp"]),
                                  "values": flat(buf["values"]), "advantages": flat(adv), "returns": flat(returns)})

            mean = lambda key: float(np.mean([e[key] for e in episodes])) if episodes else ""
            row = {"update": agent.updates, "step": agent.global_step,
                   "steps_per_sec": round(T * N / (time.monotonic() - t_upd), 1), "episodes": len(episodes),
                   "mean_x_pos": mean("x_pos"), "flag_rate": mean("flag_get"), "mean_coins": mean("coins"),
                   "mean_game_reward": mean("game_reward"),
                   "stall_rate": float(np.mean([e["truncated"] for e in episodes])) if episodes else "",
                   **{k: round(v, 5) for k, v in stats.items()},
                   "wall_time": round(time.monotonic() - t_start, 1)}
            upd_writer.writerow(row)
            upd_file.flush()
            ep_file.flush()
            print(" ".join(f"{k}={round(v, 3) if isinstance(v, float) else v}" for k, v in row.items()), flush=True)

            if agent.global_step // cfg.save_every > last_save:
                last_save = agent.global_step // cfg.save_every
                agent.save(run_dir / "latest.pt")
            if agent.global_step // cfg.snapshot_every > last_snap:
                last_snap = agent.global_step // cfg.snapshot_every
                agent.save(run_dir / "snapshots" / f"step_{agent.global_step:09d}.pt")
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        agent.save(run_dir / "latest.pt")
        print(f"[ckpt] final step={agent.global_step} -> {run_dir / 'latest.pt'}", flush=True)
        upd_file.close()
        ep_file.close()
        envs.close()


if __name__ == "__main__":
    main()
