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
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch

from agent.ppo import PPOAgent, compute_gae, compute_gae_options
from agent.vec_env import SubprocVecEnv, stack_obs
from config import PPO_BOOKKEEPING_FIELDS, PPOConfig, get_ppo_preset
from env.tiles import ACTION_SETS, MACRO_SETS, n_extras, n_policy_actions

UPDATE_FIELDS = ["update", "step", "steps_per_sec", "rollout_sec", "learn_sec", "episodes", "mean_x_pos", "flag_rate",
                 "mean_coins", "mean_game_reward", "stall_rate", "pit_death_rate", "enemy_death_rate", "left_share",
                 "noop_share", "points_per_episode", "hurts_per_episode", "practice_episodes",
                 "practice_pit_death_rate", "decision_share", "gen_episodes", "gen_flag_rate", "real_x_pos",
                 "policy_loss", "value_loss", "entropy", "approx_kl",
                 "clipfrac", "explained_variance", "wall_time"]
EPISODE_FIELDS = ["step", "stage", "mode", "practice", "procgen", "difficulty", "x_pos", "max_x", "flag_get",
                  "death_cause", "coins", "points",
                  "point_events",
                  "hurts", "jumps", "left_presses", "noop_presses", "game_reward", "reward", "length", "terminated",
                  "truncated", "r_progress", "r_time", "r_death", "r_hurt", "r_points", "r_coins", "r_flag"]
PRESETS = ["ppo_smoke", "ppo_full", "ppo_1h", "ppo_1h_v3b", "ppo_1h_v3c", "ppo_1h_v3d", "ppo_smoke_v3",
           "ppo_1h_v4", "ppo_smoke_v4", "ppo_sweep", "ppo_1h_gen"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=PRESETS, default="ppo_smoke")
    p.add_argument("--run-name")
    p.add_argument("--resume", type=Path)
    p.add_argument("--allow-config-change", action="store_true")
    p.add_argument("--total-steps", type=int)
    p.add_argument("--snapshot-every", type=int)
    p.add_argument("--n-envs", type=int)
    p.add_argument("--coin-reward", type=float)
    p.add_argument("--flag-reward", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override any config field, e.g. --set lr=0.0005 ent_coef=0.03 actions=simple_macro")
    return p.parse_args()


def parse_set(pairs: list[str]) -> dict:
    """Type the KEY=VALUE overrides from the PPOConfig dataclass fields."""
    types = {f.name: f.type for f in fields(PPOConfig)}
    out = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        if key not in types:
            raise SystemExit(f"unknown config field: {key}")
        t = types[key]
        if t == "bool":
            out[key] = raw.lower() in ("1", "true", "yes")
        elif t == "int":
            out[key] = int(raw)
        elif t == "float":
            out[key] = float(raw)
        elif t == "tuple":
            out[key] = tuple(x for x in raw.split(",") if x)
        else:
            out[key] = raw
    return out


def main():
    args = parse_args()
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    overrides = {k: v for k, v in {"total_steps": args.total_steps, "snapshot_every": args.snapshot_every,
                                   "n_envs": args.n_envs,
                                   "coin_reward": args.coin_reward, "flag_reward": args.flag_reward,
                                   "seed": args.seed}.items() if v is not None}
    overrides.update(parse_set(args.set))
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
    n_joy = len(ACTION_SETS[cfg.actions])
    macros = MACRO_SETS.get(cfg.actions, [])
    agent = PPOAgent(cfg, n_policy_actions(cfg.actions), n_extras(n_joy, cfg.obs_version, cfg.reward_version >= 4))
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
        if path.exists():
            with open(path, newline="") as existing:
                header = next(csv.reader(existing), [])
            if header != fields:  # resumed with a different logging schema: keep the old file, start a new one
                path.rename(path.with_name(f"{path.stem}_upto_{agent.global_step}{path.suffix}"))
        new = not path.exists()
        f = open(path, "a", newline="")
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        return f, w

    upd_file, upd_writer = open_csv("updates.csv", UPDATE_FIELDS)
    ep_file, ep_writer = open_csv("episodes.csv", EPISODE_FIELDS)

    T, N = cfg.rollout_len, cfg.n_envs
    obs_buf = {k: np.zeros((T, *v.shape), v.dtype) for k, v in obs.items()}  # every observation key, any version
    buf = {
        "actions": np.zeros((T, N), np.int64), "logp": np.zeros((T, N), np.float32),
        "values": np.zeros((T, N), np.float32), "rewards": np.zeros((T, N), np.float32),
        "terminated": np.zeros((T, N), bool), "truncated": np.zeros((T, N), bool),
        "trunc_values": np.zeros((T, N), np.float32), "decision": np.ones((T, N), bool),
    }
    # Macro state per env: joypad action being held and forced steps remaining after the current one.
    held_joy = np.zeros(N, np.int64)
    remaining = np.zeros(N, np.int64)
    stage_now = [None] * N
    plr_score = {s: 1.0 for s in train_stages}  # EMA of mean positive advantage per stage (learning potential)
    t_start = time.monotonic()
    last_save = agent.global_step // cfg.save_every
    last_snap = agent.global_step // cfg.snapshot_every
    try:
        while agent.global_step < cfg.total_steps:
            t_upd = time.monotonic()
            torch.set_num_threads(cfg.rollout_threads)
            episodes = []
            buf["trunc_values"][:] = 0.0
            stage_of_step = np.empty((T, N), dtype=object)
            for t in range(T):
                action, logp, value = agent.act(obs)
                decision = remaining == 0
                joy = np.where(decision, action, held_joy)
                for i in np.flatnonzero(decision):
                    if action[i] >= n_joy:  # a macro: hold its buttons for its length
                        _, held_joy[i], length = macros[action[i] - n_joy]
                        joy[i], remaining[i] = held_joy[i], length - 1
                remaining[~decision] -= 1
                next_obs, reward, terminated, truncated, infos = envs.step(joy)
                for k, v in obs.items():
                    obs_buf[k][t] = v
                buf["actions"][t], buf["logp"][t], buf["values"][t] = action, logp, value
                buf["decision"][t] = decision
                buf["rewards"][t] = reward * cfg.reward_scale
                buf["terminated"][t], buf["truncated"][t] = terminated, truncated
                remaining[terminated | truncated] = 0  # a macro never carries into the next episode

                cut = [i for i in range(N) if truncated[i] and not terminated[i]]
                if cut:
                    final = stack_obs([infos[i]["final_obs"] for i in cut])
                    buf["trunc_values"][t, cut] = agent.value(final)
                for i, info in enumerate(infos):
                    stage_of_step[t, i] = info["stage"]
                    if "episode" in info:
                        ep = info["episode"]
                        episodes.append(ep)
                        ep_writer.writerow({"step": agent.global_step, **{k: ep.get(k, "") for k in EPISODE_FIELDS[1:]}})
                obs = next_obs
                agent.global_step += N

            t_learn = time.monotonic()
            torch.set_num_threads(cfg.torch_threads)
            if macros:
                adv, returns = compute_gae_options(buf["rewards"], buf["values"], agent.value(obs), buf["terminated"],
                                                   buf["truncated"], buf["trunc_values"], buf["decision"], cfg.gamma,
                                                   cfg.gae_lambda)
            else:
                adv, returns = compute_gae(buf["rewards"], buf["values"], agent.value(obs), buf["terminated"],
                                           buf["truncated"], buf["trunc_values"], cfg.gamma, cfg.gae_lambda)
            keep = buf["decision"].reshape(-1)  # forced macro steps are not policy decisions
            flat = lambda a: a.reshape(T * N, *a.shape[2:])[keep]
            stats = agent.update({"obs": {k: flat(v) for k, v in obs_buf.items()},
                                  "actions": flat(buf["actions"]), "logp": flat(buf["logp"]),
                                  "values": flat(buf["values"]), "advantages": flat(adv), "returns": flat(returns)})

            if cfg.plr:  # rank-based prioritized level replay on mean positive advantage, mixed with uniform
                for s in train_stages:
                    m = buf["decision"] & (stage_of_step == s)
                    if m.any():
                        plr_score[s] = 0.7 * plr_score[s] + 0.3 * float(np.clip(adv[m], 0, None).mean())
                order = sorted(train_stages, key=lambda s: -plr_score[s])
                rank_w = {s: (1.0 / (r + 1)) ** (1.0 / cfg.plr_rank_beta) for r, s in enumerate(order)}
                z = sum(rank_w.values())
                weights = {s: cfg.plr_uniform_mix / len(train_stages) + (1 - cfg.plr_uniform_mix) * rank_w[s] / z
                           for s in train_stages}
                envs.set_stage_weights(weights)

            practice_eps = [e for e in episodes if e.get("practice")]
            gen_eps = [e for e in episodes if e.get("procgen")]
            episodes = [e for e in episodes if not e.get("practice")]  # progress stats from real starts only
            mean = lambda key: float(np.mean([e[key] for e in episodes])) if episodes else ""
            share = lambda pred: float(np.mean([pred(e) for e in episodes])) if episodes else ""
            steps_total = sum(e["length"] for e in episodes) or 1
            row = {"update": agent.updates, "step": agent.global_step,
                   "steps_per_sec": round(T * N / (time.monotonic() - t_upd), 1),
                   "rollout_sec": round(t_learn - t_upd, 2), "learn_sec": round(time.monotonic() - t_learn, 2),
                   "episodes": len(episodes),
                   "mean_x_pos": mean("x_pos"), "flag_rate": mean("flag_get"), "mean_coins": mean("coins"),
                   "mean_game_reward": mean("game_reward"),
                   "stall_rate": float(np.mean([e["truncated"] for e in episodes])) if episodes else "",
                   "pit_death_rate": share(lambda e: e["death_cause"] == "pit"),
                   "enemy_death_rate": share(lambda e: e["death_cause"].startswith("enemy")),
                   "left_share": round(sum(e["left_presses"] for e in episodes) / steps_total, 4),
                   "noop_share": round(sum(e["noop_presses"] for e in episodes) / steps_total, 4),
                   "points_per_episode": mean("points"), "hurts_per_episode": mean("hurts"),
                   "practice_episodes": len(practice_eps),
                   "practice_pit_death_rate": (float(np.mean([e["death_cause"] == "pit" for e in practice_eps]))
                                               if practice_eps else ""),
                   "decision_share": round(float(keep.mean()), 3),
                   "gen_episodes": len(gen_eps),
                   "gen_flag_rate": float(np.mean([e["flag_get"] for e in gen_eps])) if gen_eps else "",
                   "real_x_pos": (float(np.mean([e["x_pos"] for e in episodes if not e.get("procgen")]))
                                  if any(not e.get("procgen") for e in episodes) else ""),
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
