"""Held-out evaluation for the multi-stage tile agent.

    python eval_stages.py runs/ppo_tiles_1/latest.pt                 # train + held-out stages
    python eval_stages.py --random --config runs/ppo_tiles_1/config.json   # random baseline, same protocol
    python eval_stages.py runs/ppo_tiles_1/latest.pt --greedy --episodes 20

Reports per-stage and per-group (train vs held-out) progress, flag rate, coins and stall rate. The number
that answers "did it learn the mechanics?" is held-out performance against the random baseline, together
with the gap between train and held-out stages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np


def _eval_stage(job: dict) -> dict:
    import torch

    from agent.macros import MacroStepper
    from agent.ppo import PPOAgent
    from config import PPOConfig
    from env.tiles import TileMarioEnv

    torch.set_num_threads(1)
    cfg = PPOConfig.from_dict(job["config"])
    env = TileMarioEnv(**cfg.env_kwargs([job["stage"]]))
    stepper = MacroStepper(env, cfg.actions)
    agent = None
    if job["checkpoint"]:
        agent = PPOAgent(cfg, env.n_policy_actions, env.n_extras)
        agent.net.load_state_dict(torch.load(job["checkpoint"], map_location="cpu", weights_only=False)["model"])
        agent.net.eval()

    eps = []
    for i in range(job["episodes"]):
        seed = job["seed"] + i
        rng = np.random.default_rng(seed)
        gen = torch.Generator().manual_seed(seed)
        obs, _ = env.reset(seed=seed, stage=job["stage"], mode=job.get("mode"), practice=False)
        actions = []
        while True:
            if agent is None:
                a = int(rng.integers(env.n_policy_actions))
            else:
                batched = {k: v[None] for k, v in obs.items()}
                a = int(agent.act(batched, greedy=job["greedy"], generator=gen)[0][0])
            actions.append(a)
            obs, _, terminated, truncated, info = stepper.step(a)
            if terminated or truncated:
                break
        ep = info["episode"]
        ep["trajectory"] = hashlib.sha1(np.asarray(actions, np.int8).tobytes()).hexdigest()
        ep["seed"] = seed
        eps.append(ep)
    env.close()
    steps = sum(e["length"] for e in eps) or 1
    causes = [e["death_cause"] for e in eps]
    return {
        "stage": job["stage"], "group": job["group"], "episodes": len(eps),
        "mean_x_pos": float(np.mean([e["x_pos"] for e in eps])),
        "std_x_pos": float(np.std([e["x_pos"] for e in eps])),
        "flag_rate": float(np.mean([e["flag_get"] for e in eps])),
        "mean_coins": float(np.mean([e["coins"] for e in eps])),
        "stall_rate": float(np.mean([e["truncated"] for e in eps])),
        "mean_game_reward": float(np.mean([e["game_reward"] for e in eps])),
        "unique_trajectories": len({e["trajectory"] for e in eps}),
        "pit_death_rate": float(np.mean([c == "pit" for c in causes])),
        "enemy_death_rate": float(np.mean([c.startswith("enemy") for c in causes])),
        "left_share": sum(e["left_presses"] for e in eps) / steps,
        "noop_share": sum(e["noop_presses"] for e in eps) / steps,
        "points_per_episode": float(np.mean([e["points"] for e in eps])),
        "hurts_per_episode": float(np.mean([e["hurts"] for e in eps])),
        "death_causes": {c: causes.count(c) for c in sorted(set(causes))},
        # per-episode records, so any single episode can be replayed exactly (scripts/record_gif_ppo.py)
        "episodes_detail": [{k: e[k] for k in ("seed", "x_pos", "max_x", "flag_get", "death_cause", "coins",
                                                "points", "hurts", "jumps", "left_presses", "noop_presses",
                                                "length", "terminated", "truncated", "game_reward")} for e in eps],
    }


SUMMARY_KEYS = ("mean_x_pos", "flag_rate", "mean_coins", "stall_rate", "mean_game_reward", "pit_death_rate",
                "enemy_death_rate", "left_share", "noop_share", "points_per_episode", "hurts_per_episode")


def summarize(rows: list[dict], group: str) -> dict:
    g = [r for r in rows if r["group"] == group]
    if not g:
        return {}
    return {k: float(np.mean([r[k] for r in g])) for k in SUMMARY_KEYS} | {"stages": len(g)}


def eval_config(cfg_dict: dict, noop_max: int):
    """The run's own config, adapted for evaluation.

    --noop-max sets the random start delay, which has to reach the env's reset (see tests). Training-only aids
    are switched off: no practice returns, no generated terrain (each job fixes its own stage).
    """
    from config import PPOConfig

    return PPOConfig.from_dict({**cfg_dict, "noop_max": noop_max, "practice_prob": 0.0, "procgen_prob": 0.0})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", nargs="?", type=Path)
    p.add_argument("--random", action="store_true", help="uniform random policy baseline")
    p.add_argument("--config", type=Path, help="config.json for --random (defaults to ppo_full)")
    p.add_argument("--stages", default="all", help="all | train | test | comma list like 1-1,2-1")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=50_000)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--noop-max", type=int, default=30, help="random start delay at eval (training uses less)")
    p.add_argument("--mode", choices=["safe", "insane"], default="safe",
                   help="play style for mode-conditioned (reward_version 4) checkpoints; ignored by older ones")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if args.random == bool(args.checkpoint):
        p.error("give a checkpoint or --random (not both)")

    import torch

    from config import get_ppo_preset

    if args.checkpoint:
        cfg_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["config"]
    elif args.config:
        cfg_dict = {k: v for k, v in json.loads(args.config.read_text()).items()}
    else:
        cfg_dict = get_ppo_preset("ppo_full").to_dict()
    cfg = eval_config(cfg_dict, args.noop_max)
    train, test = cfg.resolved_stages()
    if args.stages == "all":
        chosen = [(s, "train") for s in train] + [(s, "test") for s in test]
    elif args.stages in ("train", "test"):
        chosen = [(s, args.stages) for s in (train if args.stages == "train" else test)]
    else:
        chosen = [(s, "test" if s in test else "train" if s in train else "other") for s in args.stages.split(",")]

    jobs = [{"stage": s, "group": g, "config": cfg.to_dict(), "episodes": args.episodes, "seed": args.seed,
             "greedy": args.greedy, "checkpoint": str(args.checkpoint) if args.checkpoint else None,
             "mode": args.mode}
            for s, g in chosen]
    with mp.get_context("forkserver").Pool(min(args.workers, len(jobs))) as pool:
        rows = pool.map(_eval_stage, jobs)

    result = {
        "policy": "random" if args.random else str(args.checkpoint),
        "greedy": args.greedy, "episodes_per_stage": args.episodes, "base_seed": args.seed,
        "noop_max": args.noop_max, "mode": args.mode if cfg.reward_version >= 4 else None,
        "train_summary": summarize(rows, "train"), "test_summary": summarize(rows, "test"),
        "stages": rows,
    }
    for r in rows:
        print(f"{r['group']:>5} {r['stage']}: x={r['mean_x_pos']:7.1f} flag={r['flag_rate']:.2f} "
              f"coins={r['mean_coins']:.1f} stall={r['stall_rate']:.2f} unique={r['unique_trajectories']}")
    print("train:", result["train_summary"])
    print("test: ", result["test_summary"])
    out = args.out or ((args.checkpoint.parent if args.checkpoint else Path("runs")) /
                       f"eval_stages_{'random' if args.random else 'policy'}{'_greedy' if args.greedy else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
