"""Why does the agent die in pits? Record every takeoff and classify pit deaths against measured jump physics.

    python -m scripts.pit_forensics runs/ppo_1h_a/latest_1h_frozen.pt --episodes 6

Per takeoff (ground -> air): speed, whether B (run) was held, how many steps A was held, distance to the next
pit edge and that pit's width (from the RAM metatile buffer), how far the jump went, and the outcome.
Per pit death: walked off without jumping / too slow for the gap / released A too early / took off too early /
other, plus how many steps the pit edge had been on screen and how likely the policy was to jump beforehand.
Physics reference: runs/analysis/jump_physics_8-1.csv (scripts/jump_physics.py).
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from agent.macros import MacroStepper
from agent.ppo import PPOAgent
from config import PPOConfig
from env.tiles import TileMarioEnv, _s8, screen_left_x

p = argparse.ArgumentParser()
p.add_argument("checkpoint", type=Path)
p.add_argument("--episodes", type=int, default=6, help="per stage")
p.add_argument("--seed", type=int, default=70_000)
p.add_argument("--physics", type=Path, default=Path("runs/analysis/jump_physics_8-1.csv"))
p.add_argument("--out-dir", type=Path, default=Path("runs/analysis"))
p.add_argument("--mode", choices=["safe", "insane"], default="safe")
args = p.parse_args()

# ---- max clean jump distance (px) by speed class and A-hold steps, from the measured table
phys = [r for r in csv.DictReader(open(args.physics)) if r["valid"] == "True"]


def max_distance(speed_class: str, hold: int) -> float:
    mode, runups = {"stand": ("walk", {"0"}), "walk": ("walk", {"6", "12"}), "run": ("run", {"6", "12"})}[speed_class]
    vals = [float(r["distance_px"]) for r in phys
            if r["mode"] == mode and r["runup_steps"] in runups and int(r["hold_steps"]) <= max(hold, 1)]
    return max(vals) if vals else 0.0


def speed_class(vx: int) -> str:
    return "run" if vx >= 36 else ("walk" if vx >= 12 else "stand")


torch.set_num_threads(4)
ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
cfg = PPOConfig.from_dict({**ckpt["config"], "noop_max": 30})
train, test = cfg.resolved_stages()
env = TileMarioEnv(**cfg.env_kwargs(train + test))
env.prebuild()
stepper = MacroStepper(env, cfg.actions)
agent = PPOAgent(cfg, env.n_policy_actions, env.n_extras)
agent.net.load_state_dict(ckpt["model"])
agent.net.eval()
A_ACTIONS = [i for i, b in enumerate(env.action_set) if "A" in b]  # joypad actions holding A
B_ACTIONS = {i for i, b in enumerate(env.action_set) if "B" in b}
# policy actions (buttons or macros) that press A, for "how likely was a jump"
POLICY_A = [p for p in range(env.n_policy_actions) if stepper.expand(p)[0] in A_ACTIONS]


def column_supported(ram, level_x: int, feet_row: int) -> bool:
    """Is there solid ground at or below Mario's feet row in the column containing level_x?"""
    page, col = (level_x // 256) % 2, (level_x % 256) // 16
    return any(ram[0x500 + page * 208 + r * 16 + col] != 0 for r in range(max(feet_row, 0), 13))


def pit_ahead(ram, x: int, feet_row: int):
    """(distance in px from Mario's front to the next unsupported column, pit width in px) within 10 tiles."""
    front = x + 12
    start = None
    for k in range(0, 160, 4):
        supported = column_supported(ram, front + k, feet_row)
        if start is None and not supported:
            start = k
        elif start is not None and supported:
            return start, k - start
    return (start, 160 - start) if start is not None else (None, None)


jumps, pit_deaths = [], []
ep_count = 0
for stage in train + test:
    for i in range(args.episodes):
        seed = args.seed + ep_count
        ep_count += 1
        gen = torch.Generator().manual_seed(seed)
        obs, _ = env.reset(seed=seed, stage=stage, mode=args.mode, practice=False)
        ram = env.ram
        trace = []
        open_jump = None
        queue = []  # joypad actions still to play for the current macro
        p_jump = 0.0
        while True:
            if not queue:
                probs, _ = agent.probs(obs)
                p_jump = float(probs[POLICY_A].sum())
                joy, length = stepper.expand(int(torch.multinomial(probs, 1, generator=gen)))
                queue = [joy] * length
            a = queue.pop(0)
            before_state = int(ram[0x1D])
            x_before = int(ram[0x6D]) * 256 + int(ram[0x86])
            big = int(ram[0x756]) > 0
            feet_row = (int(ram[0xCE]) + (16 if big else 0)) // 16
            pit_dist, pit_w = pit_ahead(ram, x_before, feet_row) if before_state == 0 else (None, None)
            trace.append({"x": x_before, "left": screen_left_x(ram), "p_jump": p_jump,
                          "a": a, "vx": _s8(ram[0x57]), "state": before_state, "pit_dist": pit_dist, "pit_w": pit_w,
                          "y": int(ram[0xCE])})
            obs, _, terminated, truncated, info = env.step(a)
            after_state = int(ram[0x1D])

            if open_jump is None and before_state == 0 and after_state in (1, 2):
                t0 = trace[-1]
                open_jump = {"stage": stage, "seed": seed, "kind": "jump" if after_state == 1 else "walk_off",
                             "takeoff_x": t0["x"], "speed": t0["vx"], "speed_class": speed_class(abs(t0["vx"])),
                             "run_held": a in B_ACTIONS, "hold_steps": 0, "pit_dist_px": t0["pit_dist"],
                             "pit_width_px": t0["pit_w"], "y0": t0["y"], "peak_px": 0, "start_idx": len(trace) - 1}
            if open_jump is not None:
                if a in A_ACTIONS and open_jump["hold_steps"] == len(trace) - 1 - open_jump["start_idx"]:
                    open_jump["hold_steps"] += 1
                open_jump["peak_px"] = max(open_jump["peak_px"], open_jump["y0"] - int(ram[0xCE]))
                done_air = after_state == 0 or terminated or truncated
                if done_air:
                    x_now = int(info["x_pos"])
                    open_jump["distance_px"] = x_now - open_jump["takeoff_x"]
                    cause = info.get("episode", {}).get("death_cause", "") if terminated else ""
                    open_jump["outcome"] = cause if cause else ("landed" if after_state == 0 else "ended")
                    jumps.append(open_jump)
                    if cause == "pit":
                        j = open_jump
                        edge_x = j["takeoff_x"] + 12 + (j["pit_dist_px"] or 0)
                        visible = sum(1 for s in trace[: j["start_idx"] + 1] if edge_x - s["left"] < 256)
                        need = (j["pit_dist_px"] or 0) + (j["pit_width_px"] or 0)
                        can = max_distance(j["speed_class"], j["hold_steps"])
                        can_full_hold = max_distance(j["speed_class"], 12)
                        can_running = max_distance("run", 12)
                        if j["kind"] == "walk_off":
                            why = "walked off edge (no jump)"
                        elif j["pit_dist_px"] is None:
                            why = "other (no pit found ahead at takeoff)"
                        elif can_full_hold < need <= can_running:
                            why = "too slow (needed a running jump)"
                        elif can < need <= can_full_hold:
                            why = "released A too early"
                        elif j["pit_width_px"] and max_distance(j["speed_class"], j["hold_steps"]) >= j["pit_width_px"] + 12 \
                                and j["pit_dist_px"] > 24:
                            why = "took off too early"
                        elif need > can_running:
                            why = "gap too far for any jump from there"
                        else:
                            why = "other (had enough jump; hazard, ceiling or platform)"
                        pit_deaths.append({**j, "why": why, "need_px": need, "can_px": can, "visible_steps": visible,
                                           "p_jump_last4": float(np.mean([s["p_jump"] for s in
                                                                          trace[max(0, j["start_idx"] - 3): j["start_idx"] + 1]]))})
                    open_jump = None
            if terminated or truncated:
                break
env.close()

args.out_dir.mkdir(parents=True, exist_ok=True)
tag = args.checkpoint.parent.name
with open(args.out_dir / f"jumps_{tag}.csv", "w", newline="") as f:
    fields = sorted({k for j in jumps for k in j})
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(jumps)

pit_related = [j for j in jumps if j["kind"] == "jump" and j["pit_dist_px"] is not None and j["pit_dist_px"] <= 48]
ok = [j for j in pit_related if j["outcome"] != "pit"]
bad = [j for j in pit_related if j["outcome"] == "pit"]
summary = {
    "checkpoint": str(args.checkpoint), "episodes": ep_count, "takeoffs": len(jumps),
    "pit_deaths": len(pit_deaths),
    "pit_death_reasons": dict(Counter(d["why"] for d in pit_deaths).most_common()),
    "pit_jumps_with_pit_within_3_tiles": {"cleared": len(ok), "died": len(bad)},
    "cleared_pit_jumps_median": {k: float(np.median([j[k] for j in ok])) if ok else None
                                 for k in ("speed", "hold_steps", "pit_dist_px", "pit_width_px", "distance_px")},
    "failed_pit_jumps_median": {k: float(np.median([j[k] for j in bad])) if bad else None
                                for k in ("speed", "hold_steps", "pit_dist_px", "pit_width_px", "distance_px")},
    "run_held_at_takeoff": {"cleared": float(np.mean([j["run_held"] for j in ok])) if ok else None,
                            "failed": float(np.mean([j["run_held"] for j in bad])) if bad else None},
    "pit_edge_visible_steps_before_takeoff_median": float(np.median([d["visible_steps"] for d in pit_deaths]))
    if pit_deaths else None,
    "policy_jump_probability_last4_steps_before_pit_death_median": float(
        np.median([d["p_jump_last4"] for d in pit_deaths])) if pit_deaths else None,
    "pit_deaths_by_stage": dict(Counter(d["stage"] for d in pit_deaths).most_common()),
}
(args.out_dir / f"pit_forensics_{tag}.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
