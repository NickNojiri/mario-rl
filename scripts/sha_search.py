"""Fail-fast hyperparameter search: successive halving, then a seed-replicated confirmation.

    python -m scripts.sha_search --dry-run              # print the plan and the compute estimate
    python -m scripts.sha_search                        # run (or resume) the search
    python -m scripts.sha_search --confirm              # retrain the winner on fresh seeds and compare
    python -m scripts.sha_search --report               # print the leaderboard from saved state

How it fails fast. Start N configs on a small budget, evaluate, keep the top 1/eta, give the survivors eta times
the budget, repeat. With N=27 and eta=3 that is 27 -> 9 -> 3 -> 1, and a bad config dies after ~7 minutes instead
of consuming a full run. Promoted configs RESUME from their checkpoint rather than restarting: this is equivalent
to one longer run only because nothing in PPOConfig schedules on total_steps (no LR annealing, constant entropy
bonus). If a schedule is ever added, resuming stops being equivalent and this script must restart instead.

Why this is allowed. ADR-0005: short screening ranks within-distribution changes faithfully but systematically
penalizes methods whose payoff is deferred. Everything in SPACE is an optimizer/objective setting or level
reweighting (PLR). Procedural generation, demonstrations and anything else that trains on data outside the
evaluation distribution are deliberately excluded -- a 250k screen would kill them for the wrong reason.

Why there is a confirmation stage. Seed-level SD is ~57 (README, pooled over 9 runs) and eval noise ~46. Picking
the best of 27 noisy single-seed scores selects for luck as well as quality (the winner's curse), so the search
winner is NOT the result. --confirm retrains it on seeds 0, 1, 2 -- none used during the search, which runs on
SEARCH_SEED -- at the ppo_ablate_2m budget, and compares against the default config's existing seeds 0, 1, 2 at
that budget (docs/results/prev_action_ablation/prevact_true_s*, same preset, same protocol).

Selection metric: held-out (now validation) mean x_pos, the project's primary number. The final test set is Lost
Levels; nothing here touches it.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_PRESET = "ppo_ablate_2m"  # the v3b reward, no procgen, no PLR; the confirmation baseline used it too
SEARCH_SEED = 100              # training seed for every search run; confirmation uses fresh seeds 0-2
SAMPLE_SEED = 0                # which configs get sampled
CONFIRM_SEEDS = (0, 1, 2)
CONFIRM_BUDGET = 2_000_000     # must equal the baseline runs' budget
BASELINE_GLOB = "docs/results/prev_action_ablation/prevact_true_s*_eval_final_all.json"
STEPS_PER_SEC = 630            # measured end to end at 12 envs, for the estimate only

# Log-uniform ranges for scale parameters, discrete choices for the rest. Reward, observation, action set and
# training noop_max are fixed on purpose: this tunes the optimizer, not the problem.
SPACE = {
    "lr": ("loguniform", 1e-4, 1e-3),
    "ent_coef": ("loguniform", 3e-3, 5e-2),
    "clip": ("choice", [0.1, 0.2, 0.3]),
    "gamma": ("choice", [0.95, 0.97, 0.99, 0.995]),
    "gae_lambda": ("choice", [0.9, 0.95, 0.98]),
    "epochs": ("choice", [2, 3, 4, 6]),
    "minibatches": ("choice", [4, 6, 8]),
    "rollout_len": ("choice", [128, 256, 512]),
    "vf_coef": ("choice", [0.25, 0.5, 1.0]),
    "plr": ("choice", [False, True]),
}


# ---------------------------------------------------------------- pure logic (unit-tested)
def sample_config(rng: random.Random) -> dict:
    out = {}
    for key, spec in SPACE.items():
        if spec[0] == "loguniform":
            lo, hi = math.log(spec[1]), math.log(spec[2])
            out[key] = float(f"{math.exp(rng.uniform(lo, hi)):.3g}")
        else:
            out[key] = rng.choice(spec[1])
    return out


def make_trials(n: int, sample_seed: int = SAMPLE_SEED) -> list[dict]:
    """Trial 0 is always the unmodified default, so the search can tell whether it beat doing nothing."""
    rng = random.Random(sample_seed)
    trials = [{"name": "t00", "overrides": {}, "note": "default config (no overrides)"}]
    for i in range(1, n):
        trials.append({"name": f"t{i:02d}", "overrides": sample_config(rng)})
    return trials


def budgets(min_budget: int, eta: int, n_rungs: int) -> list[int]:
    return [min_budget * eta ** k for k in range(n_rungs)]


def promote(scores: dict[str, float | None], eta: int) -> list[str]:
    """Top ceil(n/eta) by score. A failed trial (None) always ranks last."""
    ranked = sorted(scores, key=lambda k: (scores[k] is None, -(scores[k] or 0.0)))
    return ranked[:max(1, math.ceil(len(scores) / eta))]


def set_args(overrides: dict) -> list[str]:
    def fmt(v):
        return ("true" if v else "false") if isinstance(v, bool) else str(v)
    return [f"{k}={fmt(v)}" for k, v in overrides.items()]


def total_steps(n: int, eta: int, rung_budgets: list[int]) -> int:
    alive, prev, total = n, 0, 0
    for b in rung_budgets:
        total += alive * (b - prev)
        prev, alive = b, max(1, math.ceil(alive / eta))
    return total


# ---------------------------------------------------------------- orchestration
def _run(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


def train(trial: dict, budget: int, seed: int, run_name: str) -> bool:
    run_dir = ROOT / "runs" / run_name
    latest = run_dir / "latest.pt"
    base = [sys.executable, "-u", "-W", "ignore", "train_ppo.py", "--total-steps", str(budget),
            "--snapshot-every", str(10 ** 9)]
    if latest.exists():  # promoted, or interrupted: continue from where it stopped
        cmd = base + ["--resume", str(latest.relative_to(ROOT))]
    else:
        cmd = base + ["--preset", BASE_PRESET, "--run-name", run_name, "--seed", str(seed)]
        if trial["overrides"]:
            cmd += ["--set", *set_args(trial["overrides"])]
    return _run(cmd, run_dir.parent / f"{Path(run_name).name}_train.log") == 0 and latest.exists()


def evaluate(run_name: str, episodes: int, tag: str, stages: str = "test") -> dict | None:
    run_dir = ROOT / "runs" / run_name
    out = run_dir / f"eval_{tag}.json"
    cmd = [sys.executable, "-W", "ignore", "eval_stages.py", str((run_dir / "latest.pt").relative_to(ROOT)),
           "--stages", stages, "--episodes", str(episodes), "--workers", "12", "--out", str(out.relative_to(ROOT))]
    if _run(cmd, run_dir.parent / f"{Path(run_name).name}_eval.log") != 0 or not out.exists():
        return None
    return json.loads(out.read_text())["test_summary"]


def load_state(path: Path, trials: list[dict], rung_budgets: list[int], eta: int) -> dict:
    if path.exists():
        state = json.loads(path.read_text())
        if state["budgets"] != rung_budgets or state["eta"] != eta or len(state["trials"]) != len(trials):
            raise SystemExit(f"{path} was made with a different plan; move it aside to start a new search")
        return state
    return {"prefix": path.parent.name, "budgets": rung_budgets, "eta": eta, "search_seed": SEARCH_SEED,
            "base_preset": BASE_PRESET,
            "trials": {t["name"]: {**t, "rungs": {}} for t in trials}, "alive": [t["name"] for t in trials],
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def save(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)


def search(state: dict, path: Path, rung_episodes: list[int]) -> None:
    eta = state["eta"]
    for r, budget in enumerate(state["budgets"]):
        key = str(r)
        # A finished rung has already promoted; revisiting it on restart would promote the survivors again and
        # silently cut them by another factor of eta.
        if state.get("completed_rung", -1) >= r:
            continue
        alive = list(state["alive"])
        print(f"\n=== rung {r}: {len(alive)} trials at {budget:,} steps, {rung_episodes[r]} eval eps/stage "
              f"{time.strftime('%H:%M')} ===", flush=True)
        for name in alive:
            t = state["trials"][name]
            if key in t["rungs"]:
                print(f"  [done] {name} {t['rungs'][key]['score']}", flush=True)
                continue
            run_name = f"{state['prefix']}/{name}"
            ok = train(t, budget, SEARCH_SEED, run_name)
            summary = evaluate(run_name, rung_episodes[r], f"r{r}") if ok else None
            score = summary["mean_x_pos"] if summary else None
            t["rungs"][key] = {"budget": budget, "score": score, "summary": summary,
                               "failed": summary is None, "at": time.strftime("%H:%M")}
            save(state, path)
            shown = f"{score:7.1f}" if score is not None else " FAILED"
            print(f"  {name} {shown}  {t['overrides'] or 'default'}", flush=True)
        if r + 1 < len(state["budgets"]):
            scores = {n: state["trials"][n]["rungs"][key]["score"] for n in alive}
            state["alive"] = promote(scores, eta)
            print(f"  promoted: {', '.join(state['alive'])}", flush=True)
        state["completed_rung"] = r  # saved together with the promotion, so a crash cannot split them
        save(state, path)
    last = str(len(state["budgets"]) - 1)
    finals = {n: state["trials"][n]["rungs"][last]["score"] for n in state["alive"]}
    state["winner"] = promote(finals, len(finals) or 1)[0]
    state["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save(state, path)
    print(f"\nsearch winner: {state['winner']}  {state['trials'][state['winner']]['overrides'] or 'default'}")


def confirm(state: dict, path: Path) -> None:
    winner = state.get("winner")
    if not winner:
        raise SystemExit("no winner yet; run the search first")
    trial = state["trials"][winner]
    if not trial["overrides"]:
        print("the default config won the search; there is nothing to confirm against itself")
        return
    results = state.setdefault("confirm", {})
    for seed in CONFIRM_SEEDS:
        if str(seed) in results:
            continue
        run_name = f"{state['prefix']}/confirm_{winner}_s{seed}"
        ok = train(trial, CONFIRM_BUDGET, seed, run_name)
        summary = evaluate(run_name, 10, "final_all", stages="all") if ok else None
        results[str(seed)] = summary["mean_x_pos"] if summary else None
        save(state, path)
        print(f"  confirm {winner} seed {seed}: {results[str(seed)]}", flush=True)
    baseline = [json.loads(Path(p).read_text())["test_summary"]["mean_x_pos"]
                for p in sorted(glob.glob(str(ROOT / BASELINE_GLOB)))]
    report_confirmation(winner, trial, [results[str(s)] for s in CONFIRM_SEEDS], baseline, state, path)


def report_confirmation(winner, trial, ours, base, state, path) -> None:
    ours = [x for x in ours if x is not None]
    if len(ours) < 2 or len(base) < 2:
        print("not enough completed seeds to compare")
        return
    diff = statistics.mean(ours) - statistics.mean(base)
    se = (statistics.variance(ours) / len(ours) + statistics.variance(base) / len(base)) ** 0.5
    t = diff / se if se else float("inf")
    verdict = "beats the default" if t >= 2 else "does NOT clearly beat the default"
    state["confirmation"] = {"winner": winner, "overrides": trial["overrides"], "winner_seeds": ours,
                             "default_seeds": base, "diff": diff, "se": se, "t": t, "verdict": verdict}
    save(state, path)
    print(f"\nwinner {winner} over seeds {CONFIRM_SEEDS}: {[round(x, 1) for x in ours]}  "
          f"mean {statistics.mean(ours):.1f}")
    print(f"default over the same seeds:            {[round(x, 1) for x in base]}  mean {statistics.mean(base):.1f}")
    print(f"difference {diff:+.1f}  SE {se:.1f}  t {t:.2f}  ->  {verdict} (threshold t >= 2)")


def report(state: dict) -> None:
    rows = []
    for name, t in state["trials"].items():
        reached = max((int(k) for k in t["rungs"]), default=-1)
        score = t["rungs"][str(reached)]["score"] if reached >= 0 else None
        rows.append((reached, score if score is not None else -1e9, name, t))
    rows.sort(key=lambda r: (-r[0], -r[1]))
    print(f"{'trial':<5} {'rung':>4} {'held-out x':>10}  overrides")
    for reached, score, name, t in rows:
        shown = f"{score:10.1f}" if score > -1e9 else "    failed"
        print(f"{name:<5} {reached:>4} {shown}  {t['overrides'] or 'default'}")
    if "confirmation" in state:
        c = state["confirmation"]
        print(f"\nconfirmation: {c['winner']} diff {c['diff']:+.1f} SE {c['se']:.1f} t {c['t']:.2f} -> {c['verdict']}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=27)
    p.add_argument("--eta", type=int, default=3)
    p.add_argument("--min-budget", type=int, default=250_000)
    p.add_argument("--rungs", type=int, default=3)
    p.add_argument("--episodes", default="10,15,20", help="eval episodes per stage at each rung")
    p.add_argument("--prefix", default="sha", help="run directory under runs/; state lives there too")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--report", action="store_true")
    args = p.parse_args()

    rung_budgets = budgets(args.min_budget, args.eta, args.rungs)
    rung_episodes = [int(x) for x in args.episodes.split(",")]
    if len(rung_episodes) != args.rungs:
        raise SystemExit("--episodes needs one value per rung")
    trials = make_trials(args.n)
    args.state = ROOT / "runs" / args.prefix / "state.json"

    if args.dry_run:
        steps = total_steps(args.n, args.eta, rung_budgets)
        confirm_steps = len(CONFIRM_SEEDS) * CONFIRM_BUDGET
        print(f"{args.n} trials, eta {args.eta}, rungs at {', '.join(f'{b:,}' for b in rung_budgets)} steps")
        print(f"search: {steps:,} steps ~ {steps / STEPS_PER_SEC / 3600:.1f} h of training + evals")
        print(f"confirmation: {confirm_steps:,} steps ~ {confirm_steps / STEPS_PER_SEC / 3600:.1f} h + evals")
        for t in trials:
            print(f"  {t['name']}  {t['overrides'] or 'default'}")
        return

    state = load_state(args.state, trials, rung_budgets, args.eta)
    if args.report:
        report(state)
    elif args.confirm:
        confirm(state, args.state)
    else:
        search(state, args.state, rung_episodes)
        report(state)


if __name__ == "__main__":
    main()
