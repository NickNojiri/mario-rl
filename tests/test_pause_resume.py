"""A week-long run has to survive being paused, stopped, restarted and crashed without losing learning state."""
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

from agent.ppo import PPOAgent
from config import PPOConfig, get_ppo_preset
from env.tiles import n_extras
from train_ppo import PAUSE_FILE, plr_weights, wait_while_paused

ROOT = Path(__file__).resolve().parent.parent


def test_no_pause_file_means_no_wait(tmp_path):
    calls = []
    assert wait_while_paused(tmp_path, lambda: calls.append(1), poll=0.01) == 0.0
    assert calls == []


def test_pause_checkpoints_once_then_waits_until_the_file_is_removed(tmp_path):
    flag = tmp_path / PAUSE_FILE
    flag.touch()
    calls = []
    threading.Timer(0.3, flag.unlink).start()
    paused = wait_while_paused(tmp_path, lambda: calls.append(1), poll=0.02)
    assert calls == [1], "exactly one checkpoint per pause"
    assert paused >= 0.25


def test_checkpoint_carries_trainer_state_and_is_written_atomically(tmp_path):
    cfg = PPOConfig()
    agent = PPOAgent(cfg, 7, n_extras(7))
    path = tmp_path / "latest.pt"
    agent.save(path, {"plr_score": {"1-1": 0.25}})
    assert not list(tmp_path.glob("*.tmp")), "no temporary file may be left behind"
    other = PPOAgent(cfg, 7, n_extras(7))
    other.load(path)
    assert other.extra_state == {"plr_score": {"1-1": 0.25}}


def test_old_checkpoints_without_trainer_state_still_load(tmp_path):
    cfg = PPOConfig()
    agent = PPOAgent(cfg, 7, n_extras(7))
    ckpt = {"model": agent.net.state_dict(), "optimizer": agent.optimizer.state_dict(), "global_step": 5,
            "updates": 1, "config": cfg.to_dict(), "learning_hash": cfg.learning_hash(),
            "rng": {"python": __import__("random").getstate(), "numpy": __import__("numpy").random.get_state(),
                    "torch": torch.get_rng_state()}}
    torch.save(ckpt, tmp_path / "old.pt")
    agent.load(tmp_path / "old.pt")
    assert agent.extra_state == {} and agent.global_step == 5


def test_plr_weights_are_a_distribution_favouring_high_scores():
    cfg = get_ppo_preset("ppo_gen2")
    stages = ["a", "b", "c"]
    w = plr_weights({"a": 0.1, "b": 5.0, "c": 1.0}, stages, cfg)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["b"] > w["c"] > w["a"]


def _train(*args, timeout=300):
    return subprocess.Popen([sys.executable, "-W", "ignore", "train_ppo.py", *args], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def test_a_real_run_pauses_resumes_stops_and_continues(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / PAUSE_FILE).touch()  # paused from the very first update
    proc = _train("--preset", "ppo_smoke", "--run-name", str(run), "--set", "plr=true")
    deadline = time.time() + 120
    while not (run / "latest.pt").exists() and time.time() < deadline:
        time.sleep(0.5)
    assert (run / "latest.pt").exists(), "pausing must write a checkpoint before idling"
    time.sleep(2)
    assert proc.poll() is None, "while paused the run must stay alive, not exit"
    (run / PAUSE_FILE).unlink()
    out = proc.communicate(timeout=300)[0]
    assert proc.returncode == 0, out[-2000:]
    assert "[pause]" in out and "[resume]" in out
    first = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    assert first["global_step"] >= 2048 and first["extra"]["plr_score"], "PLR scores must be saved"

    # stop/restart: resume the finished run for more steps; PLR scores must come back, not reset to 1.0
    proc = _train("--resume", str(run / "latest.pt"), "--total-steps", "4096")
    out = proc.communicate(timeout=300)[0]
    assert proc.returncode == 0, out[-2000:]
    assert f"resumed step={first['global_step']}" in out
    second = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    assert second["global_step"] >= 4096
    assert set(second["extra"]["plr_score"]) == set(first["extra"]["plr_score"])
