"""v3 observation, reward and episode diagnostics against the real emulator."""
import json
from pathlib import Path

import numpy as np
import pytest

from env.tiles import EMPTY_ENEMY_ID, N_ENEMY_FEATS, N_ENEMY_SLOTS, REWARD_COMPONENTS, TileMarioEnv, n_extras

NOOP, RIGHT, RIGHT_A = 0, 1, 2  # SIMPLE_MOVEMENT
LEFT = 6


def make(**kw):
    base = dict(stages=["1-1"], noop_max=0, no_progress_steps=10_000, obs_version=3, reward_version=3)
    return TileMarioEnv(**{**base, **kw})


def test_v3_obs_shapes():
    env = make()
    obs, _ = env.reset(seed=0)
    assert obs["extras"].shape == (n_extras(env.n_actions, 3),)
    assert obs["enemy_ids"].shape == (N_ENEMY_SLOTS,) and obs["enemy_states"].shape == (N_ENEMY_SLOTS,)
    assert obs["enemies"].shape == (N_ENEMY_SLOTS, N_ENEMY_FEATS) and obs["enemies"].dtype == np.float32
    assert (obs["enemy_ids"] == EMPTY_ENEMY_ID).all(), "no enemies at the start of 1-1"
    env.close()


def test_first_goomba_is_ahead_and_walking_left():
    env = make()
    env.reset(seed=0)
    for _ in range(30):
        obs, *_ = env.step(RIGHT)
        if obs["enemy_ids"][0] == 0x06:
            break
    assert obs["enemy_ids"][0] == 0x06, "goomba in the nearest slot"
    active, dx, dy, vx, vy, on_screen = obs["enemies"][0]
    assert active == 1 and on_screen == 1
    assert dx > 0, "goomba is to Mario's right"
    assert vx < 0, "goomba walks left (RAM x speed -10, verified by probe_enemies)"
    assert abs(dy) < 0.1, "same ground level"
    env.close()


def test_progress_rewards_only_new_ground():
    env = make()
    env.reset(seed=0)
    for _ in range(10):
        env.step(RIGHT)
    # Pressing LEFT after running first skids Mario forward (real SMB momentum), so check only after that.
    back = [env.step(LEFT)[4] for _ in range(16)]
    assert back[-1]["x_pos"] < max(i["x_pos"] for i in back), "Mario is really moving left by the end"
    assert all(i["reward_components"]["progress"] == 0 for i in back[-6:]), "walking back earns no progress"
    forward = [env.step(RIGHT)[4]["reward_components"]["progress"] for _ in range(20)]
    assert sum(forward) > 0, "ground beyond the farthest point pays again"
    env.close()


def test_death_reward_and_cause_on_goomba():
    env = make(death_reward=-150.0)
    env.reset(seed=0)
    for _ in range(3000):
        _, reward, terminated, truncated, info = env.step(RIGHT)
        if terminated or truncated:
            break
    ep = info["episode"]
    assert terminated and ep["death_cause"] == "enemy_0x06"
    assert info["reward_components"]["death"] == -150.0
    assert abs(ep["reward"] - sum(ep[f"r_{k}"] for k in REWARD_COMPONENTS)) < 1e-6
    env.close()


def test_pit_death_is_classified():
    env = make(stages=["4-2"])
    env.reset(seed=0, stage="4-2")
    for _ in range(300):
        _, _, terminated, truncated, info = env.step(RIGHT)  # 4-2 opens with pits
        if terminated or truncated:
            break
    assert info["episode"]["death_cause"] == "pit"
    env.close()


def test_points_and_coin_are_separated():
    """The scripted coin-block hit from test_tiles: 1 coin (200 points) must not count as stomp points."""
    env = make()
    env.reset(seed=0)
    for t in range(42):
        action = RIGHT_A if 31 <= t < 34 else (NOOP if t >= 34 else RIGHT)
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    ep = info.get("episode") or env._ep
    assert ep["coins"] == 1 and ep["points"] == 0 and ep["r_points"] == 0.0
    env.close()


def test_points_reward_is_capped():
    env = make(points_per_100=1000.0, points_cap=150.0)
    env.reset(seed=0)
    env._score -= 5000  # pretend 5000 points were just scored
    _, _, _, _, info = env.step(NOOP)
    assert info["reward_components"]["points"] == 150.0
    env._score -= 5000
    _, _, _, _, info = env.step(NOOP)
    assert info["reward_components"]["points"] == 0.0, "cap is per episode"
    env.close()


def test_v2_reward_unchanged():
    """reward_version 2 must still pay game reward + coins + flag, whatever the v3 settings are."""
    env = TileMarioEnv(stages=["1-1"], noop_max=0, no_progress_steps=10_000, death_reward=-999.0)
    env.reset(seed=0)
    for _ in range(3000):
        _, reward, terminated, truncated, info = env.step(RIGHT)
        assert abs(reward - (info["game_reward"] + info["reward_components"]["coins"])) < 1e-6
        if terminated or truncated:
            break
    assert "enemy_ids" not in env._obs()
    env.close()


@pytest.mark.skipif(not Path("runs/ppo_1h_a/latest_1h_frozen.pt").exists(), reason="local run artifact")
def test_v2_checkpoint_still_loads_and_acts():
    import torch

    from agent.ppo import PPOAgent
    from config import PPOConfig

    ckpt = torch.load("runs/ppo_1h_a/latest_1h_frozen.pt", map_location="cpu", weights_only=False)
    cfg = PPOConfig.from_dict(ckpt["config"])
    assert cfg.learning_hash() == ckpt["learning_hash"]
    env = TileMarioEnv(**cfg.env_kwargs(["1-1"]))
    agent = PPOAgent(cfg, env.n_actions, env.n_extras)
    agent.net.load_state_dict(ckpt["model"])
    obs, _ = env.reset(seed=0)
    probs, _ = agent.probs(obs)
    assert abs(float(probs.sum()) - 1) < 1e-5
    env.close()
