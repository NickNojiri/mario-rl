"""TileMarioEnv against the real emulator."""
import numpy as np

from env.tiles import (ALL_STAGES, EXCLUDED_STAGES, ENEMY_BASE, MARIO_ID, TEST_STAGES, TRAIN_STAGES, TileMarioEnv,
                       n_extras)

RIGHT, RIGHT_A = 1, 2  # SIMPLE_MOVEMENT


def test_stage_split_is_clean():
    assert not set(TRAIN_STAGES) & set(TEST_STAGES)
    assert not (set(TRAIN_STAGES) | set(TEST_STAGES)) & set(EXCLUDED_STAGES)
    assert len(TRAIN_STAGES) + len(TEST_STAGES) + len(EXCLUDED_STAGES) == len(ALL_STAGES) == 32


def test_obs_shapes_and_start_state():
    env = TileMarioEnv(stages=["1-1"], noop_max=0)
    obs, info = env.reset(seed=0)
    assert info["stage"] == "1-1"
    assert obs["tiles"].shape == (4, 13, 16) and obs["tiles"].dtype == np.int16
    assert obs["extras"].shape == (n_extras(env.n_actions),) and obs["extras"].dtype == np.float32
    grid = obs["tiles"][-1]
    assert (grid[11:] == 0x54).all(), "1-1 starts on two full rows of ground"
    assert (grid == MARIO_ID).sum() == 1
    r, _ = np.argwhere(grid == MARIO_ID)[0]
    assert r == 10, "standing Mario is directly above the ground rows"
    assert obs["extras"][: env.n_actions].sum() == 0, "no previous action at reset"
    env.close()


def test_previous_action_is_observable():
    env = TileMarioEnv(stages=["1-1"], noop_max=0)
    env.reset(seed=0)
    obs, *_ = env.step(RIGHT_A)
    assert obs["extras"][RIGHT_A] == 1.0 and obs["extras"][: env.n_actions].sum() == 1.0
    env.close()


def test_goomba_appears_as_enemy():
    env = TileMarioEnv(stages=["1-1"], noop_max=0, no_progress_steps=10_000)
    env.reset(seed=0)
    seen = False
    for _ in range(40):
        obs, _, terminated, truncated, _ = env.step(RIGHT)
        seen |= bool((obs["tiles"][-1] == ENEMY_BASE + 6).any())  # enemy type 6 = goomba
        if terminated or truncated:
            break
    assert seen
    env.close()


def test_stage_sampling_is_seeded_and_restricted():
    env = TileMarioEnv(stages=["1-1", "1-2", "1-3"], noop_max=0)
    a = [env.reset(seed=s)[1]["stage"] for s in range(12)]
    b = [env.reset(seed=s)[1]["stage"] for s in range(12)]
    assert a == b and set(a) <= {"1-1", "1-2", "1-3"} and len(set(a)) > 1
    env.close()


def test_episode_summary_and_flag_free_death():
    env = TileMarioEnv(stages=["1-1"], noop_max=0, no_progress_steps=10_000, coin_reward=15.0, flag_reward=150.0)
    env.reset(seed=0)
    for _ in range(3000):
        _, reward, terminated, truncated, info = env.step(RIGHT)  # walks into the first goomba
        if terminated or truncated:
            break
    ep = info["episode"]
    assert terminated and not ep["flag_get"] and ep["length"] > 0
    assert abs(ep["reward"] - (ep["game_reward"] + 15.0 * ep["coins"])) < 1e-6
    env.close()


def test_coin_bonus_is_paid_for_a_real_coin():
    """Scripted: run right, jump at step 31 for 3 steps -> hits 1-1's first ?-block (found by search)."""
    NOOP = 0
    rewards = {}
    for coin_reward in (0.0, 15.0):
        env = TileMarioEnv(stages=["1-1"], noop_max=0, no_progress_steps=10_000, coin_reward=coin_reward)
        env.reset(seed=0)
        coins, total = 0, 0.0
        for t in range(42):
            action = RIGHT_A if 31 <= t < 34 else (NOOP if t >= 34 else RIGHT)
            _, reward, terminated, truncated, info = env.step(action)
            coins += info["coin_delta"]
            total += reward
            if terminated or truncated:  # this script walks into a goomba on its final step
                break
        rewards[coin_reward] = total
        assert coins == 1
        env.close()
    assert abs((rewards[15.0] - rewards[0.0]) - 15.0) < 1e-6, "same trajectory; only the coin bonus differs"
