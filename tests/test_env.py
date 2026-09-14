"""Runs the real emulator. A few seconds total."""
import numpy as np

from env.make_env import make_mario

NOOP, RIGHT = 0, 1  # indices in RIGHT_ONLY


def test_api_shapes_and_dtypes():
    env = make_mario(1, 1, noop_max=0)
    out = env.reset(seed=0)
    assert isinstance(out, tuple) and len(out) == 2
    obs, info = out
    assert obs.shape == (4, 84, 84) and obs.dtype == np.uint8
    assert all(np.array_equal(obs[0], f) for f in obs)  # reset fills stack with first frame
    step = env.step(RIGHT)
    assert len(step) == 5 and step[0].shape == (4, 84, 84) and step[0].dtype == np.uint8
    assert isinstance(step[2], bool) and isinstance(step[3], bool)
    env.close()


def test_no_progress_truncates_not_terminates():
    env = make_mario(1, 1, noop_max=0, no_progress_steps=20)
    env.reset(seed=0)
    for t in range(1, 100):
        _, _, terminated, truncated, info = env.step(NOOP)
        if terminated or truncated:
            break
    assert (t, terminated, truncated, info.get("truncated_reason")) == (20, False, True, "no_progress")
    env.close()


def test_death_is_terminal():
    env = make_mario(1, 1, noop_max=0, no_progress_steps=10_000)
    env.reset(seed=0)
    for _ in range(3000):
        _, _, terminated, truncated, info = env.step(RIGHT)  # walks into the first goomba
        if terminated or truncated:
            break
    assert terminated and not truncated and not info["flag_get"]
    env.close()


def test_noop_start_seeding_is_reproducible_and_varies():
    env = make_mario(1, 1, noop_max=30)
    first = [env.reset(seed=s)[0] for s in (3, 3)]
    assert np.array_equal(*first)
    distinct = {env.reset(seed=s)[0].tobytes() for s in range(10)}
    assert len(distinct) > 1
    env.close()
