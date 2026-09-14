"""FrameReplayBuffer must return exactly what naive (state, next_state) storage would have."""
from collections import deque

import numpy as np
import pytest

from agent.buffer import FrameReplayBuffer

STACK = 4


def frame(fid: int) -> np.ndarray:
    f = np.zeros((2, 2), dtype=np.uint8)
    f[0, 0], f[0, 1] = fid % 256, fid // 256
    return f


def ids(stack: np.ndarray) -> tuple:
    return tuple(int(f[0, 0]) + 256 * int(f[0, 1]) for f in stack)


def run_episodes(buf, lengths, rng):
    """Feed episodes like the env would. Returns {newest-frame id of state: expected transition} in write order."""
    expected, fid = {}, 0
    for n, length in enumerate(lengths):
        frames = deque([frame(fid)] * STACK, maxlen=STACK)
        fid += 1
        state = np.stack(frames)
        for t in range(length):
            frames.append(frame(fid))
            fid += 1
            next_state = np.stack(frames)
            last = t == length - 1
            terminated = last and n % 2 == 0
            truncated = last and not terminated
            a, r = int(rng.integers(5)), float(rng.normal())
            buf.add(state, a, r, next_state, terminated, truncated)
            expected[ids(state)[-1]] = (ids(state), a, np.float32(r), ids(next_state), float(terminated))
            state = next_state
    return expected


@pytest.mark.parametrize("capacity", [10_000, 37])  # no wrap, and heavy wraparound
def test_samples_match_naive_storage(capacity):
    rng = np.random.default_rng(0)
    lengths = [1, 2, 3, 1, 50, 4, 5, 1, 1, 30, 7, 2, 60]
    buf = FrameReplayBuffer(capacity, (2, 2), STACK, seed=1)
    expected = run_episodes(buf, lengths, rng)

    seen = set()
    for _ in range(200):
        s, a, r, ns, term = buf.sample(16)
        for i in range(16):
            key = ids(s[i].numpy())[-1]
            assert key in expected, "sampled a placeholder or overwritten row"
            exp_s, exp_a, exp_r, exp_ns, exp_term = expected[key]
            assert ids(s[i].numpy()) == exp_s
            assert ids(ns[i].numpy()) == exp_ns
            assert int(a[i]) == exp_a and float(r[i]) == exp_r and float(term[i]) == exp_term
            seen.add(key)

    if capacity == 10_000:
        # Only rows k < STACK-1 (the first three transitions ever written) are excluded.
        assert len(seen) >= len(expected) - (STACK - 1)
    else:
        assert len(seen) > capacity // 2


def test_rejects_out_of_order_add():
    buf = FrameReplayBuffer(100, (2, 2), STACK)
    s0 = np.stack([frame(0)] * STACK)
    s1 = np.stack([frame(0)] * 3 + [frame(1)])
    buf.add(s0, 0, 0.0, s1, False, False)
    with pytest.raises(ValueError):
        buf.add(s0, 0, 0.0, s1, False, False)


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    buf = FrameReplayBuffer(64, (2, 2), STACK, seed=3)
    run_episodes(buf, [10, 20, 50], rng)
    buf.save(tmp_path / "b.npz")
    other = FrameReplayBuffer(64, (2, 2), STACK, seed=3)
    other.load(tmp_path / "b.npz")
    for name in ("frames", "actions", "rewards", "terminated", "valid", "start"):
        np.testing.assert_array_equal(getattr(buf, name), getattr(other, name))
    assert (buf.head, buf.size) == (other.head, other.size) and other.need_start
