"""Replay buffer that stores each 84x84 frame once and rebuilds 4-frame stacks at sample time.

Memory: 100k rows * 7,056 bytes = ~706 MB, vs 5.6 GB for naive uint8 (state, next_state) pairs.

Layout: one row per observation. Row i holds the newest frame of s_t plus (a_t, r_t, terminated_t)
for the transition leaving s_t. s_{t+1} is row i+1. The final observation of an episode gets its
own row with valid=False: it is never a transition start, but it is the next-state for the last
transition (needed when that transition was truncated and we bootstrap from it).

Stacks are rebuilt from rows i-3..i, repeating the episode's first frame when the window crosses an
episode start. That matches MarioEnv.reset(), which fills the stack with copies of the first frame.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class FrameReplayBuffer:
    def __init__(self, capacity: int, frame_shape: tuple[int, int], stack: int, seed: int = 0):
        self.capacity = capacity
        self.stack = stack
        self.frames = np.zeros((capacity, *frame_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.bool_)
        self.valid = np.zeros(capacity, dtype=np.bool_)  # row starts a sampleable transition
        self.start = np.zeros(capacity, dtype=np.bool_)  # row is the first obs of an episode
        self.head = 0  # next row to write
        self.size = 0
        self.need_start = True  # next add() begins a new episode
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Number of sampleable transitions (approximate upper bound; placeholders excluded at sample time)."""
        return self.size

    def add(self, state, action, reward, next_state, terminated: bool, truncated: bool):
        if self.need_start:
            self._write_row(state[-1], start=True)
        else:
            prev = (self.head - 1) % self.capacity
            if not np.array_equal(self.frames[prev], state[-1]):
                raise ValueError("state does not continue the previous next_state; call add() once per env step in order")
        cur = (self.head - 1) % self.capacity
        self.actions[cur] = action
        self.rewards[cur] = reward
        self.terminated[cur] = terminated
        self.valid[cur] = True
        self._write_row(next_state[-1], start=False)
        self.need_start = terminated or truncated

    def end_episode(self):
        """Drop the pending episode link, e.g. when training stops mid-episode."""
        self.need_start = True

    def sample(self, batch_size: int):
        idx = self._sample_indices(batch_size)
        state = self.frames[self._stack_rows(idx)]
        next_state = self.frames[self._stack_rows((idx + 1) % self.capacity)]
        return (
            torch.from_numpy(state),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.rewards[idx]),
            torch.from_numpy(next_state),
            torch.from_numpy(self.terminated[idx].astype(np.float32)),
        )

    def _write_row(self, frame, start: bool):
        h = self.head
        self.frames[h] = frame
        self.start[h] = start
        self.valid[h] = False
        self.terminated[h] = False
        self.head = (h + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _sample_indices(self, batch_size: int) -> np.ndarray:
        # Age order k: 0 = oldest row. Stack needs rows k-(stack-1)..k+1 to be contiguous writes,
        # so k in [stack-1, size-2]. When full, the oldest row sits at head.
        lo, hi = self.stack - 1, self.size - 2
        if hi < lo:
            raise ValueError(f"buffer too small to sample: size={self.size}")
        oldest = self.head if self.size == self.capacity else 0
        out = np.empty(0, dtype=np.int64)
        for _ in range(100):
            k = self.rng.integers(lo, hi + 1, size=2 * batch_size)
            idx = (oldest + k) % self.capacity
            out = np.concatenate([out, idx[self.valid[idx]]])
            if len(out) >= batch_size:
                return out[:batch_size]
        raise RuntimeError("could not find enough valid transitions; is the buffer mostly placeholders?")

    def _stack_rows(self, idx: np.ndarray) -> np.ndarray:
        offsets = np.arange(-(self.stack - 1), 1)
        rows = (idx[:, None] + offsets) % self.capacity  # [B, stack], last column = idx
        starts = self.start[rows]
        # Latest episode start in columns 1..stack-1 cuts off everything before it.
        rev = starts[:, :0:-1]  # columns stack-1 .. 1
        has = rev.any(axis=1)
        latest = np.where(has, (self.stack - 1) - rev.argmax(axis=1), 0)
        cols = np.maximum(np.arange(self.stack)[None, :], latest[:, None])
        return np.take_along_axis(rows, cols, axis=1)

    # --- persistence ---
    def save(self, path: str | Path):
        np.savez(
            path,
            frames=self.frames[: self.size] if self.size < self.capacity else self.frames,
            actions=self.actions, rewards=self.rewards, terminated=self.terminated,
            valid=self.valid, start=self.start,
            meta=np.array([self.head, self.size, int(self.need_start), self.capacity]),
        )

    def load(self, path: str | Path):
        d = np.load(path)
        head, size, need_start, capacity = (int(x) for x in d["meta"])
        if capacity != self.capacity:
            raise ValueError(f"buffer capacity mismatch: saved {capacity}, configured {self.capacity}")
        self.frames[: len(d["frames"])] = d["frames"]
        for name in ("actions", "rewards", "terminated", "valid", "start"):
            getattr(self, name)[:] = d[name]
        self.head, self.size = head, size
        # A resumed run starts a fresh episode, so never link to the saved pending row.
        self.need_start = True
