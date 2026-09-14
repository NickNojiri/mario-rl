"""The only place that knows about gym 0.26 / nes-py quirks.

Everything outside this file sees a Gymnasium-style env:
    reset(seed=None) -> (obs, info)
    step(action)     -> (obs, reward, terminated, truncated, info)
with obs a uint8 array of shape [stack, frame_size, frame_size].

Quirks handled here (verified with scripts/probe_env.py):
  - nes-py's JoypadSpace.reset() rejects seed/options kwargs, so seeding is ours.
  - The emulator is deterministic; the only per-episode randomness is the NOOP start count.
  - Timer expiry kills Mario -> terminated=True. gym's TimeLimit is registered at 9,999,999
    steps, so truncated is never set by the base env. The only truncation is our no-progress cutoff.
  - The per-frame reward is clipped to [-15, 15]. We return the raw skip-frame sum; scaling is
    the agent's job so eval can report raw reward.
"""
from __future__ import annotations

import warnings
from collections import deque

import cv2
import gym_super_mario_bros
import numpy as np
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

ACTION_SETS = {"right_only": RIGHT_ONLY, "simple": SIMPLE_MOVEMENT}
_NES_NOOP = 0  # raw NES controller byte with no buttons pressed

warnings.filterwarnings("ignore", module=r"gym\.envs\.registration")


class MarioEnv:
    def __init__(
        self,
        world: int = 1,
        stage: int = 1,
        render_mode: str | None = None,
        actions: str = "right_only",
        skip: int = 4,
        frame_size: int = 84,
        stack: int = 4,
        no_progress_steps: int = 150,
        noop_max: int = 30,
    ):
        base = gym_super_mario_bros.make(
            f"SuperMarioBros-{world}-{stage}-v0",
            apply_api_compatibility=True,
            render_mode=render_mode,
            disable_env_checker=True,
        )
        self._joypad = JoypadSpace(base, ACTION_SETS[actions])
        self._base = base
        self.n_actions = self._joypad.action_space.n
        self.obs_shape = (stack, frame_size, frame_size)
        self._skip = skip
        self._frame_size = frame_size
        self._frames: deque[np.ndarray] = deque(maxlen=stack)
        self._no_progress_steps = no_progress_steps
        self._noop_max = noop_max
        self._rng = np.random.default_rng()
        self._best_x: int | None = None
        self._steps_since_progress = 0

    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        frame, info = self._joypad.reset()
        for _ in range(int(self._rng.integers(0, self._noop_max + 1))):
            frame, _, terminated, truncated, info = self._raw_skip(_NES_NOOP, use_joypad=False)
            if terminated or truncated:  # can't happen within 30 NOOPs on any stage start, but stay honest
                frame, info = self._joypad.reset()
        processed = self._process(frame)
        for _ in range(self._frames.maxlen):
            self._frames.append(processed)
        self._best_x = info.get("x_pos")  # None when noop_max=0: EnvCompatibility.reset() returns info={}
        self._steps_since_progress = 0
        return self._obs(), info

    def step(self, action: int):
        frame, reward, terminated, truncated, info = self._raw_skip(int(action), use_joypad=True)
        self._frames.append(self._process(frame))

        x = info["x_pos"]
        if self._best_x is not None and x > self._best_x:
            self._steps_since_progress = 0
        else:
            self._steps_since_progress += 1
        self._best_x = x if self._best_x is None else max(self._best_x, x)
        if not terminated and self._steps_since_progress >= self._no_progress_steps:
            truncated = True
            info["truncated_reason"] = "no_progress"
        return self._obs(), reward, terminated, truncated, info

    def close(self):
        self._joypad.close()

    def _raw_skip(self, action: int, use_joypad: bool):
        env = self._joypad if use_joypad else self._base
        total = 0.0
        for _ in range(self._skip):
            frame, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        return frame, total, bool(terminated), bool(truncated), info

    def _process(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (self._frame_size, self._frame_size), interpolation=cv2.INTER_AREA)

    def _obs(self) -> np.ndarray:
        return np.stack(self._frames)


def make_mario(world: int = 1, stage: int = 1, render_mode: str | None = None, **kwargs) -> MarioEnv:
    return MarioEnv(world, stage, render_mode, **kwargs)


def make_from_config(cfg, render_mode: str | None = None, **overrides) -> MarioEnv:
    kw = dict(
        world=cfg.world,
        stage=cfg.stage,
        actions=cfg.actions,
        skip=cfg.skip,
        frame_size=cfg.frame_size,
        stack=cfg.stack,
        no_progress_steps=cfg.no_progress_steps,
        noop_max=cfg.noop_max,
    )
    kw.update(overrides)
    return MarioEnv(render_mode=render_mode, **kw)
