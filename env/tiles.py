"""Tile-grid observations read from Super Mario Bros RAM, and a multi-stage env built on them.

Why tiles instead of pixels: every real level (and any future procedural generator) can be described with
the same small grid of tile ids, so skill learned on one layout is expressed in the same vocabulary as every
other layout. Pixels carry per-level colors and scenery the agent would otherwise have to learn to ignore.

Observation (dict):
  tiles  int16 [stack, 13, 16]  screen-aligned grid. 0-255 = the game's metatile id, 256+t = enemy of type t,
                                320 = Mario. Ids are embedded by the network, so no hand-made solidity table.
  extras float32 [n_extra]      previous action one-hot (makes "A is held" observable), Mario x/y speed,
                                float state one-hot (ground/jump/fall/flagpole), powerup one-hot.

Only information a player can see or already knows (layout, enemies, own motion, own last button) is used.

RAM addresses confirmed against gym_super_mario_bros/smb_env.py where it uses them (0x6D, 0x86, 0x71C,
0xB5, 0x756, 0x1D, 0x16-0x1A); the others (0x500 metatile buffer, 0x71A, enemy position tables) are from
the SMB disassembly and are verified visually by scripts/render_tiles.py.
"""
from __future__ import annotations

import warnings

import gym_super_mario_bros
import numpy as np
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

warnings.filterwarnings("ignore", module=r"gym\.envs\.registration")

ROWS, COLS = 13, 16
ENEMY_BASE = 256
N_ENEMY_TYPES = 64
MARIO_ID = ENEMY_BASE + N_ENEMY_TYPES
VOCAB = MARIO_ID + 1
ACTION_SETS = {"right_only": RIGHT_ONLY, "simple": SIMPLE_MOVEMENT, "complex": COMPLEX_MOVEMENT}
N_FLOAT_STATES, N_POWERUPS = 4, 3
_NES_NOOP = 0

ALL_STAGES = [f"{w}-{s}" for w in range(1, 9) for s in range(1, 5)]
# Water physics (2-2, 7-2) and maze castles that loop unless a hidden path is taken (4-4, 7-4, 8-4).
EXCLUDED_STAGES = ["2-2", "7-2", "4-4", "7-4", "8-4"]
# Held out from training: one of each level type, spread across worlds.
TEST_STAGES = ["2-1", "3-3", "4-2", "5-4", "7-1"]
TRAIN_STAGES = [s for s in ALL_STAGES if s not in EXCLUDED_STAGES and s not in TEST_STAGES]


def n_extras(n_actions: int) -> int:
    return n_actions + 2 + N_FLOAT_STATES + N_POWERUPS


def _s8(v) -> int:
    v = int(v)
    return v - 256 if v > 127 else v


def screen_left_x(ram) -> int:
    return int(ram[0x071A]) * 256 + int(ram[0x071C])


def read_tile_grid(ram) -> np.ndarray:
    """13x16 grid of what is on screen right now."""
    left = screen_left_x(ram)
    level_x = left + np.arange(COLS) * 16 + 8  # sample each column at its center
    page = (level_x // 256) % 2
    col = (level_x % 256) // 16
    addr = 0x500 + page[None, :] * 208 + np.arange(ROWS)[:, None] * 16 + col[None, :]
    grid = ram[addr].astype(np.int16)

    for i in range(5):
        if ram[0x0F + i] == 0 or ram[0xB6 + i] != 1:  # inactive or not on the visible screen
            continue
        sx = int(ram[0x6E + i]) * 256 + int(ram[0x87 + i]) - left
        r, c = (int(ram[0xCF + i]) + 8 - 32) // 16, (sx + 8) // 16
        if 0 <= r < ROWS and 0 <= c < COLS:
            grid[r, c] = ENEMY_BASE + min(int(ram[0x16 + i]), N_ENEMY_TYPES - 1)

    if ram[0xB5] == 1:
        sx = int(ram[0x6D]) * 256 + int(ram[0x86]) - left
        r, c = (int(ram[0xCE]) + 8 - 32) // 16, (sx + 8) // 16
        if 0 <= r < ROWS and 0 <= c < COLS:
            grid[r, c] = MARIO_ID
    return grid


def read_extras(ram, prev_action: int, n_actions: int) -> np.ndarray:
    out = np.zeros(n_extras(n_actions), dtype=np.float32)
    if prev_action >= 0:
        out[prev_action] = 1.0
    out[n_actions] = _s8(ram[0x57]) / 40.0  # horizontal speed
    out[n_actions + 1] = _s8(ram[0x9F]) / 8.0  # vertical speed
    out[n_actions + 2 + min(int(ram[0x1D]), N_FLOAT_STATES - 1)] = 1.0
    out[n_actions + 2 + N_FLOAT_STATES + min(int(ram[0x756]), N_POWERUPS - 1)] = 1.0
    return out


class TileMarioEnv:
    """Gymnasium-style env over a set of real SMB stages; a stage is sampled at every reset.

    reward = game reward (progress - clock - death) + coin_reward * coins + flag_reward * reached_flag,
    all in raw game units. info["game_reward"] keeps the unshaped part so eval can report it separately.
    """

    def __init__(
        self,
        stages=TRAIN_STAGES,
        actions: str = "simple",
        skip: int = 4,
        stack: int = 4,
        no_progress_steps: int = 200,
        noop_max: int = 30,
        coin_reward: float = 15.0,
        flag_reward: float = 150.0,
        render_mode: str | None = None,
    ):
        self.stages = list(stages)
        self.action_set = ACTION_SETS[actions]
        self.n_actions = len(self.action_set)
        self.n_extras = n_extras(self.n_actions)
        self._skip, self._stack = skip, stack
        self._no_progress_steps, self._noop_max = no_progress_steps, noop_max
        self._coin_reward, self._flag_reward = coin_reward, flag_reward
        self._render_mode = render_mode
        self._envs: dict[str, tuple] = {}
        self._rng = np.random.default_rng()
        self.stage = None

    # --- env construction (lazy: one emulator per stage actually visited) ---
    def _get(self, stage: str):
        if stage not in self._envs:
            base = gym_super_mario_bros.make(
                f"SuperMarioBros-{stage}-v0", apply_api_compatibility=True,
                render_mode=self._render_mode, disable_env_checker=True,
            )
            self._envs[stage] = (JoypadSpace(base, self.action_set), base)
        return self._envs[stage]

    @property
    def ram(self):
        return self._base.unwrapped.ram

    @property
    def screen(self) -> np.ndarray:
        return self._base.unwrapped.screen

    def reset(self, seed: int | None = None, stage: str | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.stage = stage or self.stages[int(self._rng.integers(len(self.stages)))]
        self._joypad, self._base = self._get(self.stage)
        _, info = self._joypad.reset()
        for _ in range(int(self._rng.integers(0, self._noop_max + 1))):
            _, terminated, truncated, info = self._raw_skip(_NES_NOOP, use_joypad=False)
            if terminated or truncated:
                _, info = self._joypad.reset()

        grid = read_tile_grid(self.ram)
        self._grids = [grid] * self._stack
        self._prev_action = -1
        self._coins = int(self._base.unwrapped._coins)  # same source as info["coins"]
        self._best_x = None
        self._since_progress = 0
        self._ep = {"stage": self.stage, "reward": 0.0, "game_reward": 0.0, "length": 0, "coins": 0, "x_pos": 0,
                    "flag_get": False}
        return self._obs(), {"stage": self.stage}

    def step(self, action: int):
        action = int(action)
        game_reward, terminated, truncated, info = self._raw_skip(action, use_joypad=True)

        coins_now = int(info["coins"])
        coin_delta = (coins_now - self._coins) % 100  # counter wraps at 100 (1-up)
        self._coins = coins_now
        flag = bool(info["flag_get"])
        reward = game_reward + self._coin_reward * coin_delta + (self._flag_reward if flag else 0.0)

        x = int(info["x_pos"])
        if self._best_x is not None and x > self._best_x:
            self._since_progress = 0
        else:
            self._since_progress += 1
        self._best_x = x if self._best_x is None else max(self._best_x, x)
        if not terminated and self._since_progress >= self._no_progress_steps:
            truncated = True
            info["truncated_reason"] = "no_progress"

        self._grids = self._grids[1:] + [read_tile_grid(self.ram)]
        self._prev_action = action
        ep = self._ep
        ep["reward"] += reward
        ep["game_reward"] += game_reward
        ep["length"] += 1
        ep["coins"] += coin_delta
        ep["x_pos"] = x
        ep["flag_get"] = flag
        info.update(stage=self.stage, game_reward=game_reward, coin_delta=coin_delta)
        if terminated or truncated:
            info["episode"] = dict(ep, terminated=terminated, truncated=truncated)
        return self._obs(), reward, terminated, truncated, info

    def close(self):
        for joypad, _ in self._envs.values():
            joypad.close()
        self._envs.clear()

    def _raw_skip(self, action: int, use_joypad: bool):
        env = self._joypad if use_joypad else self._base
        total, terminated, truncated, info = 0.0, False, False, {}
        for _ in range(self._skip):
            _, r, terminated, truncated, info = env.step(action)
            total += r
            if terminated or truncated:
                break
        return total, bool(terminated), bool(truncated), info

    def _obs(self) -> dict:
        return {
            "tiles": np.stack(self._grids),
            "extras": read_extras(self.ram, self._prev_action, self.n_actions),
        }
