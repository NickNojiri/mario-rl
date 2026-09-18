"""Tile-grid observations read from Super Mario Bros RAM, and a multi-stage env built on them.

Why tiles instead of pixels: every real level (and any future procedural generator) can be described with
the same small grid of tile ids, so skill learned on one layout is expressed in the same vocabulary as every
other layout. Pixels carry per-level colors and scenery the agent would otherwise have to learn to ignore.

Observation (dict), obs_version 2:
  tiles  int16 [stack, 13, 16]  screen-aligned grid. 0-255 = the game's metatile id, 256+t = enemy of type t,
                                320 = Mario. Ids are embedded by the network, so no hand-made solidity table.
  extras float32 [n_extra]      previous action one-hot (makes "A is held" observable), Mario x/y speed,
                                float state one-hot (ground/jump/fall/flagpole), powerup one-hot.
obs_version 3 adds what version 2 cannot show (enemy direction is invisible after snapping to 16 px cells:
a goomba moves 2 px per agent step, verified by scripts/probe_enemies.py):
  enemy_ids    int16 [6]        enemy type per slot (64 = empty), slots sorted nearest-first
  enemy_states int16 [6]        raw enemy state byte (walking, shell, stomped, ...)
  enemies      float32 [6, 6]   active, dx/256, dy/240 relative to Mario in pixels, x speed/16, y speed/8,
                                on-screen flag
  extras gains Mario's sub-tile x, screen y and sub-tile y (exact position for jump timing).

Only information a player can see or already knows (layout, enemies and their motion, own motion, own last
button) is used.

Reward, reward_version 2: game reward (x velocity + clock + clipped death -15) + coin and flag bonuses.
reward_version 3 (every component logged per episode):
  progress  new ground only: max(0, x - farthest x so far), so walking back costs only time
  time      in-game clock ticks (<= 0)
  death     death_reward (default -150) instead of the game's clipped -15
  hurt      hurt_reward when the powerup level drops (big -> small)
  points    score gained from stomps, blocks and items (coin points and end-of-level bonuses excluded),
            points_per_100 per 100 points, capped per episode (stage 3-1 has an infinite koopa-shell farm)
  coins, flag

RAM addresses confirmed against gym_super_mario_bros/smb_env.py where it uses them (0x6D, 0x86, 0x71C,
0xB5, 0x756, 0x1D, 0x16-0x1A); the metatile buffer and enemy tables follow the SMB disassembly's SprObject
layout and are verified by scripts/render_tiles.py and scripts/probe_enemies.py.
"""
from __future__ import annotations

import warnings

import gym_super_mario_bros
import numpy as np
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from env.procgen import LevelGenerator, TerrainPatcher

warnings.filterwarnings("ignore", module=r"gym\.envs\.registration")

ROWS, COLS = 13, 16
ENEMY_BASE = 256
N_ENEMY_TYPES = 64
MARIO_ID = ENEMY_BASE + N_ENEMY_TYPES
VOCAB = MARIO_ID + 1
N_ENEMY_SLOTS = 6
EMPTY_ENEMY_ID = N_ENEMY_TYPES  # embedding index for an empty slot
N_ENEMY_FEATS = 6
ACTION_SETS = {"right_only": RIGHT_ONLY, "simple": SIMPLE_MOVEMENT, "complex": COMPLEX_MOVEMENT,
               "simple_macro": SIMPLE_MOVEMENT + [["left", "A"], ["left", "B"]]}
# Macro actions (options): (name, joypad action index, agent steps held). Expanded by the trainer / MacroStepper;
# the env itself only ever sees joypad actions. Lengths come from scripts/jump_physics.py: a full-distance jump
# needs A held 6-8 agent steps.
MACRO_SETS = {"simple_macro": [("RUN JUMP S", 4, 2), ("RUN JUMP M", 4, 5), ("RUN JUMP L", 4, 8),
                               ("WALK JUMP L", 2, 6), ("HOP BACK", 7, 4)]}
N_FLOAT_STATES, N_POWERUPS = 4, 3
_NES_NOOP = 0
N_HINTS = 9
MODES = ("safe", "insane")
# Measured max horizontal jump distance in px (scripts/jump_physics.py, 8-1 and 3-2 agree).
JUMP_PX = {"walk_tap": 42, "walk_max": 83, "run_tap": 70, "run_max": 156}
# reward_version 4: per-mode weights. safe = survive and collect; insane = speedrun.
MODE_WEIGHTS = {
    "safe": dict(progress=1.0, time=0.5, death=-300.0, hurt=-100.0, points_per_100=10.0, points_cap=300.0,
                 coin=15.0, flag=150.0),
    "insane": dict(progress=1.5, time=3.0, death=-100.0, hurt=-25.0, points_per_100=0.0, points_cap=0.0,
                   coin=0.0, flag=400.0),
}


def policy_action_names(actions: str) -> list[str]:
    names = [" ".join(b).upper() for b in ACTION_SETS[actions]]
    return names + [m[0] for m in MACRO_SETS.get(actions, [])]


def n_policy_actions(actions: str) -> int:
    return len(ACTION_SETS[actions]) + len(MACRO_SETS.get(actions, []))

ALL_STAGES = [f"{w}-{s}" for w in range(1, 9) for s in range(1, 5)]
# Water physics (2-2, 7-2) and maze castles that loop unless a hidden path is taken (4-4, 7-4, 8-4).
EXCLUDED_STAGES = ["2-2", "7-2", "4-4", "7-4", "8-4"]
# Held out from training: one of each level type, spread across worlds.
TEST_STAGES = ["2-1", "3-3", "4-2", "5-4", "7-1"]
TRAIN_STAGES = [s for s in ALL_STAGES if s not in EXCLUDED_STAGES and s not in TEST_STAGES]
# Overworld and athletic training stages used as bases for generated terrain (they end in a flagpole and have no
# ceiling). Held-out stages are never used as bases.
PROCGEN_BASE_STAGES = ["1-1", "1-3", "3-1", "3-2", "4-1", "5-1", "5-2", "5-3", "6-1", "6-3", "8-1", "8-2", "8-3"]


def n_extras(n_actions: int, obs_version: int = 2, modes: bool = False) -> int:
    """modes=True adds the play-style one-hot (only used by reward_version 4)."""
    return (n_actions + 2 + N_FLOAT_STATES + N_POWERUPS + (3 if obs_version >= 3 else 0)
            + (N_HINTS if obs_version >= 4 else 0) + (len(MODES) if modes else 0))


def _s8(v) -> int:
    v = int(v)
    return v - 256 if v > 127 else v


def screen_left_x(ram) -> int:
    return int(ram[0x071A]) * 256 + int(ram[0x071C])


def mario_level_xy(ram) -> tuple[int, int]:
    """Mario's level x and drawn-sprite y (RAM y + 16, see read_tile_grid)."""
    return int(ram[0x6D]) * 256 + int(ram[0x86]), int(ram[0xCE]) + 16


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
        # Player y in RAM sits 16px above the drawn sprite (checked against screenshots: standing Mario at
        # y=176 is drawn on row 10, directly above ground row 11).
        sx = int(ram[0x6D]) * 256 + int(ram[0x86]) - left
        r, c = (int(ram[0xCE]) + 16 + 8 - 32) // 16, (sx + 8) // 16
        if 0 <= r < ROWS and 0 <= c < COLS:
            grid[r, c] = MARIO_ID
    return grid


def read_enemies(ram) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-slot enemy type, state and motion features relative to Mario, nearest first."""
    mx, my = mario_level_xy(ram)
    slots = []
    for i in range(N_ENEMY_SLOTS):
        if ram[0x0F + i] == 0:
            continue
        ex = int(ram[0x6E + i]) * 256 + int(ram[0x87 + i])
        ey = int(ram[0xCF + i])
        dx, dy = ex - mx, ey - my
        slots.append((abs(dx) + abs(dy), min(int(ram[0x16 + i]), N_ENEMY_TYPES - 1), int(ram[0x1E + i]),
                      [1.0, dx / 256.0, dy / 240.0, _s8(ram[0x58 + i]) / 16.0, _s8(ram[0xA0 + i]) / 8.0,
                       float(ram[0xB6 + i] == 1)]))
    slots.sort(key=lambda s: s[0])
    ids = np.full(N_ENEMY_SLOTS, EMPTY_ENEMY_ID, np.int16)
    states = np.zeros(N_ENEMY_SLOTS, np.int16)
    feats = np.zeros((N_ENEMY_SLOTS, N_ENEMY_FEATS), np.float32)
    for k, (_, eid, st, f) in enumerate(slots):
        ids[k], states[k], feats[k] = eid, st, f
    return ids, states, feats


def read_extras(ram, prev_action: int, n_actions: int, obs_version: int = 2) -> np.ndarray:
    out = np.zeros(n_extras(n_actions, min(obs_version, 3)), dtype=np.float32)  # v4 hints/mode appended in _obs
    if prev_action >= 0:
        out[prev_action] = 1.0
    out[n_actions] = _s8(ram[0x57]) / 40.0  # horizontal speed
    out[n_actions + 1] = _s8(ram[0x9F]) / 8.0  # vertical speed
    out[n_actions + 2 + min(int(ram[0x1D]), N_FLOAT_STATES - 1)] = 1.0
    out[n_actions + 2 + N_FLOAT_STATES + min(int(ram[0x756]), N_POWERUPS - 1)] = 1.0
    if obs_version >= 3:
        mx, my = mario_level_xy(ram)
        base = n_actions + 2 + N_FLOAT_STATES + N_POWERUPS
        out[base] = (mx % 16) / 16.0
        out[base + 1] = min(my, 255) / 240.0 if ram[0xB5] == 1 else 1.0
        out[base + 2] = (my % 16) / 16.0
    return out


def _column_solid_from(ram, level_x: int, from_row: int) -> int:
    """First solid row at or below from_row in the column holding level_x (13 = none, i.e. a pit)."""
    page, col = (level_x // 256) % 2, (level_x % 256) // 16
    base = 0x500 + page * 208 + col
    for r in range(max(from_row, 0), 13):
        if ram[base + r * 16] != 0:
            return r
    return 13


def read_hints(ram) -> np.ndarray:
    """Physics hints a player reads at a glance (obs_version 4), all derived from what is on screen:
      0 distance from Mario's front to the next pit (fraction of 10 tiles; 1 = none within 10 tiles)
      1 that pit's width (fraction of 10 tiles)
      2 ground step 2 tiles ahead (+ = wall/step up, - = step down), in units of 4 tiles
      3-6 can the pit (edge distance + width) be cleared by a walking tap / walking full / running tap / running
          full jump (measured distances in JUMP_PX); all 1 when there is no pit
      7 closeness in time to the nearest enemy ahead at Mario's height: 1 / (1 + agent steps to contact)
      8 stomp opportunity: Mario falling with an enemy just below
    Mario's feet row uses the small-Mario offset; searches go downward, so a big Mario still finds the ground."""
    out = np.zeros(N_HINTS, np.float32)
    if ram[0xB5] != 1:
        return out
    mx, my = mario_level_xy(ram)
    feet_row = int(ram[0xCE]) // 16
    front = mx + 12
    pit_start = pit_width = None
    for k in range(0, 160, 8):
        supported = _column_solid_from(ram, front + k, feet_row) < 13
        if pit_start is None and not supported:
            pit_start = k
        elif pit_start is not None and supported:
            pit_width = k - pit_start
            break
    if pit_start is not None:
        pit_width = pit_width if pit_width is not None else 160 - pit_start
        need = pit_start + pit_width + 8
        out[0], out[1] = pit_start / 160.0, pit_width / 160.0
        out[3:7] = [need <= JUMP_PX["walk_tap"], need <= JUMP_PX["walk_max"], need <= JUMP_PX["run_tap"],
                    need <= JUMP_PX["run_max"]]
    else:
        out[0], out[3:7] = 1.0, 1.0
    here = _column_solid_from(ram, mx + 8, feet_row)
    ahead = _column_solid_from(ram, mx + 8 + 32, feet_row - 2)
    if here < 13 and ahead < 13:
        out[2] = np.clip((here - ahead) / 4.0, -1.0, 1.0)

    mario_vx = _s8(ram[0x57]) / 16.0  # px per frame (~16 speed units per px/frame, checked against RAM motion)
    falling = int(ram[0x1D]) in (1, 2) and _s8(ram[0x9F]) > 0
    for i in range(N_ENEMY_SLOTS):
        if ram[0x0F + i] == 0:
            continue
        dx = int(ram[0x6E + i]) * 256 + int(ram[0x87 + i]) - mx
        dy = int(ram[0xCF + i]) - my
        if dx > 0 and abs(dy) < 24:
            closing = mario_vx - _s8(ram[0x58 + i]) / 16.0
            if closing > 0.05:
                steps = max(dx - 16, 0) / closing / 4.0
                out[7] = max(out[7], 1.0 / (1.0 + steps))
        if falling and abs(dx) < 16 and 0 < dy < 48:
            out[8] = 1.0
    return out


def _find_smb(env):
    """gym's EnvCompatibility reports itself as .unwrapped, so walk .env links to the real SuperMarioBrosEnv."""
    while not hasattr(env, "ram"):
        env = env.env
    return env


REWARD_COMPONENTS = ("progress", "time", "death", "hurt", "points", "coins", "flag")


class TileMarioEnv:
    """Gymnasium-style env over a set of real SMB stages; a stage is sampled at every reset.

    info["game_reward"] keeps the game's own unshaped reward so runs with different reward versions can be
    compared. info["episode"] (on the final step) carries outcome, behaviour and reward-component statistics.
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
        obs_version: int = 2,
        reward_version: int = 2,
        death_reward: float = -150.0,
        hurt_reward: float = -50.0,
        points_per_100: float = 5.0,
        points_cap: float = 150.0,
        mode_safe_prob: float = 0.5,
        practice_prob: float = 0.0,
        procgen_prob: float = 0.0,
        procgen_stages=None,
        procgen_difficulty: float = 1.0,
        render_mode: str | None = None,
    ):
        self._procgen_prob, self._procgen_difficulty = procgen_prob, procgen_difficulty
        self._procgen_stages = [s for s in (procgen_stages or PROCGEN_BASE_STAGES) if s in stages] or list(stages)
        self._patcher = None
        self.stages = list(stages)
        self.action_set = ACTION_SETS[actions]
        self.n_actions = len(self.action_set)  # joypad actions (what the emulator receives)
        self.n_policy_actions = n_policy_actions(actions)  # joypad actions + macros (what the policy chooses)
        self.policy_action_names = policy_action_names(actions)
        self._mode_safe_prob, self._practice_prob = mode_safe_prob, practice_prob
        self._stage_weights = None
        self._archive: dict[str, list] = {}  # stage -> [(noops, joypad action prefix)] just before recent deaths
        self.mode = MODES[0]
        self.obs_version, self.reward_version = obs_version, reward_version
        self.uses_modes = reward_version >= 4
        self.n_extras = n_extras(self.n_actions, obs_version, self.uses_modes)
        self._skip, self._stack = skip, stack
        self._no_progress_steps, self._noop_max = no_progress_steps, noop_max
        self._coin_reward, self._flag_reward = coin_reward, flag_reward
        self._death_reward, self._hurt_reward = death_reward, hurt_reward
        self._points_per_100, self._points_cap = points_per_100, points_cap
        self._render_mode = render_mode
        self.frame_callback = None  # optional fn() called after every emulator frame (live demo rendering)
        self._envs: dict[str, tuple] = {}
        self._rng = np.random.default_rng()
        self.stage = None
        self._left = {i for i, b in enumerate(self.action_set) if "left" in b}
        self._jump = {i for i, b in enumerate(self.action_set) if "A" in b}
        self._noop = {i for i, b in enumerate(self.action_set) if b == ["NOOP"]}

    # --- env construction (lazy: one emulator per stage actually visited) ---
    def _get(self, stage: str):
        if stage not in self._envs:
            base = gym_super_mario_bros.make(
                f"SuperMarioBros-{stage}-v0", apply_api_compatibility=True,
                render_mode=self._render_mode, disable_env_checker=True,
            )
            self._envs[stage] = (JoypadSpace(base, self.action_set), base, _find_smb(base))
        return self._envs[stage]

    def prebuild(self):
        """Build every stage's emulator now (~0.3 s each) instead of on first visit mid-rollout."""
        for stage in set(self.stages) | set(self._procgen_stages if self._procgen_prob > 0 else []):
            self._get(stage)

    @property
    def ram(self):
        return self._smb.ram

    @property
    def screen(self) -> np.ndarray:
        return self._smb.screen

    def set_stage_weights(self, weights: dict | None):
        """Prioritized stage sampling (PLR); None = uniform."""
        if weights is None:
            self._stage_weights = None
            return
        w = np.array([max(float(weights.get(s, 0.0)), 0.0) for s in self.stages])
        self._stage_weights = w / w.sum() if w.sum() > 0 else None

    def reset(self, seed: int | None = None, stage: str | None = None, mode: str | None = None,
              practice: bool | None = None, procgen: bool | None = None, difficulty: float | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if procgen is None:
            procgen = stage is None and self._rng.random() < self._procgen_prob
        self._patcher = None
        if procgen:  # generated terrain on an overworld base stage (real physics, enemies, timer and flag)
            if stage is None:
                stage = self._procgen_stages[int(self._rng.integers(len(self._procgen_stages)))]
            d = float(self._rng.uniform(0.0, self._procgen_difficulty)) if difficulty is None else difficulty
            self._patcher = TerrainPatcher(LevelGenerator(int(self._rng.integers(2 ** 31)), d))
            practice = False
        if stage is None:
            idx = (self._rng.choice(len(self.stages), p=self._stage_weights) if self._stage_weights is not None
                   else self._rng.integers(len(self.stages)))
            stage = self.stages[int(idx)]
        self.stage = stage
        if self.uses_modes:
            self.mode = mode or (MODES[0] if self._rng.random() < self._mode_safe_prob else MODES[1])
        else:
            self.mode = "none"
        self._joypad, self._base, self._smb = self._get(self.stage)

        entries = self._archive.get(self.stage, [])
        if practice is None:
            practice = bool(entries) and self._rng.random() < self._practice_prob
        practice = practice and bool(entries)
        if practice:
            noops, prefix = entries[int(self._rng.integers(len(entries)))]
        else:
            noops, prefix = int(self._rng.integers(0, self._noop_max + 1)), []

        _, info = self._joypad.reset()
        for _ in range(noops):
            _, terminated, truncated, info = self._raw_skip(_NES_NOOP, use_joypad=False)
            if terminated or truncated:
                _, info = self._joypad.reset()
        # Go-Explore style return: the emulator is deterministic, so replaying the recorded prefix lands exactly
        # where Mario was shortly before a past death. No reward or statistics are collected during the replay.
        callback, self.frame_callback = self.frame_callback, None
        for a in prefix:
            _, terminated, truncated, _ = self._raw_skip(a, use_joypad=True)
            if terminated or truncated:  # should not happen for a prefix of a recorded episode
                _, _ = self._joypad.reset()
                prefix, practice = [], False
                break
        self.frame_callback = callback
        self._noops, self._actions = noops, list(prefix)
        self.practice = practice
        if self._patcher is not None:
            self._patcher.update(self.ram)

        grid = read_tile_grid(self.ram)
        self._grids = [grid] * self._stack
        self._prev_action = prefix[-1] if prefix else -1
        smb = self._smb
        self._coins = int(smb._coins)  # same source as info["coins"]
        self._score = int(smb._score)
        self._time = int(smb._time)
        self._status = int(self.ram[0x756])
        self._far_x = mario_level_xy(self.ram)[0]
        self._points_paid = 0.0
        self._best_x = None
        self._since_progress = 0
        self._ep = {"stage": self.stage, "reward": 0.0, "game_reward": 0.0, "length": 0, "coins": 0, "x_pos": 0,
                    "max_x": 0, "flag_get": False, "left_presses": 0, "noop_presses": 0, "jumps": 0,
                    "point_events": 0, "points": 0, "hurts": 0, "death_cause": "", "mode": self.mode,
                    "practice": practice, "procgen": self._patcher is not None,
                    "difficulty": round(self._patcher.gen.d, 3) if self._patcher else "",
                    **{f"r_{k}": 0.0 for k in REWARD_COMPONENTS}}
        return self._obs(), {"stage": self.stage, "mode": self.mode, "practice": practice,
                             "procgen": self._patcher is not None}

    def step(self, action: int):
        action = int(action)
        game_reward, terminated, truncated, info = self._raw_skip(action, use_joypad=True)
        ram = self.ram
        if self._patcher is not None:
            self._patcher.update(ram)  # before the grid is read, so observations show generated terrain
        flag = bool(info["flag_get"])
        x = int(info["x_pos"])

        coins_now = int(info["coins"])
        coin_delta = (coins_now - self._coins) % 100  # counter wraps at 100 (1-up)
        self._coins = coins_now
        score_now = int(info["score"])
        points = 0 if flag else max(0, score_now - self._score - 200 * coin_delta)  # coins score 200 each
        self._score = score_now
        time_now = int(info["time"])
        time_delta = min(0, time_now - self._time)
        self._time = time_now
        status_now = int(ram[0x756])
        hurt = status_now < self._status and not terminated
        self._status = status_now
        died = terminated and not flag

        if self.reward_version >= 4:
            w = MODE_WEIGHTS[self.mode]
        else:
            w = dict(progress=1.0, time=1.0, death=self._death_reward, hurt=self._hurt_reward,
                     points_per_100=self._points_per_100, points_cap=self._points_cap, coin=self._coin_reward,
                     flag=self._flag_reward)
        comp = {
            "progress": w["progress"] * float(min(max(0, x - self._far_x), 40)),  # cap: area changes jump x
            "time": w["time"] * float(time_delta),
            "death": w["death"] if died else 0.0,
            "hurt": w["hurt"] if hurt else 0.0,
            "points": 0.0,
            "coins": w["coin"] * coin_delta,
            "flag": w["flag"] if flag else 0.0,
        }
        self._far_x = max(self._far_x, x)
        if points and self._points_paid < w["points_cap"]:
            comp["points"] = min(w["points_per_100"] * points / 100.0, w["points_cap"] - self._points_paid)
            self._points_paid += comp["points"]
        if self.reward_version >= 3:
            reward = sum(comp.values())
        else:
            reward = game_reward + comp["coins"] + comp["flag"]

        if self._best_x is not None and x > self._best_x:
            self._since_progress = 0
        else:
            self._since_progress += 1
        self._best_x = x if self._best_x is None else max(self._best_x, x)
        if not terminated and self._since_progress >= self._no_progress_steps:
            truncated = True
            info["truncated_reason"] = "no_progress"

        ep = self._ep
        self._actions.append(action)
        if died:
            ep["death_cause"] = self._death_cause(info)
            if self._practice_prob > 0 and self._patcher is None:  # generated terrain can't be replayed
                self._remember_before_death()
        self._grids = self._grids[1:] + [read_tile_grid(ram)]
        ep["jumps"] += int(action in self._jump and self._prev_action not in self._jump)
        ep["left_presses"] += int(action in self._left)
        ep["noop_presses"] += int(action in self._noop)
        self._prev_action = action
        ep["reward"] += reward
        ep["game_reward"] += game_reward
        ep["length"] += 1
        ep["coins"] += coin_delta
        ep["points"] += points
        ep["point_events"] += int(points > 0)
        ep["hurts"] += int(hurt)
        ep["x_pos"] = x
        ep["max_x"] = self._best_x
        ep["flag_get"] = flag
        for k, v in comp.items():
            ep[f"r_{k}"] += v
        info.update(stage=self.stage, game_reward=game_reward, coin_delta=coin_delta, reward_components=comp)
        if terminated or truncated:
            if not ep["death_cause"]:
                ep["death_cause"] = "flag" if flag else ("stall" if truncated else "unknown")
            info["episode"] = dict(ep, terminated=terminated, truncated=truncated)
        return self._obs(), reward, terminated, truncated, info

    def _remember_before_death(self, max_entries: int = 20):
        """Archive a replayable prefix ending 6-20 agent steps before this death (Go-Explore 'remember')."""
        back = int(self._rng.integers(6, 21))
        cut = len(self._actions) - back
        if cut <= 0:
            return
        entries = self._archive.setdefault(self.stage, [])
        entries.append((self._noops, list(self._actions[:cut])))
        if len(entries) > max_entries:
            entries.pop(0)

    def _death_cause(self, info) -> str:
        ram = self.ram
        if ram[0xB5] > 1:
            return "pit"
        if int(info["time"]) == 0:
            return "timeout"
        mx, my = mario_level_xy(ram)
        best, best_d = None, 32  # an enemy within ~2 tiles of Mario
        for i in range(N_ENEMY_SLOTS):
            if ram[0x0F + i] == 0:
                continue
            d = abs(int(ram[0x6E + i]) * 256 + int(ram[0x87 + i]) - mx) + abs(int(ram[0xCF + i]) - my)
            if d < best_d:
                best, best_d = int(ram[0x16 + i]), d
        return f"enemy_0x{best:02x}" if best is not None else "hazard"

    def close(self):
        for joypad, _, _ in self._envs.values():
            joypad.close()
        self._envs.clear()

    def _raw_skip(self, action: int, use_joypad: bool):
        env = self._joypad if use_joypad else self._base
        total, terminated, truncated, info = 0.0, False, False, {}
        for _ in range(self._skip):
            _, r, terminated, truncated, info = env.step(action)
            total += r
            if self.frame_callback is not None:
                self.frame_callback()
            if terminated or truncated:
                break
        return total, bool(terminated), bool(truncated), info

    def _obs(self) -> dict:
        ram = self.ram
        obs = {
            "tiles": np.stack(self._grids),
            "extras": read_extras(ram, self._prev_action, self.n_actions, self.obs_version),
        }
        if self.obs_version >= 3:
            obs["enemy_ids"], obs["enemy_states"], obs["enemies"] = read_enemies(ram)
        if self.obs_version >= 4:
            parts = [obs["extras"], read_hints(ram)]
            if self.uses_modes:
                parts.append(np.array([self.mode == m for m in MODES], np.float32))
            obs["extras"] = np.concatenate(parts)
        return obs
