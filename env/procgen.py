"""Procedurally generated terrain for real Super Mario Bros stages.

SMB keeps its collision map in the RAM metatile buffer (0x500-0x69F: 2 pages x 13 rows x 16 columns), and the
area parser renders each level column into it about 8 columns ahead of the screen's right edge
(render position = RAM 0x725 * 16 + 0x726; verified with scripts/probe_columns.py). A column is rendered once,
so any column that is already rendered but not yet on screen can be replaced. Collision (Mario and enemies)
and the agent's tile-grid observation both read this buffer, so generated terrain is physically real while
Mario's physics, enemies, timer and flag stay the game's own. Only the drawn screen still shows the original
level, which the agent never sees.

Limits come from the measured jump table (scripts/jump_physics.py: max 5.2 tiles walking, 9.8 running, about
4 tiles of height), kept conservative so every generated level is completable:
  pits 1-2 tiles at difficulty 0 up to 4 at difficulty 1, always with >= 3 flat tiles of run-up and landing
  step changes up to 1 + 2*difficulty tiles, ground height 0-4 tiles above the normal floor
  walls/pipes 2 wide, 2-4 tall above the local ground
"""
from __future__ import annotations

import numpy as np

ROWS = 13
GROUND, BRICK, QBLOCK = 0x54, 0x51, 0xC0  # metatile ids: ground, brick, ?-block (coin)
# Solid tile themes seen in real stages (overworld ground, underground brick, castle block), so generated
# levels use the same vocabulary as held-out underground/castle stages instead of one tile id.
SOLID_THEMES = (0x54, 0x52, 0x62)
PIPE_TOP_L, PIPE_TOP_R, PIPE_L, PIPE_R = 0x12, 0x13, 0x14, 0x15  # real pipe metatiles (seen in 1-1)
FLAGPOLE_IDS = (0x24, 0x25)
FLOOR_ROW = 11  # normal ground surface row (rows 11-12 are ground on flat SMB terrain)
PIRANHA = 0x0D
# Ground walkers whose speed is scaled on generated levels (goomba, koopas, buzzy beetle, spiny).
WALKERS = (0x00, 0x02, 0x03, 0x06, 0x12)
MAX_DIFFICULTY = 1.5


def column(height: int, extra_solid_from: int | None = None, blocks: dict | None = None,
           solid: int = GROUND) -> list[int]:
    """One 13-row column: ground from row FLOOR_ROW - height down; optional solid block from a row down to the
    ground (walls/stairs); optional floating tiles {row: metatile}. height < 0 means a pit (no ground)."""
    col = [0] * ROWS
    if height >= 0:
        top = FLOOR_ROW - height
        for r in range(top, ROWS):
            col[r] = solid
        if extra_solid_from is not None:
            for r in range(extra_solid_from, top):
                col[r] = solid
    for r, t in (blocks or {}).items():
        if 0 <= r < ROWS and col[r] == 0:
            col[r] = t
    return col


class LevelGenerator:
    """Deterministic (seed, difficulty) -> an endless stream of terrain columns, generated in segments.

    Difficulty 0-1 is the original range; 1-1.5 is "hard mode": pits up to 6 tiles (running jump max is 9.8),
    steps up to 4, walls up to 4 tall, less flat ground between obstacles. Everything stays within the
    measured jump physics, so hard levels are hard but possible."""

    def __init__(self, seed: int, difficulty: float):
        self.rng = np.random.default_rng(seed)
        self.d = float(np.clip(difficulty, 0.0, MAX_DIFFICULTY))
        self.solid = int(self.rng.choice(SOLID_THEMES))
        self.height = 0
        self.cols: list[list[int]] = []
        self.pits: list[int] = []  # widths, for tests and stats
        self._flat(4)  # always start with a little runway

    def _col(self, height, extra_solid_from=None, blocks=None):
        return column(height, extra_solid_from, blocks, solid=self.solid)

    def get(self, i: int) -> list[int]:
        while len(self.cols) <= i:
            self._segment()
        return self.cols[i]

    # --- segments ---
    def _flat(self, n: int, blocks_row_offset: int | None = None):
        for k in range(n):
            blocks = None
            if blocks_row_offset is not None and 1 <= k < n - 1:
                row = FLOOR_ROW - self.height - blocks_row_offset
                blocks = {row: QBLOCK if self.rng.random() < 0.3 else BRICK}
            self.cols.append(self._col(self.height, blocks=blocks))

    def _segment(self):
        d = self.d
        base, hard = min(d, 1.0), max(0.0, d - 1.0)  # hard: 0 up to 0.5
        choices = ["flat", "pit", "step", "wall", "stairs", "blocks", "platform_pit", "pipe"]
        weights = np.array([2.0 - 2.0 * hard, 1.0 + 1.5 * d, 1.5, 1.0 + d, 0.6 + 0.6 * d, 0.8, 0.3 + 0.9 * d, 1.0])
        kind = choices[int(self.rng.choice(len(choices), p=weights / weights.sum()))]
        rng = self.rng
        if kind == "flat":
            self._flat(int(rng.integers(3, 9 - int(round(6 * hard)))))
        elif kind == "pit":
            max_w = 2 + int(round(2 * base)) + int(round(4 * hard))  # 2 -> 4 -> 6 tiles
            w = int(rng.integers(1, max_w + 1))
            self._flat(4 if w >= 5 else 3)  # run-up: wide pits need a running start
            self.cols += [self._col(-1)] * w
            self.pits.append(w)
            self._flat(3)  # landing
        elif kind == "step":
            max_change = 1 + int(round(2 * base)) + int(round(2 * hard))  # up to 4 (jump peak ~5.4 tiles)
            change = int(rng.integers(1, max_change + 1)) * (1 if rng.random() < 0.5 else -1)
            self.height = int(np.clip(self.height + change, 0, 4))
            self._flat(int(rng.integers(3, 7)))
        elif kind in ("wall", "pipe"):
            tall = int(rng.integers(2, 3 + int(round(base)) + 1))  # 2-3 tall, up to 4 at difficulty >= 0.5
            self._flat(3)
            top = FLOOR_ROW - self.height - tall
            if kind == "wall":
                self.cols += [self._col(self.height, extra_solid_from=top)] * 2
            else:
                ground = FLOOR_ROW - self.height
                left, right = self._col(self.height), self._col(self.height)
                left[top], right[top] = PIPE_TOP_L, PIPE_TOP_R
                for r in range(top + 1, ground):
                    left[r], right[r] = PIPE_L, PIPE_R
                self.cols += [left, right]
            self._flat(3)
        elif kind == "stairs":
            n = int(rng.integers(2, 4 + int(round(base)) + int(round(2 * hard))))
            self._flat(2)
            for k in range(1, n + 1):
                self.cols.append(self._col(self.height, extra_solid_from=FLOOR_ROW - self.height - k))
            if rng.random() < 0.5 * base + hard:  # stairs ending in a pit, like the end of 1-1's staircases
                w = 2 + int(round(2 * hard))
                self.cols += [self._col(-1)] * w
                self.pits.append(w)
            else:
                for k in range(n, 0, -1):
                    self.cols.append(self._col(self.height, extra_solid_from=FLOOR_ROW - self.height - k))
            self._flat(3)
        elif kind == "blocks":
            self._flat(int(rng.integers(4, 7)), blocks_row_offset=4)
        elif kind == "platform_pit":
            # a wide pit with a floating brick platform in the middle; each gap is at most 3 (4 in hard mode)
            gap = int(rng.integers(2, 3 + int(round(base)) + int(round(2 * hard))))
            plat_row = FLOOR_ROW - self.height - 1 - int(rng.integers(0, 2))
            self._flat(3)
            self.cols += [self._col(-1)] * gap
            self.cols += [self._col(-1, blocks={plat_row: BRICK})] * 3
            self.cols += [self._col(-1)] * gap
            self.pits += [gap, gap]
            self._flat(3)


class TerrainPatcher:
    """Writes generated columns into the SMB metatile buffer as the game renders them, off screen only."""

    FIRST_COLUMN = 16  # keep the opening screen original so Mario always spawns safely
    FLATTEN_BEFORE_FLAG = 9

    def __init__(self, generator: LevelGenerator, enemy_speed: float = 1.0):
        self.gen = generator
        self.enemy_speed = enemy_speed
        self.patched_upto = self.FIRST_COLUMN - 1
        self.flag_col = None
        self.columns_patched = 0
        self._scaled: dict[int, int] = {}  # enemy slot -> id already sped up this spawn
        self.enemies_lifted = 0

    def _unbury_enemies(self, ram):
        """Enemies spawn at the original level's height; where generated ground is higher they would walk
        inside it. Lift any walker whose center is inside solid terrain onto that column's surface."""
        for i in range(6):
            if not ram[0x0F + i] or int(ram[0x16 + i]) not in WALKERS or ram[0xB6 + i] != 1:
                continue
            x = int(ram[0x6E + i]) * 256 + int(ram[0x87 + i]) + 8
            y = int(ram[0xCF + i])
            row = (y + 8 - 32) // 16
            base = self._slot(ram, x // 16)
            if not (0 <= row < ROWS) or ram[base + row * 16] == 0:
                continue
            top = row
            while top > 0 and ram[base + (top - 1) * 16] != 0:
                top -= 1
            new_y = 16 * (top - 1) + 24  # standing on row `top` (row 10 <-> y 184 on flat ground)
            if new_y < 32:
                ram[0x0F + i] = 0  # no room above: drop the enemy
            else:
                ram[0xCF + i] = new_y
                self.enemies_lifted += 1

    def _speed_up_enemies(self, ram):
        """Scale each ground walker's x speed once at spawn (the game keeps it; verified 2x and 3x in
        scripts/probe_enemy_speed.py). Wall bounces flip the sign, so the higher speed is kept."""
        for i in range(6):
            if not ram[0x0F + i]:
                self._scaled.pop(i, None)
                continue
            eid = int(ram[0x16 + i])
            if self._scaled.get(i) == eid or eid not in WALKERS:
                continue
            v = int(ram[0x58 + i])
            v = v - 256 if v > 127 else v
            ram[0x58 + i] = max(-127, min(127, int(round(v * self.enemy_speed)))) & 0xFF
            self._scaled[i] = eid

    @staticmethod
    def _slot(ram, c: int) -> int:
        page, col = (c // 16) % 2, c % 16
        return 0x500 + page * 208 + col

    def _write(self, ram, c: int, tiles: list[int]):
        base = self._slot(ram, c)
        for r in range(ROWS):
            ram[base + r * 16] = tiles[r]

    def update(self, ram):
        """Call after every agent step (and after reset). Patches every rendered, still off-screen column."""
        render_col = int(ram[0x0725]) * 16 + int(ram[0x0726])
        left = int(ram[0x071A]) * 256 + int(ram[0x071C])
        first_offscreen = (left + 255) // 16 + 1
        for i in range(6):  # piranha plants live in pipes that generated terrain removes
            if ram[0x0F + i] and ram[0x16 + i] == PIRANHA:
                ram[0x0F + i] = 0
        if self.enemy_speed != 1.0:
            self._speed_up_enemies(ram)
        self._unbury_enemies(ram)
        if self.flag_col is not None:
            return
        start = max(self.patched_upto + 1, first_offscreen)
        for c in range(start, render_col):
            base = self._slot(ram, c)
            original = [int(ram[base + r * 16]) for r in range(ROWS)]
            if any(t in FLAGPOLE_IDS for t in original):
                # Level end reached: leave the flag and castle original, give the approach flat ground.
                self.flag_col = c
                for cc in range(max(first_offscreen, c - self.FLATTEN_BEFORE_FLAG), c):
                    self._write(ram, cc, column(0))
                return
            self._write(ram, c, self.gen.get(c - self.FIRST_COLUMN))
            self.patched_upto = c
            self.columns_patched += 1
