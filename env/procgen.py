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
FLAGPOLE_IDS = (0x24, 0x25)
FLOOR_ROW = 11  # normal ground surface row (rows 11-12 are ground on flat SMB terrain)
PIRANHA = 0x0D


def column(height: int, extra_solid_from: int | None = None, blocks: dict | None = None) -> list[int]:
    """One 13-row column: ground from row FLOOR_ROW - height down; optional solid block from a row down to the
    ground (walls/stairs); optional floating tiles {row: metatile}. height < 0 means a pit (no ground)."""
    col = [0] * ROWS
    if height >= 0:
        top = FLOOR_ROW - height
        for r in range(top, ROWS):
            col[r] = GROUND
        if extra_solid_from is not None:
            for r in range(extra_solid_from, top):
                col[r] = GROUND
    for r, t in (blocks or {}).items():
        if 0 <= r < ROWS and col[r] == 0:
            col[r] = t
    return col


class LevelGenerator:
    """Deterministic (seed, difficulty) -> an endless stream of terrain columns, generated in segments."""

    def __init__(self, seed: int, difficulty: float):
        self.rng = np.random.default_rng(seed)
        self.d = float(np.clip(difficulty, 0.0, 1.0))
        self.height = 0
        self.cols: list[list[int]] = []
        self.pits: list[int] = []  # widths, for tests and stats
        self._flat(4)  # always start with a little runway

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
            self.cols.append(column(self.height, blocks=blocks))

    def _segment(self):
        r, d = self.rng.random(), self.d
        choices = ["flat", "pit", "step", "wall", "stairs", "blocks", "platform_pit"]
        weights = np.array([2.0, 1.0 + 1.5 * d, 1.5, 1.0 + d, 0.6 + 0.6 * d, 0.8, 0.3 + 0.9 * d])
        kind = choices[int(self.rng.choice(len(choices), p=weights / weights.sum()))]
        rng = self.rng
        if kind == "flat":
            self._flat(int(rng.integers(3, 9)))
        elif kind == "pit":
            max_w = 2 + int(round(2 * d))
            w = int(rng.integers(1, max_w + 1))
            self._flat(3)  # run-up
            self.cols += [column(-1)] * w
            self.pits.append(w)
            self._flat(3)  # landing
        elif kind == "step":
            max_change = 1 + int(round(2 * d))
            change = int(rng.integers(1, max_change + 1)) * (1 if rng.random() < 0.5 else -1)
            self.height = int(np.clip(self.height + change, 0, 4))
            self._flat(int(rng.integers(3, 7)))
        elif kind == "wall":
            tall = int(rng.integers(2, 3 + int(round(d)) + 1))  # 2-3 tall, up to 4 at high difficulty
            self._flat(3)
            top = FLOOR_ROW - self.height - tall
            self.cols += [column(self.height, extra_solid_from=top)] * 2
            self._flat(3)
        elif kind == "stairs":
            n = int(rng.integers(2, 4 + int(round(d))))
            self._flat(2)
            for k in range(1, n + 1):
                self.cols.append(column(self.height, extra_solid_from=FLOOR_ROW - self.height - k))
            if rng.random() < 0.5 * d:  # stairs ending in a pit, like the end of 1-1's staircases
                self.cols += [column(-1)] * 2
                self.pits.append(2)
            else:
                for k in range(n, 0, -1):
                    self.cols.append(column(self.height, extra_solid_from=FLOOR_ROW - self.height - k))
            self._flat(3)
        elif kind == "blocks":
            self._flat(int(rng.integers(4, 7)), blocks_row_offset=4)
        elif kind == "platform_pit":
            # a wide pit with a floating brick platform in the middle; each gap is at most 3 tiles
            gap = int(rng.integers(2, 3 + int(round(d))))
            plat_row = FLOOR_ROW - self.height - 1 - int(rng.integers(0, 2))
            self._flat(3)
            self.cols += [column(-1)] * gap
            self.cols += [column(-1, blocks={plat_row: BRICK})] * 3
            self.cols += [column(-1)] * gap
            self.pits += [gap, gap]
            self._flat(3)


class TerrainPatcher:
    """Writes generated columns into the SMB metatile buffer as the game renders them, off screen only."""

    FIRST_COLUMN = 16  # keep the opening screen original so Mario always spawns safely
    FLATTEN_BEFORE_FLAG = 9

    def __init__(self, generator: LevelGenerator):
        self.gen = generator
        self.patched_upto = self.FIRST_COLUMN - 1
        self.flag_col = None
        self.columns_patched = 0

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
