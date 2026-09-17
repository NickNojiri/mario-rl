"""Find stages whose opening two screens have long flat ground with open sky above (for jump measurements)."""
import numpy as np

from env.tiles import ALL_STAGES, EXCLUDED_STAGES, TileMarioEnv, read_tile_grid, screen_left_x


def buffer_columns(ram):
    """The 2-screen metatile buffer as a [13, 32] grid in level order starting at the current screen."""
    left = screen_left_x(ram)
    cols = []
    for k in range(32):
        lx = left - (left % 16) + k * 16
        page, col = (lx // 256) % 2, (lx % 256) // 16
        cols.append([int(ram[0x500 + page * 208 + r * 16 + col]) for r in range(13)])
    return np.array(cols).T, left - (left % 16)


for stage in ALL_STAGES:
    if stage in EXCLUDED_STAGES:
        continue
    env = TileMarioEnv(stages=[stage], noop_max=0)
    env.reset(seed=0, stage=stage)
    grid, x0 = buffer_columns(env.ram)
    ground_row = 11
    best, run, start = 0, 0, 0
    for c in range(32):
        flat = grid[ground_row, c] != 0 and grid[ground_row - 1, c] == 0
        open_sky = (grid[:ground_row, c] == 0).all()
        if flat and open_sky:
            run += 1
            if run > best:
                best, start = run, c - run + 1
        else:
            run = 0
    mx = int(env.ram[0x6D]) * 256 + int(env.ram[0x86])
    print(f"{stage}: longest open flat run {best:2d} tiles, level x {x0 + start * 16}-{x0 + (start + best) * 16}, "
          f"mario x {mx}")
    env.close()
