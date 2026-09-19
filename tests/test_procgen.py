"""Procedural terrain: generator limits, buffer patching rules, and integration with the real game."""
import numpy as np

from env.procgen import FLOOR_ROW, GROUND, ROWS, LevelGenerator, TerrainPatcher, column
from env.tiles import TileMarioEnv, read_tile_grid

RIGHT_B = 3


def ground_top(col):
    """Row of the topmost tile of the solid run touching the bottom row (13 = pit)."""
    if col[ROWS - 1] == 0:
        return 13
    r = ROWS - 1
    while r > 0 and col[r - 1] != 0:
        r -= 1
    return r


def test_generator_is_deterministic():
    a, b = LevelGenerator(7, 0.6), LevelGenerator(7, 0.6)
    assert [a.get(i) for i in range(300)] == [b.get(i) for i in range(300)]
    assert [LevelGenerator(8, 0.6).get(i) for i in range(300)] != [a.get(i) for i in range(300)]


def test_generator_respects_jump_limits():
    for d, max_pit in ((0.0, 2), (1.0, 4)):
        for seed in range(40):
            g = LevelGenerator(seed, d)
            cols = [g.get(i) for i in range(400)]
            assert max(g.pits) <= max_pit
            # every pit has at least 3 columns of run-up ground right before it
            for i in range(3, len(cols)):
                if ground_top(cols[i]) == 13 and ground_top(cols[i - 1]) != 13 and not any(cols[i]):
                    assert all(ground_top(cols[i - k]) != 13 for k in (1, 2, 3)) or \
                        any(cols[i - k][r] for k in (1, 2, 3) for r in range(ROWS) if ground_top(cols[i - k]) == 13)
            tops = [ground_top(c) for c in cols if ground_top(c) != 13]
            assert min(tops) >= FLOOR_ROW - 4 - 4, "ground height + wall/stair never taller than 8 rows"
            assert all(t <= FLOOR_ROW for t in tops)


def test_hard_mode_limits_pipes_and_themes():
    from env.procgen import PIPE_TOP_L, PIPE_TOP_R, SOLID_THEMES

    seen_pipe, themes = False, set()
    for seed in range(60):
        g = LevelGenerator(seed, 1.5)
        cols = [g.get(i) for i in range(500)]
        themes.add(g.solid)
        assert max(g.pits) <= 6, "hard mode pits stay within the 9.8-tile running jump"
        for i, c in enumerate(cols):  # pits of 5-6 tiles always get 4 columns of run-up ground
            if i >= 4 and ground_top(c) == 13 and not any(c) and ground_top(cols[i - 1]) != 13:
                w = 0
                while i + w < len(cols) and not any(cols[i + w]):
                    w += 1
                if w >= 5:
                    assert all(ground_top(cols[i - k]) != 13 for k in range(1, 5))
        solid = {t for c in cols for t in c} - {0}
        assert g.solid in solid and g.solid in SOLID_THEMES
        if any(c[r] == PIPE_TOP_L for c in cols for r in range(ROWS)):
            seen_pipe = True
            left = next(i for i, c in enumerate(cols) if PIPE_TOP_L in c)
            assert PIPE_TOP_R in cols[left + 1], "pipes are 2 wide with matching top tiles"
    assert seen_pipe and themes == set(SOLID_THEMES)


def test_faster_enemies_and_shorter_clock_on_generated_levels():
    env = TileMarioEnv(stages=["1-1"], noop_max=0, procgen_prob=1.0, procgen_enemy_speed=2.0, procgen_clock_min=200)
    for s in range(6):
        _, info = env.reset(seed=s)
        ep = env._ep
        assert 200 <= env._smb._time <= 400 and ep["clock"] == env._smb._time
        assert 1.0 <= ep["enemy_speed"] <= 2.0
    _, info = env.reset(seed=0, stage="1-1")  # real stage: untouched
    assert env._smb._time == 400 and env._ep["clock"] == ""
    env.close()


def test_patcher_only_writes_rendered_offscreen_columns_and_handles_flag():
    ram = np.zeros(0x800, np.uint8)
    ram[0x071A], ram[0x071C] = 0, 0  # screen at level x 0 -> columns 0-15 visible
    ram[0x0725], ram[0x0726] = 1, 8  # game has rendered columns 0..23
    p = TerrainPatcher(LevelGenerator(0, 0.5))
    p.update(ram)
    assert p.patched_upto == 23 and p.columns_patched == 8, "patched exactly the off-screen rendered columns 16-23"
    # flag rendered at column 30: stop generating, flatten the off-screen approach, keep the flag
    ram[0x071C] = 64  # screen moved 4 columns -> first off-screen column is 20
    slot = 0x500 + 1 * 208 + (30 % 16)
    ram[slot + 5 * 16] = 0x25
    ram[0x0725], ram[0x0726] = 2, 0  # render position 32 -> columns up to 31 rendered
    p.update(ram)
    assert p.flag_col == 30
    for c in range(21, 30):
        base = 0x500 + ((c // 16) % 2) * 208 + c % 16
        assert [int(ram[base + r * 16]) for r in range(ROWS)] == column(0), f"column {c} flattened"
    assert ram[slot + 5 * 16] == 0x25, "flagpole untouched"


def test_env_patches_real_buffer_and_changes_collision_map():
    plain = TileMarioEnv(stages=["1-1"], noop_max=0)
    plain.reset(seed=0, stage="1-1", procgen=False)
    before = np.array(plain.ram[0x500:0x6A0])
    plain.close()
    env = TileMarioEnv(stages=["1-1"], noop_max=0)
    _, info = env.reset(seed=0, stage="1-1", procgen=True, difficulty=1.0)
    assert info["procgen"] is True
    after = np.array(env.ram[0x500:0x6A0])
    assert (before != after).any(), "generated columns differ from 1-1's own"
    first_page = after[:208].reshape(13, 16)
    assert (first_page == before[:208].reshape(13, 16)).all(), "the opening screen stays original"
    for _ in range(30):
        *_, terminated, truncated, _ = env.step(RIGHT_B)
        if terminated or truncated:  # the 1-1 goomba may end the run; patching before that is what counts
            break
    assert env._patcher.columns_patched > 8, "keeps patching as the screen scrolls"
    assert read_tile_grid(env.ram).shape == (13, 16)
    env.close()


def test_explicit_stage_never_generates_terrain():
    """Evaluation passes an explicit stage; procgen_prob must not leak generated terrain into it."""
    env = TileMarioEnv(stages=["1-1"], noop_max=0, procgen_prob=1.0)
    for s in range(5):
        _, info = env.reset(seed=s, stage="1-1")
        assert info["procgen"] is False
    _, info = env.reset(seed=0)
    assert info["procgen"] is True
    env.close()
