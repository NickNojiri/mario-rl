"""Extra context for the agent: look-ahead columns and landing hints.

The look-ahead reads columns the game has loaded past the screen edge. Its whole value rests on those columns being
the real upcoming terrain, so the central test checks every look-ahead column against what the same column shows once
it scrolls on screen. The landing hints are checked on hand-built RAM where the right answer is known exactly.
"""
import numpy as np
import pytest

from config import PPOConfig, get_ppo_preset
from env.tiles import (COLS, ENEMY_BASE, JUMP_BY_SPEED, MAX_LOOKAHEAD, N_LANDING, ROWS, UNKNOWN_ID, VOCAB,
                       TileMarioEnv, full_jump_px, n_extras, read_landing_hints, read_tile_grid, screen_left_x)

RIGHT_B, RIGHT_AB = 3, 4  # SIMPLE_MOVEMENT


# ---------------------------------------------------------------- look-ahead
def test_default_observation_is_unchanged():
    env = TileMarioEnv(stages=["1-1"], noop_max=0)
    obs, _ = env.reset(seed=0)
    assert obs["tiles"].shape == (4, ROWS, COLS)
    env.close()


def test_lookahead_widens_the_grid():
    env = TileMarioEnv(stages=["1-1"], noop_max=0, lookahead=8)
    obs, _ = env.reset(seed=0)
    assert obs["tiles"].shape == (4, ROWS, COLS + 8)
    assert (obs["tiles"][-1][:, :COLS] == read_tile_grid(env.ram)).all(), "the on-screen part must not change"
    env.close()


@pytest.mark.parametrize("bad", [-1, MAX_LOOKAHEAD + 1])
def test_lookahead_out_of_range_is_refused(bad):
    with pytest.raises(ValueError):
        TileMarioEnv(stages=["1-1"], lookahead=bad)


def _episode_overlap(env, stage, period, hold):
    """One scripted episode: (cells compared, cells wrong) between look-ahead predictions and the later screen."""
    env.reset(seed=0, stage=stage)
    predicted, seen = {}, {}
    for t in range(600):
        ram = env.ram
        grid = read_tile_grid(ram, MAX_LOOKAHEAD)
        left_col = (screen_left_x(ram) + 8) // 16
        for c in range(COLS + MAX_LOOKAHEAD):
            column = grid[:, c]
            if c < COLS:
                seen[left_col + c] = column.copy()
            elif not (column == UNKNOWN_ID).all():
                predicted.setdefault(left_col + c, column.copy())
        _, _, terminated, truncated, _ = env.step(RIGHT_AB if t % period < hold else RIGHT_B)
        if terminated or truncated:
            break
    compared = mismatched = 0
    for col, pred in predicted.items():
        if col in seen:
            terrain = seen[col] < ENEMY_BASE  # enemies and Mario are drawn on screen only
            compared += int(terrain.sum())
            mismatched += int((pred[terrain] != seen[col][terrain]).sum())
    return compared, mismatched


def test_lookahead_columns_match_the_screen_once_they_scroll_into_view():
    """Every non-masked look-ahead column must equal that same level column once it is on screen.

    Scripted runs die early, so several stages and jump cycles are pooled to get enough overlap to judge."""
    compared = mismatched = 0
    for stage in ("1-1", "4-1", "3-2", "8-1"):
        env = TileMarioEnv(stages=[stage], noop_max=0, no_progress_steps=10_000, lookahead=MAX_LOOKAHEAD)
        for period, hold in ((14, 8), (10, 6), (16, 8), (6, 3), (20, 10)):
            c, m = _episode_overlap(env, stage, period, hold)
            compared, mismatched = compared + c, mismatched + m
        env.close()
    assert compared > 2_000, f"too little overlap to judge ({compared} cells)"
    # A block Mario bumps (? -> used) changes after it was predicted; nothing else should.
    assert mismatched / compared < 0.01, f"{mismatched}/{compared} look-ahead cells were wrong"


def test_unrendered_columns_are_masked_not_shown_stale():
    """At reset the game has rendered only part of the look-ahead; the rest must read UNKNOWN, not old terrain."""
    env = TileMarioEnv(stages=["1-1"], noop_max=0, lookahead=MAX_LOOKAHEAD)
    env.reset(seed=0)
    ram = env.ram.copy()
    ram[0x0725], ram[0x0726] = 0, COLS  # pretend the game has rendered exactly the visible screen
    ram[0x071A], ram[0x071C] = 0, 0
    grid = read_tile_grid(ram, MAX_LOOKAHEAD)
    assert (grid[:, COLS:] == UNKNOWN_ID).all()
    assert (grid[:, :COLS] != UNKNOWN_ID).all()
    env.close()


# ---------------------------------------------------------------- landing hints on hand-built RAM
def _ram(speed=0, pit_cols=(), rendered_cols=32):
    """Flat ground on rows 11-12 across both buffer pages, small Mario standing at level x 40."""
    ram = np.zeros(0x800, np.uint8)
    for page in (0, 1):
        for col in range(16):
            level_col = page * 16 + col
            if level_col in pit_cols:
                continue
            for row in (11, 12):
                ram[0x500 + page * 208 + row * 16 + col] = 0x54
    ram[0xB5] = 1                                  # Mario on the visible screen
    ram[0x6D], ram[0x86] = 0, 40                   # level x
    ram[0xCE] = 176                                # standing on row 11 (see read_tile_grid)
    ram[0x57] = speed & 0xFF                       # signed horizontal speed
    ram[0x0725], ram[0x0726] = divmod(rendered_cols, 16)
    return ram


def test_jump_table_matches_the_measurements():
    for speed, px in JUMP_BY_SPEED:
        assert full_jump_px(speed) == px
    assert full_jump_px(34) == pytest.approx((83 + 155) / 2)  # interpolated between walk max and 40
    assert full_jump_px(-48) == full_jump_px(48)


def test_standing_on_flat_ground_the_landing_is_safe_and_there_is_no_alarm():
    h = read_landing_hints(_ram(speed=0))
    assert h.shape == (N_LANDING,)
    safe, reach, alarm, known = h
    assert safe == 1.0 and known == 1.0 and alarm == 0.0
    assert reach == pytest.approx(53 / 160)


def test_a_standing_jump_into_a_pit_is_flagged_unsafe_but_a_running_jump_clears_it():
    pit = (5, 6)  # level x 80-111; a standing jump from x 40 lands at 101
    assert read_landing_hints(_ram(speed=0, pit_cols=pit))[0] == 0.0
    assert read_landing_hints(_ram(speed=48, pit_cols=pit))[0] == 1.0  # lands at 205, past the pit


def test_the_walk_off_alarm_rises_as_the_pit_gets_closer_in_time():
    pit = (5, 6)
    slow = read_landing_hints(_ram(speed=12, pit_cols=pit))[2]
    fast = read_landing_hints(_ram(speed=48, pit_cols=pit))[2]
    assert 0.0 < slow < fast < 1.0, "same distance, higher speed = less time = louder alarm"
    # front at 52; the pit edge is found at k = 32 px; walking at 24 units = 1.5 px/frame = 6 px per agent step
    assert read_landing_hints(_ram(speed=24, pit_cols=pit))[2] == pytest.approx(1 / (1 + 32 / 6))


def test_a_landing_the_game_has_not_rendered_is_not_claimed_safe():
    h = read_landing_hints(_ram(speed=48, rendered_cols=8))  # rendered up to x 128; a run jump lands at 205
    assert h[3] == 0.0 and h[0] == 0.0


def test_moving_left_never_raises_the_alarm():
    assert read_landing_hints(_ram(speed=-24, pit_cols=(5, 6)))[2] == 0.0


# ---------------------------------------------------------------- config and network plumbing
def test_defaults_do_not_change_the_learning_hash():
    assert PPOConfig().learning_hash() == PPOConfig(obs_lookahead=0, obs_hints=False).learning_hash()
    assert PPOConfig().learning_hash() != PPOConfig(obs_lookahead=8).learning_hash()
    assert PPOConfig().learning_hash() != PPOConfig(obs_hints=True).learning_hash()


def test_config_extras_width_matches_the_env():
    cfg = get_ppo_preset("ppo_ablate_2m", obs_hints=True, obs_lookahead=8)
    env = TileMarioEnv(**cfg.env_kwargs(["1-1"]))
    obs, _ = env.reset(seed=0)
    assert obs["extras"].shape == (cfg.n_extras(),) == (env.n_extras,)
    assert cfg.n_extras() == n_extras(env.n_actions) + 9 + N_LANDING
    env.close()


def test_the_network_runs_on_the_wider_grid_and_old_shapes_are_untouched():
    import torch

    from agent.ppo import PPOAgent

    old = PPOAgent(PPOConfig(), 7, PPOConfig().n_extras())
    assert old.net.embed.num_embeddings == VOCAB, "without look-ahead the vocabulary must not grow"

    cfg = PPOConfig(obs_lookahead=8, obs_hints=True)
    agent = PPOAgent(cfg, 7, cfg.n_extras())
    assert agent.net.embed.num_embeddings == VOCAB + 1
    obs = {"tiles": np.full((2, 4, ROWS, COLS + 8), UNKNOWN_ID, np.int16),
           "extras": np.zeros((2, cfg.n_extras()), np.float32)}
    logits, value = agent.net(agent.tensors(obs))
    assert logits.shape == (2, 7) and value.shape == (2,)
    assert torch.isfinite(logits).all()
