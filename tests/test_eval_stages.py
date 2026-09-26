"""The eval CLI's --noop-max has to reach the emulator, not just sit in the config.

The random start delay is the only thing that varies an evaluation episode, so a flag that silently failed to
reach reset() would make every episode identical without any test noticing.
"""
import eval_stages
from env.tiles import TileMarioEnv


def test_noop_max_reaches_env_kwargs():
    cfg = eval_stages.eval_config({}, noop_max=7)
    assert cfg.noop_max == 7
    assert cfg.env_kwargs(["1-1"])["noop_max"] == 7


def test_eval_config_switches_off_training_only_aids():
    """Practice returns and generated terrain would both invalidate a held-out number."""
    cfg = eval_stages.eval_config({"practice_prob": 0.5, "procgen_prob": 0.9}, noop_max=30)
    assert cfg.practice_prob == 0.0 and cfg.procgen_prob == 0.0


def test_noop_max_zero_gives_an_identical_start_every_time():
    env = TileMarioEnv(**eval_stages.eval_config({}, noop_max=0).env_kwargs(["1-1"]))
    try:
        for seed in range(5):
            env.reset(seed=seed, stage="1-1")
            assert env._noops == 0
    finally:
        env.close()


def test_noop_max_bounds_the_start_delay_in_the_emulator():
    env = TileMarioEnv(**eval_stages.eval_config({}, noop_max=20).env_kwargs(["1-1"]))
    try:
        seen = set()
        for seed in range(15):
            env.reset(seed=seed, stage="1-1")
            seen.add(env._noops)
        assert seen, "no episodes were reset"
        assert max(seen) <= 20, f"start delay exceeded --noop-max: {sorted(seen)}"
        assert len(seen) > 1, "the start delay never varied, so episodes are not independent"
    finally:
        env.close()
