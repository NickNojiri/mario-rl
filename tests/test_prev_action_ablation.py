"""Hiding the previous action must change exactly one thing and nothing else.

An ablation is only worth running if the two arms differ in the one variable under test. Here the previous
action's one-hot slot is zeroed rather than removed, so tensor shapes, n_extras and the learning hash stay
identical and the arms are otherwise the same experiment.
"""
from config import PPOConfig, get_ppo_preset
from env.tiles import TileMarioEnv

RIGHT_A = 2  # SIMPLE_MOVEMENT


def _stepped_extras(obs_prev_action: bool):
    env = TileMarioEnv(stages=["1-1"], noop_max=0, obs_prev_action=obs_prev_action)
    try:
        env.reset(seed=0)
        obs, *_ = env.step(RIGHT_A)
        return obs["extras"], env.n_actions
    finally:
        env.close()


def test_the_previous_action_is_visible_by_default():
    extras, n_actions = _stepped_extras(True)
    assert extras[RIGHT_A] == 1.0
    assert extras[:n_actions].sum() == 1.0


def test_the_ablation_hides_it():
    extras, n_actions = _stepped_extras(False)
    assert extras[:n_actions].sum() == 0.0, "the previous-action one-hot must be entirely zero"


def test_the_slot_is_kept_so_shapes_are_identical():
    on, n_actions = _stepped_extras(True)
    off, _ = _stepped_extras(False)
    assert on.shape == off.shape and on.dtype == off.dtype


def test_everything_except_the_one_hot_is_unchanged():
    """Same seed, same action: only the ablated slice may differ."""
    on, n_actions = _stepped_extras(True)
    off, _ = _stepped_extras(False)
    assert (on[n_actions:] == off[n_actions:]).all(), "the ablation leaked into speed/float/powerup features"


def test_the_default_does_not_change_the_learning_hash():
    """Existing checkpoints must still resume: the new field is excluded from the hash at its default."""
    assert PPOConfig().learning_hash() == PPOConfig(obs_prev_action=True).learning_hash()
    assert PPOConfig().learning_hash() != PPOConfig(obs_prev_action=False).learning_hash()


def test_the_flag_reaches_the_env_through_the_config():
    cfg = get_ppo_preset("ppo_ablate_2m", obs_prev_action=False)
    assert cfg.env_kwargs(["1-1"])["obs_prev_action"] is False
    assert get_ppo_preset("ppo_ablate_2m").env_kwargs(["1-1"])["obs_prev_action"] is True
