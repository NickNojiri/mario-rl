"""A fixed seed must reproduce a run exactly.

Every comparison in this project is one seed against another, so silent nondeterminism would make every
result unfalsifiable: a rerun that disagreed could not be distinguished from a real effect. This drives the
actual training stack -- SubprocVecEnv worker processes, the emulator, and the policy's action sampling --
rather than a stub, because that is where nondeterminism would come from.
"""
import random

import numpy as np
import torch

from agent.ppo import PPOAgent
from agent.vec_env import SubprocVecEnv
from config import get_ppo_preset
from env.tiles import ACTION_SETS, n_extras, n_policy_actions

N_ACTIONS = 200


def _first_actions(n: int = N_ACTIONS, seed: int = 0) -> np.ndarray:
    """Seed exactly as train_ppo.main() does, then collect the first n sampled actions."""
    cfg = get_ppo_preset("ppo_smoke", seed=seed, n_envs=2)
    assert cfg.device == "cpu"

    torch.set_num_threads(1)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    train_stages, _ = cfg.resolved_stages()
    n_joy = len(ACTION_SETS[cfg.actions])
    agent = PPOAgent(cfg, n_policy_actions(cfg.actions), n_extras(n_joy, cfg.obs_version, cfg.reward_version >= 4))
    envs = SubprocVecEnv(cfg.n_envs, cfg.env_kwargs(train_stages))
    try:
        obs = envs.reset(seed=cfg.seed * 10_007)
        actions: list[int] = []
        while len(actions) < n:
            action, _, _ = agent.act(obs)
            actions.extend(int(a) for a in action)
            obs = envs.step(action)[0]
        return np.array(actions[:n], dtype=np.int64)
    finally:
        envs.close()


def test_same_seed_gives_identical_actions():
    first = _first_actions()
    second = _first_actions()
    assert first.shape == (N_ACTIONS,)
    mismatch = np.flatnonzero(first != second)
    assert mismatch.size == 0, f"diverged at action {mismatch[0]}: {first[mismatch[0]]} vs {second[mismatch[0]]}"


def test_the_actions_are_not_all_the_same():
    """Guards against a vacuous pass: two constant sequences would also be 'identical'."""
    assert len(set(_first_actions().tolist())) > 1


def test_a_different_seed_gives_different_actions():
    assert not np.array_equal(_first_actions(seed=0), _first_actions(seed=1))
