from dataclasses import replace

import numpy as np
import torch

from agent.ppo import PPOAgent, TilePolicy, compute_gae
from config import PPOConfig, get_ppo_preset
from env.tiles import EMPTY_ENEMY_ID, MARIO_ID, N_ENEMY_FEATS, N_ENEMY_SLOTS, n_extras

N_ACTIONS = 7


def test_gae_matches_hand_computation():
    # One env, 3 steps. Step 0 normal, step 1 truncated (bootstrap 5.0), step 2 terminated.
    g, lam = 0.9, 0.8
    r = np.array([[1.0], [2.0], [3.0]], np.float32)
    v = np.array([[10.0], [20.0], [30.0]], np.float32)
    term = np.array([[False], [False], [True]])
    trunc = np.array([[False], [True], [False]])
    tv = np.array([[0.0], [5.0], [0.0]], np.float32)
    adv, ret = compute_gae(r, v, np.array([99.0], np.float32), term, trunc, tv, g, lam)

    d2 = 3.0 - 30.0  # terminated: no future
    d1 = 2.0 + g * 5.0 - 20.0  # truncated: bootstrap from the real final obs, not v[2]
    d0 = 1.0 + g * 20.0 - 10.0
    a2, a1 = d2, d1  # chain is cut at every episode end
    a0 = d0 + g * lam * a1
    np.testing.assert_allclose(adv[:, 0], [a0, a1, a2], rtol=1e-6)
    np.testing.assert_allclose(ret, adv + v, rtol=1e-6)


def test_gae_bootstraps_last_step_when_not_done():
    r = np.array([[1.0]], np.float32)
    v = np.array([[0.0]], np.float32)
    f = np.array([[False]])
    adv, _ = compute_gae(r, v, np.array([4.0], np.float32), f, f, np.zeros((1, 1), np.float32), 0.5, 1.0)
    assert adv[0, 0] == 1.0 + 0.5 * 4.0


def _obs(n, obs_version=2, n_actions=N_ACTIONS, seed=0):
    rng = np.random.default_rng(seed)
    tiles = rng.integers(0, 256, (n, 4, 13, 16)).astype(np.int16)
    tiles[:, :, 10, 3] = MARIO_ID
    obs = {"tiles": tiles, "extras": rng.normal(size=(n, n_extras(n_actions, obs_version))).astype(np.float32)}
    if obs_version >= 3:
        ids = rng.integers(0, 64, (n, N_ENEMY_SLOTS)).astype(np.int16)
        ids[:, 3:] = EMPTY_ENEMY_ID
        obs["enemy_ids"] = ids
        obs["enemy_states"] = rng.integers(0, 256, (n, N_ENEMY_SLOTS)).astype(np.int16)
        obs["enemies"] = rng.normal(size=(n, N_ENEMY_SLOTS, N_ENEMY_FEATS)).astype(np.float32)
    return obs


def _batch(n, obs_version=2, n_actions=N_ACTIONS, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "obs": _obs(n, obs_version, n_actions, seed), "actions": rng.integers(n_actions, size=n),
        "logp": np.full(n, np.log(1 / n_actions), np.float32), "values": np.zeros(n, np.float32),
        "advantages": rng.normal(size=n).astype(np.float32), "returns": rng.normal(size=n).astype(np.float32),
    }


def test_policy_shapes_and_initial_uniformity():
    for version in (2, 3):
        net = TilePolicy(N_ACTIONS, 4, n_extras(N_ACTIONS, version), obs_version=version)
        logits, value = net({k: torch.as_tensor(v) for k, v in _obs(5, version).items()})
        assert logits.shape == (5, N_ACTIONS) and value.shape == (5,)
        probs = torch.softmax(logits, 1)
        assert (probs - 1 / N_ACTIONS).abs().max() < 0.05, "small policy-head init starts near uniform"


def test_v3_policy_is_invariant_to_enemy_slot_order():
    net = TilePolicy(N_ACTIONS, 4, n_extras(N_ACTIONS, 3), obs_version=3)
    obs = _obs(2, 3)
    perm = np.array([2, 0, 1, 5, 4, 3])
    shuffled = dict(obs, enemy_ids=obs["enemy_ids"][:, perm], enemy_states=obs["enemy_states"][:, perm],
                    enemies=obs["enemies"][:, perm])
    a = net({k: torch.as_tensor(v) for k, v in obs.items()})[0]
    b = net({k: torch.as_tensor(v) for k, v in shuffled.items()})[0]
    assert torch.allclose(a, b, atol=1e-5)


def test_update_runs_and_changes_weights():
    for preset, version, n_act in (("ppo_smoke", 2, 7), ("ppo_smoke_v3", 3, 12)):
        cfg = replace(get_ppo_preset(preset), minibatches=2, epochs=2)
        agent = PPOAgent(cfg, n_act, n_extras(n_act, version))
        before = [p.detach().clone() for p in agent.net.parameters()]
        stats = agent.update(_batch(64, version, n_act))
        assert all(np.isfinite(v) for v in stats.values())
        assert any(not torch.equal(a, p) for a, p in zip(before, agent.net.parameters()))


def test_act_is_reproducible_with_generator():
    cfg = get_ppo_preset("ppo_smoke")
    agent = PPOAgent(cfg, N_ACTIONS, n_extras(N_ACTIONS))
    obs = _obs(8)
    a1 = agent.act(obs, generator=torch.Generator().manual_seed(3))[0]
    a2 = agent.act(obs, generator=torch.Generator().manual_seed(3))[0]
    assert (a1 == a2).all()


def test_checkpoint_roundtrip(tmp_path):
    cfg = get_ppo_preset("ppo_smoke")
    a = PPOAgent(cfg, N_ACTIONS, n_extras(N_ACTIONS))
    a.update(_batch(64))
    a.global_step = 1234
    a.save(tmp_path / "ck.pt")
    b = PPOAgent(cfg, N_ACTIONS, n_extras(N_ACTIONS))
    b.load(tmp_path / "ck.pt")
    assert b.global_step == 1234 and b.updates == 1
    for pa, pb in zip(a.net.parameters(), b.net.parameters()):
        assert torch.equal(pa, pb)
    try:
        PPOAgent(replace(cfg, coin_reward=0.0), N_ACTIONS, n_extras(N_ACTIONS)).load(tmp_path / "ck.pt")
    except ValueError:
        pass
    else:
        raise AssertionError("coin_reward change must change the learning hash")


def test_config_roundtrip_preserves_stage_tuples():
    cfg = get_ppo_preset("ppo_smoke")
    assert PPOConfig.from_dict(cfg.to_dict()) == cfg


def test_v2_hash_unchanged_by_v3_fields():
    """A config saved before the v3 fields existed must still hash (and so resume) identically."""
    cfg = get_ppo_preset("ppo_1h")
    legacy = {k: v for k, v in cfg.to_dict().items()
              if k not in ("obs_version", "reward_version", "death_reward", "hurt_reward", "points_per_100",
                           "points_cap")}
    assert PPOConfig.from_dict(legacy).learning_hash() == cfg.learning_hash()
    assert get_ppo_preset("ppo_1h_v3b").learning_hash() != cfg.learning_hash()
