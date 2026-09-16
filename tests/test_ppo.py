from dataclasses import replace

import numpy as np
import torch

from agent.ppo import PPOAgent, TilePolicy, compute_gae
from config import PPOConfig, get_ppo_preset
from env.tiles import MARIO_ID, n_extras

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


def _batch(n, seed=0):
    rng = np.random.default_rng(seed)
    tiles = rng.integers(0, 256, (n, 4, 13, 16)).astype(np.int16)
    tiles[:, :, 10, 3] = MARIO_ID
    return {
        "tiles": tiles, "extras": rng.normal(size=(n, n_extras(N_ACTIONS))).astype(np.float32),
        "actions": rng.integers(N_ACTIONS, size=n), "logp": np.full(n, np.log(1 / N_ACTIONS), np.float32),
        "values": np.zeros(n, np.float32), "advantages": rng.normal(size=n).astype(np.float32),
        "returns": rng.normal(size=n).astype(np.float32),
    }


def test_policy_shapes_and_initial_uniformity():
    net = TilePolicy(N_ACTIONS, 4, n_extras(N_ACTIONS))
    b = _batch(5)
    logits, value = net(torch.as_tensor(b["tiles"]), torch.as_tensor(b["extras"]))
    assert logits.shape == (5, N_ACTIONS) and value.shape == (5,)
    probs = torch.softmax(logits, 1)
    assert (probs - 1 / N_ACTIONS).abs().max() < 0.05, "small policy-head init starts near uniform"


def test_update_runs_and_changes_weights():
    cfg = replace(get_ppo_preset("ppo_smoke"), minibatches=2, epochs=2)
    agent = PPOAgent(cfg, N_ACTIONS, n_extras(N_ACTIONS))
    before = [p.detach().clone() for p in agent.net.parameters()]
    stats = agent.update(_batch(64))
    assert all(np.isfinite(v) for v in stats.values())
    assert any(not torch.equal(a, p) for a, p in zip(before, agent.net.parameters()))


def test_act_is_reproducible_with_generator():
    cfg = get_ppo_preset("ppo_smoke")
    agent = PPOAgent(cfg, N_ACTIONS, n_extras(N_ACTIONS))
    b = {k: _batch(8)[k] for k in ("tiles", "extras")}
    a1 = agent.act(b, generator=torch.Generator().manual_seed(3))[0]
    a2 = agent.act(b, generator=torch.Generator().manual_seed(3))[0]
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
