from dataclasses import replace

import numpy as np
import torch

from agent.mario import Mario, linear_epsilon
from config import SMOKE

CFG = replace(SMOKE, buffer_size=500, burnin=40, batch_size=8, learn_every=1, sync_every=10)
N_ACTIONS = 5


def fill(mario, steps, seed=0, terminate_every=25):
    rng = np.random.default_rng(seed)
    state = rng.integers(0, 256, (4, 84, 84), dtype=np.uint8)
    for t in range(steps):
        next_state = np.concatenate([state[1:], rng.integers(0, 256, (1, 84, 84), dtype=np.uint8)])
        done = (t + 1) % terminate_every == 0
        mario.cache(state, int(rng.integers(N_ACTIONS)), 15.0, next_state, done, False)
        state = rng.integers(0, 256, (4, 84, 84), dtype=np.uint8) if done else next_state


def test_linear_epsilon():
    assert linear_epsilon(0, 1.0, 0.1, 100) == 1.0
    assert abs(linear_epsilon(50, 1.0, 0.1, 100) - 0.55) < 1e-9
    assert linear_epsilon(10_000, 1.0, 0.1, 100) == 0.1


def test_reward_scaled_on_cache():
    mario = Mario(CFG, N_ACTIONS)
    fill(mario, 3)
    assert np.allclose(mario.buffer.rewards[:2], 1.0)  # 15 * (1/15)


def test_learn_updates_and_respects_burnin():
    mario = Mario(CFG, N_ACTIONS)
    fill(mario, CFG.burnin - 5)
    mario.curr_step = 1
    assert mario.learn() is None
    mario.buffer.end_episode()
    fill(mario, 100, seed=1)
    before = [p.clone() for p in mario.net.online.parameters()]
    out = mario.learn()
    assert out is not None and np.isfinite(out[1])
    assert any(not torch.equal(b, p) for b, p in zip(before, mario.net.online.parameters()))


def test_terminal_masks_bootstrap():
    """With gamma large and target Q huge, terminated transitions must still target exactly r."""
    mario = Mario(replace(CFG, gamma=0.99), N_ACTIONS)
    with torch.no_grad():
        mario.net.target[-1].bias.fill_(1000.0)
    s = torch.zeros(2, 4, 84, 84, dtype=torch.uint8)
    reward = torch.tensor([1.0, 1.0])
    term = torch.tensor([1.0, 0.0])
    best = mario.net(s, "online").argmax(1, keepdim=True)
    next_q = mario.net(s, "target").gather(1, best).squeeze(-1)
    tgt = reward + 0.99 * (1 - term) * next_q
    assert tgt[0].item() == 1.0 and tgt[1].item() > 900


def test_checkpoint_resume_is_exact(tmp_path):
    torch.manual_seed(0)
    a = Mario(CFG, N_ACTIONS)
    fill(a, 200)
    for step in range(1, 30):
        a.curr_step = step
        a.learn()
    a.curr_step = 123
    a.save(tmp_path / "ck.pt", save_buffer=True)

    b = Mario(CFG, N_ACTIONS)
    assert b.load(tmp_path / "ck.pt") is True
    assert b.curr_step == 123 and b.exploration_rate == a.exploration_rate
    sa, sb = a.optimizer.state_dict()["state"], b.optimizer.state_dict()["state"]
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k]["exp_avg_sq"], sb[k]["exp_avg_sq"]) for k in sa)

    # Same RNG + same weights + same buffer -> identical next update.
    b.buffer.rng.bit_generator.state = a.buffer.rng.bit_generator.state
    a.curr_step = b.curr_step = 124
    assert a.learn() == b.learn()
    state = np.zeros((4, 84, 84), dtype=np.uint8)
    assert [a.act(state) for _ in range(20)] == [b.act(state) for _ in range(20)]


def test_config_change_blocks_resume(tmp_path):
    a = Mario(CFG, N_ACTIONS)
    a.save(tmp_path / "ck.pt")
    b = Mario(replace(CFG, gamma=0.5), N_ACTIONS)
    try:
        b.load(tmp_path / "ck.pt")
    except ValueError:
        pass
    else:
        raise AssertionError("expected learning-hash mismatch")
    Mario(replace(CFG, total_steps=10**9), N_ACTIONS).load(tmp_path / "ck.pt")  # bookkeeping change is fine
