"""v4: macro advantages, physics hints, modes, practice replay, prioritized stage sampling."""
import numpy as np

from agent.macros import MacroStepper
from agent.ppo import compute_gae, compute_gae_options
from env.tiles import MACRO_SETS, MODE_WEIGHTS, MODES, N_HINTS, TileMarioEnv, n_extras, n_policy_actions

RIGHT, RIGHT_AB = 1, 4


def make(**kw):
    base = dict(stages=["1-1"], noop_max=0, no_progress_steps=10_000, obs_version=4, reward_version=4,
                actions="simple_macro")
    return TileMarioEnv(**{**base, **kw})


# ------------------------------------------------------------------ semi-MDP advantages
def test_options_gae_equals_gae_when_every_step_is_a_decision():
    rng = np.random.default_rng(0)
    T, N = 40, 3
    r = rng.normal(size=(T, N)).astype(np.float32)
    v = rng.normal(size=(T, N)).astype(np.float32)
    last = rng.normal(size=N).astype(np.float32)
    term = rng.random((T, N)) < 0.05
    trunc = (rng.random((T, N)) < 0.05) & ~term
    tv = rng.normal(size=(T, N)).astype(np.float32)
    a1, r1 = compute_gae(r, v, last, term, trunc, tv, 0.97, 0.9)
    a2, r2 = compute_gae_options(r, v, last, term, trunc, tv, np.ones((T, N), bool), 0.97, 0.9)
    np.testing.assert_allclose(a1, a2, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(r1, r2, rtol=1e-5, atol=1e-5)


def test_options_gae_hand_computation():
    # steps: 0 decision (macro of 3 steps: 0,1,2), 3 decision, 4 decision; rollout ends after 4 (bootstrap 10)
    g, lam = 0.9, 0.5
    r = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], np.float32)
    v = np.array([[1.0], [0.0], [0.0], [2.0], [3.0]], np.float32)
    dec = np.array([[True], [False], [False], [True], [True]])
    f = np.zeros((5, 1), bool)
    adv, ret = compute_gae_options(r, v, np.array([10.0], np.float32), f, f, np.zeros((5, 1), np.float32), dec, g, lam)
    d4 = 5 + g * 10 - 3
    a4 = d4
    d3 = 4 + g * 3 - 2
    a3 = d3 + g * lam * a4
    R0 = 1 + g * 2 + g ** 2 * 3
    d0 = R0 + g ** 3 * 2 - 1
    a0 = d0 + (g * lam) ** 3 * a3
    np.testing.assert_allclose([adv[0, 0], adv[3, 0], adv[4, 0]], [a0, a3, a4], rtol=1e-5)


def test_options_gae_macro_cut_by_death():
    g, lam = 0.9, 0.95
    r = np.array([[1.0], [1.0], [-10.0], [7.0]], np.float32)
    v = np.array([[0.5], [0.0], [0.0], [4.0]], np.float32)
    dec = np.array([[True], [False], [False], [True]])  # macro at 0 dies at step 2; step 3 is a new episode
    term = np.array([[False], [False], [True], [False]])
    f = np.zeros((4, 1), bool)
    adv, _ = compute_gae_options(r, v, np.array([100.0], np.float32), term, f, np.zeros((4, 1), np.float32), dec,
                                 g, lam)
    assert np.isclose(adv[0, 0], 1 + g * 1 + g ** 2 * -10 - 0.5), "no bootstrap and no chaining across death"


# ------------------------------------------------------------------ env
def test_policy_action_space_and_macro_expansion():
    env = make()
    assert env.n_policy_actions == n_policy_actions("simple_macro") == 9 + len(MACRO_SETS["simple_macro"])
    stepper = MacroStepper(env, "simple_macro")
    assert stepper.expand(RIGHT) == (RIGHT, 1)
    assert stepper.expand(9 + 2) == (RIGHT_AB, 8), "RUN JUMP L holds right+A+B for 8 steps"
    env.close()


def test_long_macro_jumps_farther_than_tap():
    dist = {}
    for name, policy_action in (("tap", 9 + 0), ("long", 9 + 2)):
        env = make(stages=["8-1"])
        stepper = MacroStepper(env, "simple_macro")
        env.reset(seed=0, stage="8-1", mode="safe")
        for _ in range(8):
            stepper.step(3)  # right+B run-up
        x0 = int(env.ram[0x6D]) * 256 + int(env.ram[0x86])
        stepper.step(policy_action)
        for _ in range(20):
            if int(env.ram[0x1D]) == 0:
                break
            stepper.step(3)
        dist[name] = int(env.ram[0x6D]) * 256 + int(env.ram[0x86]) - x0
        env.close()
    assert dist["long"] > dist["tap"] + 32, dist


def test_v4_obs_has_hints_and_mode():
    env = make()
    obs, info = env.reset(seed=0, mode="insane")
    assert obs["extras"].shape == (n_extras(env.n_actions, 4),)
    hints, mode = obs["extras"][-N_HINTS - len(MODES):-len(MODES)], obs["extras"][-len(MODES):]
    assert list(mode) == [0.0, 1.0] and info["mode"] == "insane"
    assert hints[0] == 1.0 and hints[1] == 0.0 and (hints[3:7] == 1).all(), "no pit within 10 tiles at 1-1 start"
    env.close()


def test_pit_hint_on_4_2():
    env = make(stages=["4-2"])
    obs, _ = env.reset(seed=0, stage="4-2", mode="safe")
    seen = []
    for _ in range(12):
        hints = obs["extras"][-N_HINTS - len(MODES):-len(MODES)]
        seen.append((float(hints[0]), float(hints[1])))
        obs, _, terminated, truncated, _ = env.step(RIGHT)
        if terminated or truncated:
            break
    assert any(d < 1.0 and w > 0 for d, w in seen), f"4-2 opens with pits: {seen}"
    env.close()


def test_mode_weights_change_death_reward():
    for mode in MODES:
        env = make()
        env.reset(seed=0, mode=mode)
        for _ in range(3000):
            _, _, terminated, truncated, info = env.step(RIGHT)
            if terminated or truncated:
                break
        assert info["reward_components"]["death"] == MODE_WEIGHTS[mode]["death"]
        assert info["episode"]["mode"] == mode
        env.close()


def test_practice_replay_returns_to_the_recorded_spot():
    env = make(practice_prob=1.0)
    env.reset(seed=0, mode="safe", practice=False)
    xs = []
    for _ in range(3000):
        _, _, terminated, truncated, info = env.step(RIGHT)
        xs.append(int(info["x_pos"]))
        if terminated or truncated:
            break
    assert env._archive["1-1"], "death was archived"
    noops, prefix = env._archive["1-1"][0]
    expected_x = xs[len(prefix) - 1]
    _, info = env.reset(seed=1, mode="safe")
    assert info["practice"] is True
    assert int(env.ram[0x6D]) * 256 + int(env.ram[0x86]) == expected_x, "deterministic replay lands on the same x"
    env.close()


def test_stage_weights_restrict_sampling():
    env = make(stages=["1-1", "1-2", "1-3"])
    env.set_stage_weights({"1-1": 0.0, "1-2": 1.0, "1-3": 0.0})
    stages = {env.reset(seed=s)[1]["stage"] for s in range(8)}
    assert stages == {"1-2"}
    env.set_stage_weights(None)
    assert len({env.reset(seed=s)[1]["stage"] for s in range(12)}) > 1
    env.close()
