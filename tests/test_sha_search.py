"""Successive-halving logic. A bug here silently kills the best config, so the pure parts are pinned down."""
import random

import pytest

from config import PPOConfig
from scripts import sha_search as sha


def test_budgets_grow_by_eta():
    assert sha.budgets(250_000, 3, 3) == [250_000, 750_000, 2_250_000]


def test_promote_keeps_the_top_third_rounded_up():
    scores = {f"t{i}": float(i) for i in range(27)}
    kept = sha.promote(scores, 3)
    assert len(kept) == 9
    assert kept[0] == "t26" and set(kept) == {f"t{i}" for i in range(18, 27)}
    assert len(sha.promote({"a": 1.0, "b": 2.0}, 3)) == 1, "always keep at least one"


def test_a_failed_trial_is_never_promoted_over_a_real_score():
    scores = {"good": 400.0, "crashed": None, "bad": 100.0}
    assert sha.promote(scores, 3) == ["good"]
    assert sha.promote({"crashed": None, "bad": -50.0}, 2) == ["bad"]


def test_trial_zero_is_the_unmodified_default():
    trials = sha.make_trials(27)
    assert trials[0]["name"] == "t00" and trials[0]["overrides"] == {}
    assert len(trials) == 27 and len({t["name"] for t in trials}) == 27


def test_sampling_is_deterministic():
    assert sha.make_trials(27) == sha.make_trials(27)
    assert sha.make_trials(27, sample_seed=1) != sha.make_trials(27)


def test_samples_stay_inside_the_space():
    rng = random.Random(123)
    for _ in range(200):
        cfg = sha.sample_config(rng)
        assert set(cfg) == set(sha.SPACE)
        for key, spec in sha.SPACE.items():
            if spec[0] == "loguniform":
                assert spec[1] * 0.999 <= cfg[key] <= spec[2] * 1.001
            else:
                assert cfg[key] in spec[1]


def test_every_searched_field_is_a_real_config_field_the_cli_can_set():
    """train_ppo --set rejects unknown fields; a typo here would crash 26 runs at rung 0."""
    fields = set(PPOConfig.__dataclass_fields__)
    assert set(sha.SPACE) <= fields


def test_the_search_never_touches_the_problem_definition():
    """Reward, observation, action set, training noop_max and procgen are out of scope by design (ADR-0005)."""
    forbidden = {"reward_version", "obs_version", "actions", "noop_max", "procgen_prob", "obs_prev_action",
                 "death_reward", "coin_reward", "flag_reward", "train_stages", "test_stages"}
    assert not set(sha.SPACE) & forbidden


def test_set_args_format_booleans_the_way_the_cli_parses_them():
    args = sha.set_args({"plr": True, "lr": 0.0003, "epochs": 4})
    assert args == ["plr=true", "lr=0.0003", "epochs=4"]
    assert sha.set_args({"plr": False}) == ["plr=false"]


def test_total_steps_counts_resumed_budget_only_once():
    # 27 x 250k, then 9 more x 500k, then 3 more x 1.5M
    assert sha.total_steps(27, 3, [250_000, 750_000, 2_250_000]) == 27 * 250_000 + 9 * 500_000 + 3 * 1_500_000


def test_resuming_is_only_valid_while_nothing_schedules_on_total_steps():
    """Promotion resumes from a checkpoint instead of restarting. That equals one longer run only if no config
    field depends on total_steps. If an LR or entropy schedule appears, this must fail and the search must
    switch to restarting."""
    scheduled = {f for f in PPOConfig.__dataclass_fields__ if "anneal" in f or "schedule" in f or "decay" in f}
    assert not scheduled, f"schedule fields {scheduled} make resume != longer run"


def test_the_confirmation_compares_like_with_like():
    assert sha.BASE_PRESET == "ppo_ablate_2m"
    from config import get_ppo_preset
    assert get_ppo_preset(sha.BASE_PRESET).total_steps == sha.CONFIRM_BUDGET
    assert sha.SEARCH_SEED not in sha.CONFIRM_SEEDS, "confirmation seeds must be unseen during the search"


def _stub(monkeypatch, crash_at=None):
    """Replace training and evaluation with a deterministic score per trial; optionally crash mid-rung."""
    calls = []

    def fake_train(trial, budget, seed, run_name):
        return True

    def fake_eval(run_name, episodes, tag, stages="test"):
        name = run_name.split("/")[-1]
        calls.append((tag, name))
        if crash_at and (tag, len([c for c in calls if c[0] == tag])) == crash_at:
            raise KeyboardInterrupt  # the machine went down mid-rung
        return {"mean_x_pos": float(int(name[1:]))}  # t26 is best, t00 worst

    monkeypatch.setattr(sha, "train", fake_train)
    monkeypatch.setattr(sha, "evaluate", fake_eval)
    return calls


def test_a_full_search_halves_27_to_9_to_3_to_1(monkeypatch, tmp_path):
    calls = _stub(monkeypatch)
    path = tmp_path / "state.json"
    state = sha.load_state(path, sha.make_trials(27), [1, 3, 9], 3)
    sha.search(state, path, [1, 1, 1])
    assert [sum(1 for c in calls if c[0] == f"r{r}") for r in range(3)] == [27, 9, 3]
    assert state["winner"] == "t26"


def test_a_restart_mid_rung_does_not_promote_twice(monkeypatch, tmp_path):
    """The bug this guards: on restart, revisiting a finished rung re-promoted the survivors, cutting 9 to 3."""
    path = tmp_path / "state.json"
    _stub(monkeypatch, crash_at=("r1", 4))  # die on the 4th evaluation of rung 1
    state = sha.load_state(path, sha.make_trials(27), [1, 3, 9], 3)
    with pytest.raises(KeyboardInterrupt):
        sha.search(state, path, [1, 1, 1])

    calls = _stub(monkeypatch)  # restart from the saved state file, as a new process would
    state = sha.load_state(path, sha.make_trials(27), [1, 3, 9], 3)
    sha.search(state, path, [1, 1, 1])
    rung1 = [n for n, t in state["trials"].items() if "1" in t["rungs"]]
    assert len(rung1) == 9, f"rung 1 must still hold 9 trials after a restart, got {len(rung1)}"
    assert not [c for c in calls if c[0] == "r0"], "a finished rung must not be re-evaluated"
    assert state["winner"] == "t26"


@pytest.mark.parametrize("bad", [{"a": 1.0}, {}])
def test_promote_handles_tiny_rungs(bad):
    if bad:
        assert sha.promote(bad, 3) == ["a"]
    else:
        assert sha.promote(bad, 3) == []
