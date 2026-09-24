"""The scripted baseline: run right, press jump on a fixed cycle, never look at the observation.

A uniform random policy is a weak floor, so beating it proves very little. This one sets a floor made of the
level layout itself: whatever it scores is what the game gives away for free.
"""
import pytest

import eval_stages
from config import get_ppo_preset
from env.tiles import policy_action_names

SIMPLE = policy_action_names("simple")


def test_actions_are_found_by_button_name():
    run_a, jump_a = eval_stages.scripted_actions(SIMPLE)
    assert SIMPLE[run_a] == "RIGHT B", "the run action should hold B"
    assert SIMPLE[jump_a] == "RIGHT A B", "the jump action should hold A and B"


def test_actions_fall_back_when_there_is_no_run_button():
    run_a, jump_a = eval_stages.scripted_actions(policy_action_names("right_only"))
    names = policy_action_names("right_only")
    assert "A" in names[jump_a] and "A" not in names[run_a]


@pytest.mark.parametrize("text,expected", [("14,8", (14, 8)), ("6,3", (6, 3)), ("2,2", (2, 2))])
def test_parse_accepts_valid_cycles(text, expected):
    assert eval_stages.parse_scripted(text) == expected


@pytest.mark.parametrize("bad", ["14", "14,8,2", "", "a,b", "8,14", "14,0", "0,0"])
def test_parse_rejects_nonsense(bad):
    with pytest.raises(SystemExit):
        eval_stages.parse_scripted(bad)


def _job(stage="1-1", episodes=1, scripted=(14, 8)):
    cfg = get_ppo_preset("ppo_full")
    return {"stage": stage, "group": "train", "config": cfg.to_dict(), "episodes": episodes, "seed": 50_000,
            "greedy": False, "checkpoint": None, "mode": "safe", "scripted": scripted}


def test_the_script_runs_right_and_actually_jumps():
    row = eval_stages._eval_stage(_job())
    ep = row["episodes_detail"][0]
    assert ep["jumps"] > 0, "a jump cycle that never jumps is not a jump cycle"
    assert ep["left_presses"] == 0, "the script never presses left"
    assert row["mean_x_pos"] > 0


def test_the_script_is_deterministic_for_a_fixed_seed():
    a = eval_stages._eval_stage(_job())
    b = eval_stages._eval_stage(_job())
    assert a["mean_x_pos"] == b["mean_x_pos"]
    assert a["episodes_detail"][0]["x_pos"] == b["episodes_detail"][0]["x_pos"]


def test_a_different_cycle_plays_differently():
    """Guards against the cycle being ignored and every run being the same policy."""
    slow = eval_stages._eval_stage(_job(scripted=(24, 2)))
    fast = eval_stages._eval_stage(_job(scripted=(4, 3)))
    assert slow["episodes_detail"][0]["jumps"] != fast["episodes_detail"][0]["jumps"]
