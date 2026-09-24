"""Level-length normalization.

Raw x_pos is not comparable across stages of different lengths, so a mean over stages silently weights the
long ones. progress_fraction fixes that -- but only if the flagpole position was measured rather than
guessed, so the unmeasured case has to stay None all the way through instead of defaulting to something.
"""
import json
from pathlib import Path

import eval_stages
from scripts.measure_stage_lengths import entry, harvest

FIXTURE = Path(__file__).parent / "data" / "flag_episode_eval.json"


def test_a_recorded_flag_episode_is_full_progress():
    """The point of the whole feature: reaching the flagpole must read as 1.0."""
    episode = json.loads(FIXTURE.read_text())["stages"][0]["episodes_detail"][0]
    assert episode["flag_get"] is True

    flag_x = eval_stages.load_stage_lengths()["1-1"]
    assert flag_x is not None, "1-1 has recorded flag episodes, so stages.json must have measured it"
    assert eval_stages.progress_fraction(episode["x_pos"], flag_x) == 1.0


def test_harvest_reads_the_flagpole_from_a_recorded_episode():
    found = harvest(str(FIXTURE))
    assert set(found) == {"1-1"}
    e = entry("1-1", found["1-1"])
    assert e["flag_x"] == 3161
    assert "RAM" in e["method"]


def test_unmeasured_stages_stay_none_rather_than_guessed():
    assert eval_stages.progress_fraction(1234, None) is None
    e = entry("2-1", [])
    assert e["flag_x"] is None and e["method"] == "not measured"


def test_disagreeing_episodes_are_refused_not_averaged():
    records = [{"x_pos": 3161, "seed": 1, "checkpoint": "a", "source": "a"},
               {"x_pos": 2900, "seed": 2, "checkpoint": "a", "source": "a"}]
    e = entry("1-1", records)
    assert e["flag_x"] is None, "two different values cannot both be the flagpole; averaging them would hide it"


def test_stages_json_covers_every_stage_and_holds_no_guesses():
    lengths = eval_stages.load_stage_lengths()
    from env.tiles import ALL_STAGES

    assert set(lengths) == set(ALL_STAGES)
    measured = {s: x for s, x in lengths.items() if x is not None}
    assert measured, "no stage lengths measured at all"
    assert all(isinstance(x, int) and x > 0 for x in measured.values())

    stages = json.loads(eval_stages.STAGES_JSON.read_text())["stages"]
    for stage, e in stages.items():
        assert e["method"], f"{stage} has no method note"
        if e["flag_x"] is not None:
            assert e["episodes"] >= 1, f"{stage} claims a length with no supporting episode"


def test_progress_fraction_is_a_ratio():
    assert eval_stages.progress_fraction(1580.5, 3161) == 0.5
    assert eval_stages.progress_fraction(0, 3161) == 0.0
