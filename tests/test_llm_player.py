"""A pretrained model as a zero-shot player. The model itself is stubbed; what is tested is that the screen is
described correctly, answers map to the right action, and the player is scored with the standard protocol."""
import numpy as np
import pytest

import eval_stages
from agent import llm_player
from config import get_ppo_preset
from env.tiles import ENEMY_BASE, MARIO_ID, UNKNOWN_ID, TileMarioEnv, policy_action_names
from tests.test_context import _ram


def test_tiles_become_readable_characters():
    assert [llm_player.tile_char(v) for v in (0, 0x54, 0x12, 0xC0, ENEMY_BASE + 6, MARIO_ID, UNKNOWN_ID)] == \
        [".", "#", "P", "Q", "E", "M", "?"]


def test_the_text_map_is_the_screen_one_character_per_tile():
    env = TileMarioEnv(stages=["1-1"], noop_max=0)
    obs, _ = env.reset(seed=0)
    text = llm_player.text_map(obs["tiles"][-1])
    rows = text.split("\n")
    assert len(rows) == 13 and all(len(r) == 16 for r in rows)
    assert text.count("M") == 1
    assert set(rows[11]) == {"#"} and set(rows[12]) == {"#"}, "1-1 starts on two full rows of ground"
    env.close()


def test_physics_are_described_in_words():
    near_pit = llm_player.describe_physics(_ram(speed=24, pit_cols=(5, 6)))
    assert "Next pit starts" in near_pit and "walk off" in near_pit
    flat = llm_player.describe_physics(_ram(speed=0))
    assert "No pit within 10 tiles" in flat and "solid ground" in flat


def test_action_labels_are_single_tokens():
    assert llm_player.action_labels(policy_action_names("simple")) == \
        ["NOOP", "RIGHT", "RIGHT_A", "RIGHT_B", "RIGHT_A_B", "A", "LEFT"]


def test_backend_specs():
    with pytest.raises(ValueError):
        llm_player.make_backend("ollama:")
    with pytest.raises(ValueError):
        llm_player.make_backend("gpt")
    with pytest.raises(NotImplementedError):
        llm_player.make_backend("jev")


class _Stub:
    """Always answers RIGHT; records every prompt it was shown."""

    def __init__(self):
        self.latencies, self.prompts = [], []

    def choose(self, prompt, labels):
        self.prompts.append(prompt)
        self.latencies.append(0.0)
        return "RIGHT"


def test_the_llm_player_is_scored_with_the_standard_protocol(monkeypatch):
    stub = _Stub()
    monkeypatch.setattr(llm_player, "make_backend", lambda spec: stub)
    job = {"stage": "1-1", "group": "test", "config": get_ppo_preset("ppo_full").to_dict(), "episodes": 1,
           "seed": 50_000, "greedy": False, "checkpoint": None, "mode": "safe", "llm": "stub", "flag_x": None}
    row = eval_stages._eval_stage(job)
    ep = row["episodes_detail"][0]
    assert row["llm_decisions"] == ep["length"] == len(stub.prompts), "one model call per agent decision"
    assert ep["left_presses"] == 0 and row["mean_x_pos"] > 0, "every RIGHT was actually played"
    assert "Screen:" in stub.prompts[0] and "Last button choice: none" in stub.prompts[0]
    assert "Last button choice: RIGHT" in stub.prompts[1]


def _ollama_up():
    try:
        return llm_player.OllamaBackend("llama3.2:1b")
    except Exception:
        return None


@pytest.mark.skipif(_ollama_up() is None, reason="no local Ollama server")
def test_live_ollama_answers_with_a_valid_action():
    backend = llm_player.OllamaBackend("llama3.2:1b")
    labels = llm_player.action_labels(policy_action_names("simple"))
    grid = np.zeros((13, 16), np.int16)
    grid[11:] = 0x54
    grid[10, 3] = MARIO_ID
    answer = backend.choose(llm_player.build_prompt(grid, _ram(), labels, None), labels)
    assert answer in labels
    assert backend.info()["digest"], "the results file must record which model weights were used"
