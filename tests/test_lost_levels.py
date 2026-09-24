"""Lost Levels as a candidate final test set.

The current held-out five have been used for model selection, so they are a validation set and the headline
numbers are optimistically biased. Lost Levels is 16 usable real levels the agent has never touched. Before
any of them can be evaluated, our RAM map has to be valid there -- if it silently were not, the resulting
numbers would look plausible and mean nothing.
"""
from scripts.probe_lost_levels import LOST_LEVELS_WORLDS, STAGES, probe


def test_only_worlds_one_to_four_are_claimed():
    """gym_super_mario_bros refuses lost-levels worlds 5-12, so the test set is 16 stages, not 32."""
    assert list(LOST_LEVELS_WORLDS) == [1, 2, 3, 4]
    assert len(STAGES) == 16


def test_our_x_pos_addresses_are_valid_on_lost_levels():
    result = probe(1, 1)
    assert result["error"] is None, result["error"]
    assert result["loads"]
    assert result["x_pos_matches"], "0x6D/0x86 disagreed with the game's own x_pos"
    assert result["has_ground"], "the metatile buffer decoded to nothing solid"
    assert result["moves_right"]


def test_an_unsupported_world_is_reported_not_silently_wrong():
    result = probe(5, 1)
    assert result["error"] is not None
    assert not result["loads"]
