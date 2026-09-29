"""The long-run supervisor's two Python pieces: log rollback after a crash, and the divergence health check."""
import csv
import sys
from pathlib import Path

from train_ppo import roll_back_rows

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from run_health import check  # noqa: E402


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def read_steps(path):
    with open(path, newline="") as f:
        return [int(r["step"]) for r in csv.DictReader(f)]


def test_rows_past_the_checkpoint_move_aside(tmp_path):
    path = tmp_path / "updates.csv"
    write_csv(path, ["update", "step"], [[1, 100], [2, 200], [3, 300], [4, 400]])
    assert roll_back_rows(path, lambda s: s <= 200) == 2
    assert read_steps(path) == [100, 200]
    assert read_steps(tmp_path / "updates_rolled_back.csv") == [300, 400]


def test_second_crash_appends_to_the_same_side_file(tmp_path):
    path = tmp_path / "episodes.csv"
    write_csv(path, ["step", "x_pos"], [[100, 5], [200, 6]])
    roll_back_rows(path, lambda s: s < 200)
    with open(path, "a", newline="") as f:
        csv.writer(f).writerows([[200, 7], [300, 8]])
    roll_back_rows(path, lambda s: s < 250)
    assert read_steps(path) == [100, 200]
    assert read_steps(tmp_path / "episodes_rolled_back.csv") == [200, 300]


def test_nothing_to_roll_back_leaves_files_untouched(tmp_path):
    path = tmp_path / "updates.csv"
    write_csv(path, ["update", "step"], [[1, 100]])
    before = path.read_bytes()
    assert roll_back_rows(path, lambda s: s <= 100) == 0
    assert path.read_bytes() == before
    assert not (tmp_path / "updates_rolled_back.csv").exists()
    assert roll_back_rows(tmp_path / "missing.csv", lambda s: True) == 0


HEADER = ["step", "steps_per_sec", "mean_x_pos", "policy_loss", "value_loss", "entropy", "approx_kl"]


def test_health_ok_and_diverged(tmp_path):
    write_csv(tmp_path / "updates.csv", HEADER, [[100, 600, 500, 0.01, 20.0, 0.9, 0.01]])
    code, msg = check(tmp_path)
    assert code == 0 and "step=100" in msg
    write_csv(tmp_path / "updates.csv", HEADER, [[100, 600, 500, 0.01, 20.0, 0.9, 0.01],
                                                 [200, 600, 500, "nan", "inf", 0.9, 0.01]])
    code, msg = check(tmp_path)
    assert code == 2 and "policy_loss" in msg and "value_loss" in msg


def test_health_ignores_a_row_still_being_written(tmp_path):
    path = tmp_path / "updates.csv"
    write_csv(path, HEADER, [[100, 600, 500, 0.01, 20.0, 0.9, 0.01]])
    with open(path, "a") as f:
        f.write("200,600,5")  # the trainer is mid-write: no newline yet
    code, msg = check(tmp_path)
    assert code == 0 and "step=100" in msg


def test_health_with_nothing_logged_yet(tmp_path):
    assert check(tmp_path)[0] == 1
    write_csv(tmp_path / "updates.csv", HEADER, [])
    assert check(tmp_path)[0] == 1
