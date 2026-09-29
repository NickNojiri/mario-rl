"""Health check for a running PPO run, used by scripts/week_run.sh every few minutes.

    python scripts/run_health.py runs/week1

Prints one status line from the last row of updates.csv. Exit code 2 means the last update logged a non-finite
loss: training has diverged and every further step is wasted, so the supervisor pauses the run. Exit 1 means there
is nothing to check yet (no updates.csv or no rows).
"""
from __future__ import annotations

import csv
import io
import math
import sys
from pathlib import Path

CHECKED = ("policy_loss", "value_loss", "entropy", "approx_kl")


def last_row(path: Path) -> dict | None:
    """The header and the final row of a CSV, without reading a file that grows to hundreds of MB."""
    with open(path, "rb") as f:
        header = f.readline().decode()
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 65536))
        tail = f.read().decode(errors="replace").split("\n")
    tail = tail[:-1]  # "" after a final newline, or a row still being written: never judge a half-written row
    rows = [line.rstrip("\r") for line in tail if line.strip() and line.strip() != header.strip()]
    if not rows:
        return None
    return next(csv.DictReader(io.StringIO(header + rows[-1] + "\n")))


def check(run_dir: Path) -> tuple[int, str]:
    path = Path(run_dir) / "updates.csv"
    if not path.exists():
        return 1, f"no {path}"
    row = last_row(path)
    if row is None:
        return 1, f"no rows in {path}"
    bad = []
    for k in CHECKED:
        try:
            if not math.isfinite(float(row[k])):
                bad.append(k)
        except (KeyError, ValueError):
            bad.append(k)
    status = (f"step={row.get('step')} steps_per_sec={row.get('steps_per_sec')} x={row.get('mean_x_pos')} "
              f"entropy={row.get('entropy')} value_loss={row.get('value_loss')}")
    if bad:
        return 2, f"DIVERGED ({', '.join(bad)} not finite) {status}"
    return 0, status


if __name__ == "__main__":
    code, msg = check(Path(sys.argv[1]))
    print(msg, flush=True)
    sys.exit(code)
