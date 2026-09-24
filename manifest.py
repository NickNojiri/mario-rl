"""Provenance for a training run: which code, which settings, which machine, and when.

Written into the run directory at train start as `manifest.json`, so a checkpoint months later can still be
traced back to the exact commit and environment that produced it. A run whose manifest says `"dirty": true` was
trained from uncommitted code and its numbers are not reproducible from the SHA alone.

Nothing here may raise: a run must never fail because provenance could not be collected. Every collector
returns None or an {"error": ...} record instead.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
MAX_DIRTY_FILES = 50


def _git(*args: str) -> str | None:
    try:
        done = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def git_state() -> dict:
    """Commit, branch, and whether the working tree had uncommitted changes."""
    sha = _git("rev-parse", "HEAD")
    if sha is None:
        return {"sha": None, "dirty": None, "note": "git unavailable or not a checkout"}
    status = _git("status", "--porcelain") or ""
    changed = [line[3:] for line in status.splitlines() if line.strip()]
    return {
        "sha": sha,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(changed),
        "dirty_files": changed[:MAX_DIRTY_FILES],
        "dirty_file_count": len(changed),
    }


def pip_freeze() -> list[str] | dict:
    """`pip freeze`, falling back to importlib metadata if pip is missing or slow."""
    try:
        done = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120)
        if done.returncode == 0:
            return done.stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        from importlib.metadata import distributions

        return sorted(f"{d.name}=={d.version}" for d in distributions() if d.name)
    except Exception as exc:  # provenance is never worth failing a run over
        return {"error": repr(exc)}


def cpu_info() -> dict:
    model = platform.processor() or None
    try:  # /proc/cpuinfo is the only place with the real model string under WSL
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"model": model, "logical_cores": os.cpu_count(), "platform": platform.platform(),
            "python": sys.version.split()[0]}


def build_manifest(cfg, run_dir, *, extra: dict | None = None, include_packages: bool = True) -> dict:
    return {
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "start_time_unix": time.time(),
        "run_dir": str(run_dir),
        "argv": list(sys.argv),
        "git": git_state(),
        "seed": getattr(cfg, "seed", None),
        "learning_hash": cfg.learning_hash(),
        "config": cfg.to_dict(),
        "cpu": cpu_info(),
        "packages": pip_freeze() if include_packages else None,
        **(extra or {}),
    }


def write_manifest(cfg, run_dir, *, resumed_from_step: int | None = None, extra: dict | None = None,
                   include_packages: bool = True) -> Path:
    """Write the manifest and return its path. Never clobbers an existing one.

    A resume gets its own `manifest_resume_<step>.json`, so the original run's provenance survives.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    name = "manifest.json" if resumed_from_step is None else f"manifest_resume_{resumed_from_step:09d}.json"
    path = run_dir / name
    if path.exists():
        path = run_dir / f"{path.stem}_{int(time.time())}.json"
    manifest = build_manifest(cfg, run_dir, extra=extra, include_packages=include_packages)
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path
