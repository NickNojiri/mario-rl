"""Run provenance. A manifest that silently lost a field would only be noticed months later."""
import json
import re

import manifest
from config import PPOConfig, get_ppo_preset

REQUIRED = {"start_time", "start_time_unix", "run_dir", "argv", "git", "seed", "learning_hash", "config", "cpu",
            "packages"}


def test_manifest_has_every_required_field():
    cfg = get_ppo_preset("ppo_smoke", seed=1234)
    m = manifest.build_manifest(cfg, "runs/whatever", include_packages=False)
    assert REQUIRED <= set(m)
    assert m["seed"] == 1234 == m["config"]["seed"]
    assert m["config"] == cfg.to_dict(), "the full config has to round-trip, not a summary of it"
    assert m["learning_hash"] == cfg.learning_hash()


def test_git_state_is_a_sha_and_a_dirty_flag():
    g = manifest.git_state()
    if g["sha"] is None:  # not a checkout; the run must still start
        assert g["dirty"] is None
    else:
        assert re.fullmatch(r"[0-9a-f]{40}", g["sha"])
        assert isinstance(g["dirty"], bool)


def test_cpu_info_reports_cores():
    cpu = manifest.cpu_info()
    assert isinstance(cpu["logical_cores"], int) and cpu["logical_cores"] >= 1
    assert cpu["python"].count(".") == 2


def test_pip_freeze_lists_a_package_we_depend_on():
    pkgs = manifest.pip_freeze()
    assert isinstance(pkgs, list) and pkgs
    assert any(p.lower().startswith("torch==") for p in pkgs)


def test_write_manifest_is_valid_json_and_never_clobbers(tmp_path):
    cfg = get_ppo_preset("ppo_smoke")
    first = manifest.write_manifest(cfg, tmp_path, include_packages=False)
    assert first.name == "manifest.json"
    # JSON has no tuples, so stage lists come back as lists; what matters is that the config reconstructs.
    assert PPOConfig.from_dict(json.loads(first.read_text())["config"]) == cfg

    second = manifest.write_manifest(cfg, tmp_path, include_packages=False)
    assert second != first and second.exists() and first.exists()


def test_resume_gets_its_own_manifest(tmp_path):
    cfg = get_ppo_preset("ppo_smoke")
    manifest.write_manifest(cfg, tmp_path, include_packages=False)
    resumed = manifest.write_manifest(cfg, tmp_path, resumed_from_step=7_765_248, include_packages=False)
    assert resumed.name == "manifest_resume_007765248.json"
    assert json.loads(resumed.read_text())["start_time_unix"] > 0
