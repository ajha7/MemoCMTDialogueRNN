import importlib.util
import json
import os

import pytest

spec = importlib.util.spec_from_file_location(
    "run_experiments", os.path.join(os.path.dirname(__file__), "..", "scripts", "run_experiments.py")
)
run_experiments = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_experiments)


def _r(exp, ua):
    return {"experiment": exp, "test": {"ua": ua, "wa": ua, "macro_f1": ua, "weighted_f1": ua}}


def test_summarize_mean_std_in_percent():
    rows = run_experiments.summarize([_r("w1", 0.70), _r("w1", 0.72), _r("w1", 0.74), _r("w6", 0.80)])
    w1 = next(r for r in rows if r["experiment"] == "w1")
    assert w1["n"] == 3
    assert w1["ua_mean"] == pytest.approx(72.0)
    assert w1["ua_std"] == pytest.approx(2.0)  # sample std (ddof=1)
    w6 = next(r for r in rows if r["experiment"] == "w6")
    assert w6["ua_std"] == 0.0


def test_command_passes_overrides():
    cmd = run_experiments.build_command("w6_text_only", 1, "experiments/runs/x.json")
    assert "context_window=6" in cmd and "ablate_audio=True" in cmd and "seed=1" in cmd


def _write(path, **fields):
    result = {"context_window": 6, "ablate_audio": False, "ablate_text": False, "seed": 0,
              "test": {"ua": 0.8, "wa": 0.8, "macro_f1": 0.8, "weighted_f1": 0.8}, **fields}
    path.write_text(json.dumps(result))


def test_results_whose_config_does_not_match_the_filename_are_not_counted(tmp_path):
    _write(tmp_path / "w6_seed0.json", context_window=3)  # stale: a w3 run saved under the w6 name
    _write(tmp_path / "w6_seed1.json", seed=1)
    (tmp_path / ".DS_Store").write_bytes(b"\x00\x01")
    results = run_experiments.load_results(str(tmp_path))
    assert [(r["experiment"], r["seed"]) for r in results] == [("w6", 1)]


def test_a_mismatched_file_does_not_count_as_done(tmp_path):
    _write(tmp_path / "w6_seed0.json", context_window=3)
    assert not run_experiments.is_done(str(tmp_path / "w6_seed0.json"), "w6", 0)
    _write(tmp_path / "w6_seed1.json", seed=1)
    assert run_experiments.is_done(str(tmp_path / "w6_seed1.json"), "w6", 1)
