"""Run the context-window + modality grid over seeds and summarize test metrics.

Run from scripts/:  python3 run_experiments.py [--only w1 w6] [--seeds 0 1 2] [--dry-run]
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from typing import List

import numpy as np

CONFIG = "../src/configs/hubert_dialogue.py"
OUT_DIR = "../experiments"
METRICS = ["ua", "wa", "macro_f1", "weighted_f1"]

EXPERIMENTS = {
    "w1": {"context_window": 1},  # no-context baseline, same encoder/fusion/training as the rest
    "w3": {"context_window": 3},
    "w6": {"context_window": 6},
    "w_full": {"context_window": None},
    "w6_text_only": {"context_window": 6, "ablate_audio": True},
    "w6_audio_only": {"context_window": 6, "ablate_text": True},
}


def build_command(exp: str, seed: int, results_path: str) -> List[str]:
    overrides = {**EXPERIMENTS[exp], "seed": seed, "name": f"{exp}_seed{seed}", "results_path": results_path}
    return [sys.executable, "train.py", "-cfg", CONFIG, "--set"] + [f"{k}={v}" for k, v in overrides.items()]


def matches(result: dict, exp: str, seed: int) -> bool:
    """True if a results.json was produced by this experiment's settings and seed."""
    expected = {"ablate_audio": False, "ablate_text": False, **EXPERIMENTS[exp], "seed": seed}
    return all(result.get(key) == value for key, value in expected.items())


def _parse_name(name: str):
    exp, _, seed = name[: -len(".json")].rpartition("_seed")
    return exp, int(seed) if seed.isdigit() else None


def is_done(path: str, exp: str, seed: int) -> bool:
    if not os.path.exists(path):
        return False
    with open(path) as f:
        return matches(json.load(f), exp, seed)


def load_results(runs_dir: str) -> List[dict]:
    """Read <exp>_seed<k>.json files, skipping anything whose contents don't match its name."""
    results = []
    for name in sorted(os.listdir(runs_dir)):
        if not name.endswith(".json"):
            continue
        exp, seed = _parse_name(name)
        if exp not in EXPERIMENTS or seed is None:
            print(f"ignoring {name}: not <experiment>_seed<k>.json")
            continue
        with open(os.path.join(runs_dir, name)) as f:
            result = json.load(f)
        if not matches(result, exp, seed):
            print(f"ignoring {name}: its settings don't match {exp} seed {seed}")
            continue
        results.append({"experiment": exp, **result})
    return results


def summarize(results: List[dict]) -> List[dict]:
    rows = []
    for exp in dict.fromkeys(r["experiment"] for r in results):
        runs = [r for r in results if r["experiment"] == exp]
        row = {"experiment": exp, "n": len(runs)}
        for m in METRICS:
            vals = np.array([r["test"][m] for r in runs]) * 100
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return rows


def write_summary(rows: List[dict]):
    with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| Experiment | n | UA | WA | Macro-F1 | Weighted-F1 |", "|---|---|---|---|---|---|"]
    for r in rows:
        cells = [f"{r[f'{m}_mean']:.2f} ± {r[f'{m}_std']:.2f}" for m in METRICS]
        lines.append(f"| {r['experiment']} | {r['n']} | " + " | ".join(cells) + " |")
    with open(os.path.join(OUT_DIR, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    runs_dir = os.path.join(OUT_DIR, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    for exp in args.only:
        for seed in args.seeds:
            path = os.path.join(runs_dir, f"{exp}_seed{seed}.json")
            if is_done(path, exp, seed):
                print(f"skip {exp} seed {seed} (done)")
                continue
            if os.path.exists(path):
                print(f"rerunning {exp} seed {seed}: {path} has other settings")
            cmd = build_command(exp, seed, path)
            print(" ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)

    results = load_results(runs_dir)
    if results:
        write_summary(summarize(results))


if __name__ == "__main__":
    main()
