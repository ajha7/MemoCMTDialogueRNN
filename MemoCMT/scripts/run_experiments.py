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
            if os.path.exists(path):
                print(f"skip {exp} seed {seed} (done)")
                continue
            cmd = build_command(exp, seed, path)
            print(" ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)

    results = []
    for name in sorted(os.listdir(runs_dir)):
        with open(os.path.join(runs_dir, name)) as f:
            results.append({"experiment": name.rsplit("_seed", 1)[0], **json.load(f)})
    if results:
        write_summary(summarize(results))


if __name__ == "__main__":
    main()
