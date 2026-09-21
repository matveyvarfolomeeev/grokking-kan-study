#!/usr/bin/env python
"""Aggregate stage-3 controls without hiding failed or ambiguous seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import LEARNING_DYNAMICS, canonical_learning_dynamics  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures/stage3_controls"))
    args = parser.parse_args()
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for item in args.input:
        label, raw_dir = item.split("=", 1)
        for summary_path in sorted(Path(raw_dir).glob("*/summary.json")):
            run_dir = summary_path.parent
            summary = read_json(summary_path)
            metadata = read_json(run_dir / "metadata.json")
            config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
            phase = summary.get("phase_metrics", {})
            if metadata["model"]["type"] == "spline_kan":
                model_variant = (
                    f"g{config['model'].get('grid_size', 5)}_"
                    f"o{config['model'].get('spline_order', 3)}_"
                    f"w{config['model'].get('width', 14)}"
                )
            else:
                model_variant = str(config["model"].get("activation", "unknown"))
            rows.append(
                {
                    "control": label,
                    "run": run_dir.name,
                    "model": metadata["model"]["type"],
                    "model_variant": model_variant,
                    "p": metadata["dataset"]["p"],
                    "input_encoding": metadata["dataset"].get("input_encoding", "one_hot"),
                    "split_mode": metadata["dataset"].get("split_mode", "random"),
                    "train_fraction": metadata["dataset"]["train_fraction_realized"],
                    "seed": metadata["seed"],
                    "parameters": metadata["model"]["parameters"],
                    "dynamics": canonical_learning_dynamics(summary),
                    "is_classical_grokking": bool(phase.get("is_classical_grokking", False)),
                    "memorization_step": phase.get("memorization_step"),
                    "generalization_step": phase.get("generalization_step"),
                    "generalization_delay": phase.get("generalization_delay"),
                    "final_test_accuracy": summary["final"]["test_accuracy"],
                }
            )
    if not rows:
        raise SystemExit("No completed stage-3 runs found")
    write_csv(args.tables_dir / "stage3_control_runs.csv", rows)

    groups: dict[tuple[str, str, str, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["control"]),
                str(row["model"]),
                str(row["model_variant"]),
                round(float(row["train_fraction"]), 2),
            )
        ].append(row)
    summaries: list[dict[str, object]] = []
    for (control, model, model_variant, fraction), group in sorted(groups.items()):
        dynamics = Counter(str(row["dynamics"]) for row in group)
        classified = sum(dynamics[name] for name in LEARNING_DYNAMICS)
        if classified != len(group):
            raise ValueError(
                f"Phase counts do not match seed count for {control}/{model}: "
                f"{classified} != {len(group)}"
            )
        accuracies = [float(row["final_test_accuracy"]) for row in group]
        delays = [
            float(row["generalization_delay"])
            for row in group
            if row["generalization_delay"] is not None
        ]
        std = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
        summaries.append(
            {
                "control": control,
                "model": model,
                "model_variant": model_variant,
                "train_fraction": fraction,
                "seeds": len(group),
                **{name: dynamics[name] for name in LEARNING_DYNAMICS},
                "test_accuracy_mean": statistics.mean(accuracies),
                "test_accuracy_ci95": 1.96 * std / math.sqrt(len(accuracies)),
                "classical_delay_median": statistics.median(delays) if delays else "",
            }
        )
    write_csv(args.tables_dir / "stage3_control_summary.csv", summaries)

    figure, axis = plt.subplots(figsize=(8.3, 4.8))
    circular = [row for row in summaries if row["control"] == "circular"]
    for model in sorted({str(row["model"]) for row in circular}):
        selected = [row for row in circular if row["model"] == model]
        axis.errorbar(
            [float(row["train_fraction"]) for row in selected],
            [float(row["test_accuracy_mean"]) for row in selected],
            yerr=[float(row["test_accuracy_ci95"]) for row in selected],
            marker="o",
            capsize=4,
            label=model,
        )
    axis.set(
        xlabel="train fraction",
        ylabel="final test accuracy",
        ylim=(0, 1.02),
        title="Circular encoding intervention",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.figures_dir / "circular_accuracy.png", dpi=180)
    plt.close(figure)

    dynamics_order = [
        "classical_grokking",
        "weak_delayed_generalization",
        "joint_generalization",
        "memorized_not_generalized",
        "not_memorized",
    ]
    labels = [
        f"{row['control']}\n{row['model']} {row['model_variant']} f={row['train_fraction']:.2f}"
        for row in summaries
    ]
    bottoms = [0] * len(summaries)
    figure, axis = plt.subplots(figsize=(max(9, len(labels) * 0.85), 5.2))
    for dynamics_name in dynamics_order:
        values = [int(row[dynamics_name]) for row in summaries]
        axis.bar(labels, values, bottom=bottoms, label=dynamics_name)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    axis.set(ylabel="seed count", title="All stage-3 outcomes (no seed filtering)")
    axis.tick_params(axis="x", rotation=30)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(args.figures_dir / "control_dynamics.png", dpi=180)
    plt.close(figure)
    print("Saved stage-3 control tables and figures")


if __name__ == "__main__":
    main()
