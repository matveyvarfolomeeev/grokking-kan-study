#!/usr/bin/env python
"""Aggregate the preregistered paired phase map without filtering seeds."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import LEARNING_DYNAMICS, canonical_learning_dynamics  # noqa: E402
from src.study import wilson_interval, write_csv  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _label(row: dict[str, Any]) -> str:
    return (
        f"{row['model'].upper()} p={row['p']} f={float(row['train_fraction']):.2f} "
        f"wd={float(row['weight_decay']):g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    manifest = _read_json(args.input_dir / "study_manifest.json")
    entries = manifest["runs"]
    if len(entries) != int(manifest["expected_runs"]):
        raise ValueError("Study manifest is incomplete")

    rows: list[dict[str, Any]] = []
    for entry in entries:
        if entry["status"] not in {"completed", "skipped_existing"}:
            raise ValueError(f"Run is not complete: {entry['run']} ({entry['status']})")
        run_dir = Path(entry["run_dir"])
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        summary = _read_json(run_dir / "summary.json")
        metadata = _read_json(run_dir / "metadata.json")
        config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        phase = summary["phase_metrics"]
        last_step = int(summary["final"]["step"])
        memorization = phase.get("memorization_step")
        generalization = phase.get("generalization_step")
        rows.append(
            {
                "run": entry["run"],
                "model": entry["model"],
                "p": int(entry["p"]),
                "train_fraction": float(entry["train_fraction"]),
                "train_fraction_realized": metadata["dataset"]["train_fraction_realized"],
                "weight_decay": float(entry["weight_decay"]),
                "seed": int(entry["seed"]),
                "parameters": metadata["model"]["parameters"],
                "max_steps": int(config["training"]["max_steps"]),
                "dynamics": canonical_learning_dynamics(summary),
                "is_classical_grokking": bool(phase["is_classical_grokking"]),
                "memorization_step": memorization,
                "generalization_step": generalization,
                "generalization_delay": phase.get("generalization_delay"),
                "test_accuracy_at_memorization": phase.get("test_accuracy_at_memorization"),
                "accuracy_gap_at_memorization": phase.get("accuracy_gap_at_memorization"),
                "plateau_duration_steps": phase.get("plateau_duration_steps"),
                "final_train_accuracy": summary["final"]["train_accuracy"],
                "final_test_accuracy": summary["final"]["test_accuracy"],
                "censored": generalization is None,
                "observed_or_censor_delay": (
                    int(generalization - memorization)
                    if generalization is not None and memorization is not None
                    else int(last_step - memorization)
                    if memorization is not None
                    else last_step
                ),
            }
        )

    write_csv(args.tables_dir / "paired_runs.csv", rows)
    groups: dict[tuple[str, int, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["model"]),
                int(row["p"]),
                float(row["train_fraction"]),
                float(row["weight_decay"]),
            )
        ].append(row)

    summaries: list[dict[str, Any]] = []
    for (model, p, fraction, decay), group in sorted(groups.items()):
        counts = Counter(str(row["dynamics"]) for row in group)
        if sum(counts[name] for name in LEARNING_DYNAMICS) != len(group):
            raise ValueError(f"Incomplete phase counts for {model}/p{p}/f{fraction}/wd{decay}")
        summary_row: dict[str, Any] = {
            "model": model,
            "p": p,
            "train_fraction": fraction,
            "weight_decay": decay,
            "seeds": len(group),
            "parameters": group[0]["parameters"],
        }
        for name in LEARNING_DYNAMICS:
            low, high = wilson_interval(counts[name], len(group))
            summary_row[f"{name}_count"] = counts[name]
            summary_row[f"{name}_rate"] = counts[name] / len(group)
            summary_row[f"{name}_ci95_low"] = low
            summary_row[f"{name}_ci95_high"] = high
        delays = [
            int(row["generalization_delay"])
            for row in group
            if row["generalization_delay"] is not None
        ]
        summary_row.update(
            {
                "generalized_count": len(delays),
                "censored_count": len(group) - len(delays),
                "delay_median": statistics.median(delays) if delays else "",
                "delay_min": min(delays) if delays else "",
                "delay_max": max(delays) if delays else "",
                "final_test_accuracy_median": statistics.median(
                    float(row["final_test_accuracy"]) for row in group
                ),
            }
        )
        summaries.append(summary_row)
    write_csv(args.tables_dir / "paired_summary.csv", summaries)

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    labels = [_label(row) for row in summaries]
    figure, axis = plt.subplots(figsize=(max(11, len(labels) * 0.9), 5.8))
    bottoms = [0] * len(summaries)
    colors = ["#18864b", "#f59e0b", "#64748b", "#dc2626", "#111827"]
    phase_labels = {
        "classical_grokking": "классический grokking",
        "weak_delayed_generalization": "слабая задержка",
        "joint_generalization": "совместное обобщение",
        "memorized_not_generalized": "запомнила без обобщения",
        "not_memorized": "не запомнила",
    }
    for name, color in zip(LEARNING_DYNAMICS, colors):
        values = [int(row[f"{name}_count"]) for row in summaries]
        axis.bar(labels, values, bottom=bottoms, label=phase_labels[name], color=color)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    axis.set(ylabel="число seed", title="Парная фазовая карта: все зафиксированные запуски")
    axis.tick_params(axis="x", rotation=45, labelsize=8)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(args.figures_dir / "paired_phase_map.png", dpi=200)
    plt.close(figure)

    phase_matrix = [[int(row[f"{name}_count"]) for name in LEARNING_DYNAMICS] for row in summaries]
    figure, axis = plt.subplots(figsize=(10.2, 6.5))
    image = axis.imshow(phase_matrix, cmap="Blues", vmin=0, vmax=5, aspect="auto")
    for row_index, values in enumerate(phase_matrix):
        for column_index, value in enumerate(values):
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value >= 3 else "#111827",
                fontsize=9,
                fontweight="bold" if value else "normal",
            )
    axis.set_xticks(
        range(len(LEARNING_DYNAMICS)),
        ("classical", "weak delay", "joint", "memorized", "not memorized"),
        rotation=25,
        ha="right",
    )
    axis.set_yticks(range(len(summaries)), [_label(row) for row in summaries], fontsize=8)
    axis.set(title="Фазовая карта: число seed из пяти")
    figure.colorbar(image, ax=axis, label="число seed", fraction=0.03, pad=0.03)
    figure.tight_layout()
    figure.savefig(args.figures_dir / "paired_phase_heatmap.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True)
    for axis, model in zip(axes, ("kan", "mlp")):
        ordered_keys = [key for key in sorted(groups) if key[0] == model]
        for x, key in enumerate(ordered_keys):
            group = groups[key]
            observed = [row for row in group if not row["censored"]]
            censored = [row for row in group if row["censored"]]
            if observed:
                values = [float(row["generalization_delay"]) for row in observed]
                offsets = [(index - (len(values) - 1) / 2) * 0.055 for index in range(len(values))]
                axis.scatter(
                    [x + offset for offset in offsets],
                    values,
                    color="#2563eb",
                    s=28,
                    zorder=3,
                )
                median = statistics.median(values)
                axis.plot([x - 0.25, x + 0.25], [median, median], color="#ea580c", linewidth=2)
            if censored:
                values = [float(row["observed_or_censor_delay"]) for row in censored]
                offsets = [(index - (len(values) - 1) / 2) * 0.055 for index in range(len(values))]
                axis.scatter(
                    [x + offset for offset in offsets],
                    values,
                    marker="v",
                    color="#dc2626",
                    s=34,
                    zorder=3,
                )
        axis.set_xticks(range(len(ordered_keys)))
        axis.set_xticklabels(
            [f"p={key[1]} f={key[2]:.2f}\nwd={key[3]:g}" for key in ordered_keys],
            fontsize=8,
        )
        axis.set(title=model.upper(), xlabel="условие")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("задержка или время цензурирования, шаги")
    figure.suptitle("Задержка обобщения; красные треугольники - незавершённые запуски")
    figure.tight_layout()
    figure.savefig(args.figures_dir / "paired_delays.png", dpi=200)
    plt.close(figure)
    print(f"Saved {len(rows)} runs and {len(summaries)} paired conditions")


if __name__ == "__main__":
    main()
