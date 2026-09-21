#!/usr/bin/env python
"""Create accuracy and loss figures from a saved run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import correct_class_score_table, fourier_power  # noqa: E402


def load_history(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.run_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    history = load_history(args.run_dir / "history.jsonl")
    summary_path = args.run_dir / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    )
    steps = [row["step"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(steps, [row["train_accuracy"] for row in history], label="train")
    axes[0].plot(steps, [row["test_accuracy"] for row in history], label="test")
    axes[0].set(title="Точность", xlabel="шаг", ylabel="accuracy")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend()
    axes[1].plot(steps, [row["train_loss"] for row in history], label="train")
    axes[1].plot(steps, [row["test_loss"] for row in history], label="test")
    axes[1].set(title="MSE", xlabel="шаг", ylabel="loss")
    axes[1].legend()
    if summary is not None and "phase_metrics" in summary:
        phase = summary["phase_metrics"]
        for axis in axes:
            if phase.get("memorization_step") is not None:
                axis.axvline(
                    phase["memorization_step"],
                    color="#2563eb",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.8,
                    label="запоминание",
                )
            if phase.get("generalization_step") is not None:
                axis.axvline(
                    phase["generalization_step"],
                    color="#15803d",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.8,
                    label="обобщение",
                )
        axes[0].legend()
        axes[1].legend()
    fig.tight_layout()
    figure_path = output_dir / "learning_curves.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    print(f"Saved figure to {figure_path}")

    predictions_path = args.run_dir / "predictions.pt"
    if predictions_path.exists():
        artifact = torch.load(predictions_path, map_location="cpu", weights_only=True)
        predictions = artifact["predictions"]
        p = int(predictions.shape[1])
        predicted_classes = predictions.argmax(dim=1).reshape(p, p)
        score_table = correct_class_score_table(predictions, p)
        power = fourier_power(score_table)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(predicted_classes, origin="lower", cmap="twilight", interpolation="nearest")
        axes[0].set(title="Predicted class", xlabel="b", ylabel="a")
        axes[1].imshow(torch.log10(power + 1e-12), origin="lower", cmap="magma")
        axes[1].set(title="2-D Fourier power (log10)", xlabel="frequency b", ylabel="frequency a")
        fig.tight_layout()
        spectral_path = output_dir / "solution_and_fourier.png"
        fig.savefig(spectral_path, dpi=160)
        plt.close(fig)
        print(f"Saved figure to {spectral_path}")

    snapshots_path = args.run_dir / "prediction_snapshots.pt"
    if snapshots_path.exists():
        snapshots = torch.load(snapshots_path, map_location="cpu", weights_only=True)
        names = [name for name in ("memorization", "generalization") if name in snapshots]
        if names:
            fig, axes = plt.subplots(len(names), 2, figsize=(9, 4 * len(names)), squeeze=False)
            for row, name in enumerate(names):
                predictions = snapshots[name]["predictions"]
                p = int(predictions.shape[1])
                step = int(snapshots[name]["step"])
                classes = predictions.argmax(dim=1).reshape(p, p)
                power = fourier_power(correct_class_score_table(predictions, p))
                axes[row, 0].imshow(
                    classes, origin="lower", cmap="twilight", interpolation="nearest"
                )
                axes[row, 0].set(title=f"{name}: predictions, step {step}", xlabel="b", ylabel="a")
                axes[row, 1].imshow(torch.log10(power + 1e-12), origin="lower", cmap="magma")
                axes[row, 1].set(
                    title=f"{name}: Fourier power", xlabel="frequency b", ylabel="frequency a"
                )
            fig.tight_layout()
            phases_path = output_dir / "phase_fourier_comparison.png"
            fig.savefig(phases_path, dpi=160)
            plt.close(fig)
            print(f"Saved figure to {phases_path}")

    if summary is not None:
        print(summary)


if __name__ == "__main__":
    main()
