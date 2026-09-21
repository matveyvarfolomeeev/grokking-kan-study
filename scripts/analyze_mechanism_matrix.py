#!/usr/bin/env python
"""Run and aggregate multi-seed mechanism analysis for the paired study."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.study import write_csv  # noqa: E402

PHASE_ORDER = ("memorization", "generalization", "final")
PHASE_LABELS = {
    "memorization": "запоминание",
    "generalization": "обобщение",
    "final": "финал",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _run_analysis(run_dir: Path, skip_kan_details: bool = False) -> Path:
    output_dir = run_dir / "mechanisms"
    command = [sys.executable, str(ROOT / "scripts" / "analyze_mechanisms.py"), str(run_dir)]
    if skip_kan_details:
        command.append("--skip-kan-details")
    subprocess.run(command, check=True)
    return output_dir


def _aggregate_phase_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["model"]), str(row["phase"]))].append(row)
    summaries: list[dict[str, Any]] = []
    for (model, phase), group in sorted(groups.items()):
        item: dict[str, Any] = {"model": model, "phase": phase, "seeds": len(group)}
        for metric in (
            "modular_diagonal_power_fraction",
            "hidden_pc1_variance",
            "hidden_pc2_variance",
            "hidden_participation_rank",
            "train_accuracy",
            "test_accuracy",
        ):
            values = [float(row[metric]) for row in group]
            item[f"{metric}_median"] = statistics.median(values)
            item[f"{metric}_min"] = min(values)
            item[f"{metric}_max"] = max(values)
        summaries.append(item)
    return summaries


def _aggregate_spectra(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["model"]),
            str(row["phase"]),
            int(row["frequency_a"]),
            int(row["frequency_b"]),
        )
        groups[key].append(row)
    summaries: list[dict[str, Any]] = []
    for (model, phase, frequency_a, frequency_b), group in sorted(groups.items()):
        values = [float(row["power_fraction"]) for row in group]
        summaries.append(
            {
                "model": model,
                "phase": phase,
                "frequency_a": frequency_a,
                "frequency_b": frequency_b,
                "seeds": len(group),
                "power_fraction_median": statistics.median(values),
                "power_fraction_min": min(values),
                "power_fraction_max": max(values),
                "on_modular_diagonal": frequency_a == frequency_b and frequency_a != 0,
            }
        )
    return summaries


def _plot_spectrum_heatmaps(rows: list[dict[str, Any]], output_path: Path) -> None:
    models = ("mlp", "kan")
    frequencies = sorted({int(row["frequency_a"]) for row in rows})
    index = {frequency: position for position, frequency in enumerate(frequencies)}
    figure, axes = plt.subplots(
        2, 3, figsize=(13.2, 7.6), squeeze=False, layout="constrained"
    )
    plotted = []
    for model_index, model in enumerate(models):
        for phase_index, phase in enumerate(PHASE_ORDER):
            axis = axes[model_index, phase_index]
            phase_rows = [
                row for row in rows if row["model"] == model and row["phase"] == phase
            ]
            if not phase_rows:
                axis.text(0.5, 0.5, "нет checkpoint", ha="center", va="center")
                axis.set_axis_off()
                continue
            spectrum = np.full((len(frequencies), len(frequencies)), np.nan)
            for row in phase_rows:
                spectrum[
                    index[int(row["frequency_a"])], index[int(row["frequency_b"])]
                ] = float(row["power_fraction_median"])
            image = axis.imshow(
                np.log10(np.clip(spectrum, 1e-10, None)),
                cmap="magma",
                origin="lower",
                extent=(
                    frequencies[0] - 0.5,
                    frequencies[-1] + 0.5,
                    frequencies[0] - 0.5,
                    frequencies[-1] + 0.5,
                ),
                vmin=-6.0,
                vmax=-1.0,
            )
            plotted.append(image)
            axis.plot(frequencies, frequencies, color="cyan", alpha=0.45, linewidth=0.8)
            axis.set_title(f"{model.upper()} · {PHASE_LABELS[phase]}")
            axis.set_xlabel("частота b")
            axis.set_ylabel("частота a")
    if plotted:
        figure.colorbar(
            plotted[-1],
            ax=axes.ravel().tolist(),
            label="log10 доли мощности",
            fraction=0.025,
            pad=0.02,
        )
    figure.suptitle("Медианная Fourier-структура по seed")
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _plot_phase_metric(
    rows: list[dict[str, Any]], metric: str, ylabel: str, output_path: Path
) -> None:
    models = sorted({str(row["model"]) for row in rows})
    figure, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 4.4), sharey=False)
    if len(models) == 1:
        axes = [axes]
    for axis, model in zip(axes, models):
        model_rows = [row for row in rows if row["model"] == model]
        for seed in sorted({int(row["seed"]) for row in model_rows}):
            seed_rows = {row["phase"]: row for row in model_rows if int(row["seed"]) == seed}
            phases = [phase for phase in PHASE_ORDER if phase in seed_rows]
            axis.plot(
                [PHASE_ORDER.index(phase) for phase in phases],
                [float(seed_rows[phase][metric]) for phase in phases],
                marker="o",
                linewidth=1,
                alpha=0.45,
                label=f"seed {seed}",
            )
        for phase_index, phase in enumerate(PHASE_ORDER):
            values = [float(row[metric]) for row in model_rows if row["phase"] == phase]
            if values:
                axis.scatter(
                    [phase_index],
                    [statistics.median(values)],
                    marker="D",
                    s=58,
                    color="#111827",
                    zorder=4,
                )
        axis.set_xticks(range(len(PHASE_ORDER)), [PHASE_LABELS[phase] for phase in PHASE_ORDER])
        axis.set(title=model.upper(), ylabel=ylabel)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _aggregate_ablation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(float(row["fraction"]), str(row["strategy"]))].append(row)
    summaries: list[dict[str, Any]] = []
    for (fraction, strategy), group in sorted(groups.items()):
        accuracies = [float(row["test_accuracy"]) for row in group]
        deltas = [float(row["test_accuracy_delta"]) for row in group]
        summaries.append(
            {
                "fraction": fraction,
                "strategy": strategy,
                "models": len({int(row["seed"]) for row in group}),
                "observations": len(group),
                "test_accuracy_median": statistics.median(accuracies),
                "test_accuracy_min": min(accuracies),
                "test_accuracy_max": max(accuracies),
                "test_accuracy_delta_median": statistics.median(deltas),
                "test_accuracy_delta_min": min(deltas),
                "test_accuracy_delta_max": max(deltas),
            }
        )
    return summaries


def _plot_ablation(rows: list[dict[str, Any]], output_path: Path) -> None:
    strategies = ("top", "random", "bottom")
    colors = {"top": "#dc2626", "random": "#64748b", "bottom": "#2563eb"}
    figure, axis = plt.subplots(figsize=(7.3, 4.8))
    for strategy in strategies:
        strategy_rows = [row for row in rows if row["strategy"] == strategy]
        for seed in sorted({int(row["seed"]) for row in strategy_rows}):
            points = []
            for fraction in sorted({float(row["fraction"]) for row in strategy_rows}):
                values = [
                    float(row["test_accuracy_delta"])
                    for row in strategy_rows
                    if int(row["seed"]) == seed and float(row["fraction"]) == fraction
                ]
                if values:
                    points.append((fraction, statistics.mean(values)))
            axis.plot(
                [point[0] * 100 for point in points],
                [point[1] for point in points],
                color=colors[strategy],
                alpha=0.22,
                linewidth=1,
            )
        fractions = sorted({float(row["fraction"]) for row in strategy_rows})
        medians = [
            statistics.median(
                float(row["test_accuracy_delta"])
                for row in strategy_rows
                if float(row["fraction"]) == fraction
            )
            for fraction in fractions
        ]
        axis.plot(
            [fraction * 100 for fraction in fractions],
            medians,
            marker="o",
            linewidth=2.5,
            color=colors[strategy],
            label=strategy,
        )
    axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
    axis.set(
        xlabel="удалённые рёбра, %",
        ylabel="изменение test accuracy",
        title="Абляция рёбер KAN на пяти независимо обученных моделях",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--symmetry-dir", type=Path)
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    manifest = json.loads((args.study_dir / "study_manifest.json").read_text(encoding="utf-8"))
    selected = [
        entry
        for entry in manifest["runs"]
        if int(entry["p"]) == 31
        and abs(float(entry["train_fraction"]) - 0.35) < 1e-9
        and abs(float(entry["weight_decay"]) - 0.3) < 1e-9
    ]
    if len(selected) != 10:
        raise ValueError(f"Expected ten mechanism runs, found {len(selected)}")

    phase_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    top_mode_rows: list[dict[str, Any]] = []
    fourier_runs_dir = args.figures_dir / "fourier_runs"
    pca_runs_dir = args.figures_dir / "pca_runs"
    edge_runs_dir = args.figures_dir / "kan_edge_functions"
    for directory in (fourier_runs_dir, pca_runs_dir, edge_runs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for entry in selected:
        run_dir = Path(entry["run_dir"])
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        output_dir = _run_analysis(run_dir)
        for row in _read_csv(output_dir / "phase_mechanisms.csv"):
            phase_rows.append(
                {"run": entry["run"], "model": entry["model"], "seed": entry["seed"], **row}
            )
        for row in _read_csv(output_dir / "fourier_spectrum.csv"):
            spectrum_rows.append(
                {"run": entry["run"], "model": entry["model"], "seed": entry["seed"], **row}
            )
        for row in _read_csv(output_dir / "fourier_top_modes.csv"):
            top_mode_rows.append(
                {"run": entry["run"], "model": entry["model"], "seed": entry["seed"], **row}
            )
        shutil.copyfile(
            output_dir / "fourier_by_phase.png", fourier_runs_dir / f"{entry['run']}.png"
        )
        shutil.copyfile(
            output_dir / "hidden_pca_by_phase.png", pca_runs_dir / f"{entry['run']}.png"
        )
        if entry["model"] == "kan":
            shutil.copyfile(
                output_dir / "kan_top_edge_functions.png",
                edge_runs_dir / f"{entry['run']}.png",
            )
            baseline = float(phase_rows[-1]["test_accuracy"])
            for row in _read_csv(output_dir / "kan_edge_ablation.csv"):
                accuracy = float(row["test_accuracy"])
                ablation_rows.append(
                    {
                        "run": entry["run"],
                        "seed": entry["seed"],
                        **row,
                        "baseline_test_accuracy": baseline,
                        "test_accuracy_delta": accuracy - baseline,
                    }
                )

    write_csv(args.tables_dir / "mechanism_runs.csv", phase_rows)
    write_csv(args.tables_dir / "mechanism_summary.csv", _aggregate_phase_rows(phase_rows))
    write_csv(args.tables_dir / "edge_ablation_runs.csv", ablation_rows)
    write_csv(args.tables_dir / "edge_ablation_summary.csv", _aggregate_ablation(ablation_rows))
    write_csv(args.tables_dir / "fourier_spectra.csv", spectrum_rows)
    write_csv(args.tables_dir / "fourier_top_modes.csv", top_mode_rows)
    spectrum_summary = _aggregate_spectra(spectrum_rows)
    write_csv(args.tables_dir / "fourier_spectrum_summary.csv", spectrum_summary)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    _plot_phase_metric(
        phase_rows,
        "modular_diagonal_power_fraction",
        "мощность на модульной Fourier-диагонали",
        args.figures_dir / "fourier_multiseed.png",
    )
    _plot_phase_metric(
        phase_rows,
        "hidden_participation_rank",
        "participation rank скрытого слоя",
        args.figures_dir / "participation_rank_multiseed.png",
    )
    _plot_ablation(ablation_rows, args.figures_dir / "edge_ablation_multiseed.png")
    _plot_spectrum_heatmaps(
        spectrum_summary, args.figures_dir / "fourier_spectrum_heatmaps.png"
    )

    if args.symmetry_dir:
        symmetry_rows: list[dict[str, Any]] = []
        for run_dir in sorted(
            path for path in args.symmetry_dir.iterdir() if (path / "summary.json").exists()
        ):
            output_dir = _run_analysis(run_dir, skip_kan_details=True)
            seed = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))["seed"]
            for row in _read_csv(output_dir / "phase_mechanisms.csv"):
                symmetry_rows.append(
                    {"run": run_dir.name, "model": "kan_symmetry", "seed": seed, **row}
                )
        final_random = [
            row for row in phase_rows if row["model"] == "kan" and row["phase"] == "final"
        ]
        final_symmetry = [row for row in symmetry_rows if row["phase"] == "final"]
        comparison = [{"split": "random", **row} for row in final_random] + [
            {"split": "symmetry_safe", **row} for row in final_symmetry
        ]
        write_csv(args.tables_dir / "symmetry_fourier_comparison.csv", comparison)
        figure, axis = plt.subplots(figsize=(6.5, 4.4))
        for x, split in enumerate(("random", "symmetry_safe")):
            values = [
                float(row["modular_diagonal_power_fraction"])
                for row in comparison
                if row["split"] == split
            ]
            axis.scatter([x] * len(values), values, s=35, alpha=0.7)
            axis.plot(
                [x - 0.18, x + 0.18], [statistics.median(values)] * 2, color="#111827", linewidth=2
            )
        axis.set_xticks((0, 1), ("случайное", "symmetry-safe"))
        axis.set(
            ylabel="финальная мощность Fourier-диагонали",
            title="Представление KAN зависит от разбиения",
        )
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(args.figures_dir / "symmetry_fourier_comparison.png", dpi=200)
        plt.close(figure)
    print("Saved multi-seed mechanism analysis")


if __name__ == "__main__":
    main()
