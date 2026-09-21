#!/usr/bin/env python
"""Analyze Fourier structure, hidden geometry, and causal KAN edge ablations."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import (  # noqa: E402
    classwise_fourier_power,
    mean_centered_fourier_spectrum,
    modular_diagonal_power_fraction,
    pca_summary,
    validate_snapshot_steps,
)
from src.data import GromovAdditionData, make_gromov_addition_data  # noqa: E402
from src.models import GromovQuadraticNetwork, SplineKAN  # noqa: E402
from src.utils import load_config, write_json  # noqa: E402


def build_model(config: dict[str, Any], input_dim: int, p: int) -> torch.nn.Module:
    model_config = config["model"]
    model_type = str(model_config["type"])
    if model_type == "gromov_quadratic":
        return GromovQuadraticNetwork(
            input_dim=input_dim,
            output_dim=p,
            width=int(model_config.get("width", 128)),
            activation=str(model_config.get("activation", "quadratic")),
            initialization_scale=float(model_config.get("initialization_scale", 1.0)),
        )
    if model_type == "spline_kan":
        return SplineKAN(
            input_dim=input_dim,
            output_dim=p,
            width=int(model_config.get("width", 14)),
            grid_size=int(model_config.get("grid_size", 5)),
            spline_order=int(model_config.get("spline_order", 3)),
            grid_range=tuple(float(v) for v in model_config.get("grid_range", [-2.0, 2.0])),
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def load_data(config: dict[str, Any]) -> GromovAdditionData:
    data_config = config["data"]
    seed = int(config["training"]["seed"])
    return make_gromov_addition_data(
        p=int(data_config["p"]),
        train_fraction=float(data_config["train_fraction"]),
        split_seed=int(data_config.get("split_seed", seed)),
        split_mode=str(data_config.get("split_mode", "random")),
        input_encoding=str(data_config.get("input_encoding", "one_hot")),
    )


def state_sequence(run_dir: Path) -> list[tuple[str, int, dict[str, torch.Tensor]]]:
    states: list[tuple[str, int, dict[str, torch.Tensor]]] = []
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    expected_steps = {
        "memorization": summary["phase_metrics"].get("memorization_step"),
        "generalization": summary["phase_metrics"].get("generalization_step"),
    }
    snapshots_path = run_dir / "model_snapshots.pt"
    snapshots: dict[str, Any] = {}
    if snapshots_path.exists():
        snapshots = torch.load(snapshots_path, map_location="cpu", weights_only=True)
    observed_steps = {name: int(item["step"]) for name, item in snapshots.items()}
    try:
        validate_snapshot_steps(expected_steps, observed_steps)
    except ValueError as error:
        raise ValueError(f"{run_dir.name}: {error}") from error
    for name in ("memorization", "generalization"):
        if name in snapshots:
            states.append((name, int(snapshots[name]["step"]), snapshots[name]["model"]))
    final = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    states.append(("final", int(final["step"]), final["model"]))
    return states


def accuracy(logits: torch.Tensor, targets: torch.Tensor, indices: torch.Tensor) -> float:
    return float((logits[indices].argmax(1) == targets[indices].argmax(1)).float().mean())


def plot_fourier(phases: list[dict[str, Any]], output_path: Path) -> None:
    figure, axes = plt.subplots(1, len(phases), figsize=(4.2 * len(phases), 3.8), squeeze=False)
    for axis, phase in zip(axes[0], phases):
        spectrum = np.fft.fftshift(phase["mean_power"])
        image = axis.imshow(np.log10(spectrum + 1e-10), cmap="magma", origin="lower")
        axis.set_title(f"{phase['name']} · step {phase['step']}")
        axis.set_xlabel("frequency b")
        axis.set_ylabel("frequency a")
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.suptitle("Mean classwise log Fourier power")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def spectrum_rows(
    phase: str, step: int, spectrum: np.ndarray, p: int
) -> list[dict[str, Any]]:
    frequencies = np.rint(np.fft.fftshift(np.fft.fftfreq(p) * p)).astype(int)
    rows: list[dict[str, Any]] = []
    for row_index, frequency_a in enumerate(frequencies):
        for column_index, frequency_b in enumerate(frequencies):
            rows.append(
                {
                    "phase": phase,
                    "step": step,
                    "frequency_a": int(frequency_a),
                    "frequency_b": int(frequency_b),
                    "power_fraction": float(spectrum[row_index, column_index]),
                    "on_modular_diagonal": bool(
                        frequency_a == frequency_b and frequency_a != 0
                    ),
                }
            )
    return rows


def top_spectrum_modes(
    phase: str, step: int, rows: list[dict[str, Any]], count: int = 12
) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if not (int(row["frequency_a"]) == 0 and int(row["frequency_b"]) == 0)
    ]
    candidates.sort(key=lambda row: float(row["power_fraction"]), reverse=True)
    return [
        {"phase": phase, "step": step, "rank": rank, **row}
        for rank, row in enumerate(candidates[:count], 1)
    ]


def plot_pca(phases: list[dict[str, Any]], labels: np.ndarray, output_path: Path) -> None:
    figure, axes = plt.subplots(1, len(phases), figsize=(4.4 * len(phases), 4.0), squeeze=False)
    for axis, phase in zip(axes[0], phases):
        projection = phase["projection"]
        scatter = axis.scatter(
            projection[:, 0], projection[:, 1], c=labels, s=8, cmap="hsv", alpha=0.75
        )
        evr = phase["explained_variance_ratio"]
        axis.set_title(
            f"{phase['name']} · step {phase['step']}\n"
            f"EV={evr[0]:.2f}+{evr[1]:.2f}, PR={phase['participation_rank']:.1f}"
        )
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
    figure.colorbar(scatter, ax=axes.ravel().tolist(), label="(a+b) mod p", fraction=0.025)
    figure.suptitle("Hidden representations")
    figure.subplots_adjust(left=0.07, right=0.92, bottom=0.14, top=0.78, wspace=0.30)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def ranked_kan_edges(model: SplineKAN, inputs: torch.Tensor) -> list[dict[str, Any]]:
    with torch.no_grad():
        hidden = model.hidden_representation(inputs)
        edge_tensors = (
            ("first", model.first.edge_values(inputs)),
            ("second", model.second.edge_values(hidden)),
        )
    rows: list[dict[str, Any]] = []
    for layer_name, values in edge_tensors:
        importance = values.square().mean(dim=0).sqrt()
        for output_index in range(importance.shape[0]):
            for input_index in range(importance.shape[1]):
                rows.append(
                    {
                        "layer": layer_name,
                        "output": output_index,
                        "input": input_index,
                        "rms_contribution": float(importance[output_index, input_index]),
                    }
                )
    rows.sort(key=lambda row: row["rms_contribution"], reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def zero_edges(model: SplineKAN, edges: list[dict[str, Any]]) -> None:
    with torch.no_grad():
        for edge in edges:
            layer = model.first if edge["layer"] == "first" else model.second
            output_index = int(edge["output"])
            input_index = int(edge["input"])
            layer.base_weight[output_index, input_index] = 0.0
            layer.spline_weight[output_index, input_index, :] = 0.0


def evaluate_ablation(
    model: SplineKAN,
    ranked: list[dict[str, Any]],
    data: GromovAdditionData,
    fractions: tuple[float, ...] = (0.01, 0.05, 0.10),
    random_repeats: int = 10,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fraction in fractions:
        count = max(1, round(len(ranked) * fraction))
        selections: list[tuple[str, int, list[dict[str, Any]]]] = [
            ("top", 0, ranked[:count]),
            ("bottom", 0, ranked[-count:]),
        ]
        generator = torch.Generator().manual_seed(1729 + count)
        for repeat in range(random_repeats):
            indices = torch.randperm(len(ranked), generator=generator)[:count].tolist()
            selections.append(("random", repeat, [ranked[index] for index in indices]))
        for strategy, repeat, selected in selections:
            ablated = copy.deepcopy(model)
            zero_edges(ablated, selected)
            with torch.no_grad():
                logits = ablated(data.inputs)
            rows.append(
                {
                    "fraction": fraction,
                    "edge_count": count,
                    "strategy": strategy,
                    "repeat": repeat,
                    "train_accuracy": accuracy(logits, data.targets, data.train_indices),
                    "test_accuracy": accuracy(logits, data.targets, data.test_indices),
                }
            )
    return rows


def plot_ablation(rows: list[dict[str, Any]], baseline_test: float, output_path: Path) -> None:
    fractions = sorted({float(row["fraction"]) for row in rows})
    strategies = ("top", "random", "bottom")
    colors = {"top": "#d62728", "random": "#7f7f7f", "bottom": "#1f77b4"}
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.axhline(baseline_test, color="black", linestyle="--", linewidth=1, label="no ablation")
    for strategy in strategies:
        means, errors = [], []
        for fraction in fractions:
            values = [
                row["test_accuracy"]
                for row in rows
                if row["strategy"] == strategy and row["fraction"] == fraction
            ]
            means.append(float(np.mean(values)))
            errors.append(float(np.std(values)))
        axis.errorbar(
            np.array(fractions) * 100,
            means,
            yerr=errors,
            marker="o",
            label=strategy,
            color=colors[strategy],
        )
    axis.set_xlabel("Ablated edges (%)")
    axis.set_ylabel("Test accuracy")
    axis.set_ylim(-0.02, 1.03)
    axis.set_title("Causal KAN edge ablation")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def edge_curve(layer: Any, output_index: int, input_index: int, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        basis = layer.b_spline_basis(x[:, None])[:, 0, :]
        return (
            layer.base_weight[output_index, input_index] * torch.nn.functional.silu(x)
            + basis @ layer.spline_weight[output_index, input_index]
        )


def plot_top_edges(model: SplineKAN, ranked: list[dict[str, Any]], output_path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    for axis, edge in zip(axes.ravel(), ranked[:6]):
        layer = model.first if edge["layer"] == "first" else model.second
        x = torch.linspace(
            float(layer.knots[layer.spline_order]),
            float(layer.knots[-layer.spline_order - 1]),
            300,
        )
        y = edge_curve(layer, int(edge["output"]), int(edge["input"]), x)
        axis.plot(x.numpy(), y.numpy(), color="#4c78a8")
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_title(
            f"#{edge['rank']} {edge['layer']} {edge['input']}→{edge['output']}\n"
            f"RMS={edge['rms_contribution']:.3g}"
        )
        axis.grid(alpha=0.2)
    figure.suptitle("Highest-RMS learned KAN edge functions")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--skip-kan-details",
        action="store_true",
        help="Skip edge ranking and ablation when only phase metrics are needed.",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "mechanisms").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(run_dir / "config.yaml")
    data = load_data(config)
    model = build_model(config, int(data.inputs.shape[1]), data.p)
    phase_rows: list[dict[str, Any]] = []
    fourier_phases: list[dict[str, Any]] = []
    fourier_spectrum_rows: list[dict[str, Any]] = []
    fourier_top_rows: list[dict[str, Any]] = []
    pca_phases: list[dict[str, Any]] = []
    final_model: torch.nn.Module | None = None

    for name, step, state in state_sequence(run_dir):
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            logits = model(data.inputs)
            hidden = model.hidden_representation(data.inputs)
            power = classwise_fourier_power(logits, data.p)
            centered_spectrum = mean_centered_fourier_spectrum(logits, data.p).numpy()
            pca = pca_summary(hidden)
        phase_rows.append(
            {
                "phase": name,
                "step": step,
                "train_accuracy": accuracy(logits, data.targets, data.train_indices),
                "test_accuracy": accuracy(logits, data.targets, data.test_indices),
                "modular_diagonal_power_fraction": modular_diagonal_power_fraction(logits, data.p),
                "hidden_pc1_variance": float(pca["explained_variance_ratio"][0]),
                "hidden_pc2_variance": float(pca["explained_variance_ratio"][1]),
                "hidden_participation_rank": float(pca["participation_rank"]),
            }
        )
        fourier_phases.append({"name": name, "step": step, "mean_power": power.mean(0).numpy()})
        current_spectrum_rows = spectrum_rows(name, step, centered_spectrum, data.p)
        fourier_spectrum_rows.extend(current_spectrum_rows)
        fourier_top_rows.extend(top_spectrum_modes(name, step, current_spectrum_rows))
        pca_phases.append(
            {
                "name": name,
                "step": step,
                "projection": pca["projection"].numpy(),
                "explained_variance_ratio": pca["explained_variance_ratio"].numpy(),
                "participation_rank": pca["participation_rank"],
            }
        )
        if name == "final":
            final_model = copy.deepcopy(model)

    write_csv(output_dir / "phase_mechanisms.csv", phase_rows)
    write_csv(output_dir / "fourier_spectrum.csv", fourier_spectrum_rows)
    write_csv(output_dir / "fourier_top_modes.csv", fourier_top_rows)
    plot_fourier(fourier_phases, output_dir / "fourier_by_phase.png")
    plot_pca(pca_phases, data.targets.argmax(1).numpy(), output_dir / "hidden_pca_by_phase.png")

    try:
        portable_run_dir = run_dir.relative_to(ROOT).as_posix()
    except ValueError:
        portable_run_dir = run_dir.name
    summary: dict[str, Any] = {
        "run_dir": portable_run_dir,
        "model_type": config["model"]["type"],
        "p": data.p,
        "input_encoding": data.input_encoding,
        "phases": phase_rows,
    }
    if isinstance(final_model, SplineKAN) and not args.skip_kan_details:
        ranked = ranked_kan_edges(final_model, data.inputs)
        write_csv(output_dir / "kan_edge_ranking.csv", ranked)
        ablations = evaluate_ablation(final_model, ranked, data)
        write_csv(output_dir / "kan_edge_ablation.csv", ablations)
        baseline_test = phase_rows[-1]["test_accuracy"]
        plot_ablation(ablations, baseline_test, output_dir / "kan_edge_ablation.png")
        plot_top_edges(final_model, ranked, output_dir / "kan_top_edge_functions.png")
        summary["kan"] = {
            "edge_count": len(ranked),
            "baseline_test_accuracy": baseline_test,
            "top_edges": ranked[:10],
            "ablation_rows": len(ablations),
        }
    write_json(output_dir / "mechanism_summary.json", summary)
    print(f"Saved mechanism analysis to {output_dir}")


if __name__ == "__main__":
    main()
