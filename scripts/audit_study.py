#!/usr/bin/env python
"""Audit the final paired study against the pre-registered protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.study import wilson_interval  # noqa: E402

PHASES = (
    "classical_grokking",
    "weak_delayed_generalization",
    "joint_generalization",
    "memorized_not_generalized",
    "not_memorized",
)
MODELS = ("mlp", "kan")
SEEDS = tuple(range(10, 15))
WEIGHT_DECAYS = (0.1, 0.3)
CONDITIONS = ((31, 0.35, 15_000), (31, 0.50, 15_000), (53, 0.30, 20_000), (53, 0.35, 20_000))
MECHANISM_CONDITION = (31, 0.35, 0.3)
ABLATION_FRACTIONS = (0.01, 0.05, 0.10)
EXCLUDED_CHECKSUM_PARTS = {".git", ".venv", ".pytest_cache", ".ruff_cache", "__pycache__", "tmp"}


class AuditError(ValueError):
    """Raised when release evidence contradicts the fixed protocol."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: str) -> float | None:
    return None if value == "" else float(value)


def close(actual: float, expected: float, *, tolerance: float = 1e-10) -> bool:
    return math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance)


def expected_matrix() -> set[tuple[str, int, float, float, int, int]]:
    return {
        (model, p, fraction, weight_decay, seed, max_steps)
        for model, (p, fraction, max_steps), weight_decay, seed in product(
            MODELS, CONDITIONS, WEIGHT_DECAYS, SEEDS
        )
    }


def audit_manifest_and_runs(root: Path) -> dict[str, dict[str, str]]:
    core = root / "results" / "core_runs"
    manifest = read_json(core / "study_manifest.json")
    records = manifest["runs"]
    require(manifest["expected_runs"] == 80, "Manifest expected_runs must equal 80")
    require(len(records) == 80, "Manifest must contain 80 records")
    require(len({record["run"] for record in records}) == 80, "Manifest run names must be unique")

    observed_matrix: set[tuple[str, int, float, float, int, int]] = set()
    for record in records:
        config = record["config"]
        data = config["data"]
        training = config["training"]
        key = (
            record["model"],
            int(record["p"]),
            float(record["train_fraction"]),
            float(record["weight_decay"]),
            int(record["seed"]),
            int(record["max_steps"]),
        )
        observed_matrix.add(key)
        require(record["status"] == "completed", f"Incomplete manifest record: {record['run']}")
        require(
            record["started_at"] and record["finished_at"], f"Missing timestamps: {record['run']}"
        )
        require(float(record["wall_seconds"]) >= 0.0, f"Invalid wall time: {record['run']}")
        run_dir = Path(record["run_dir"])
        require(
            not run_dir.is_absolute() and not run_dir.drive, f"Absolute run path: {record['run']}"
        )
        require(data["input_encoding"] == "one_hot", f"Non-one-hot input: {record['run']}")
        require(
            data["split_mode"] == "random", f"Non-random split in paired study: {record['run']}"
        )
        require(int(data["split_seed"]) == record["seed"], f"Split seed mismatch: {record['run']}")
        require(training["optimizer"] == "adamw", f"Optimizer mismatch: {record['run']}")
        require(
            close(float(training["learning_rate"]), 0.001),
            f"Learning-rate mismatch: {record['run']}",
        )
        require(
            int(training["eval_every"]) == 100, f"Evaluation interval mismatch: {record['run']}"
        )
        require(training["stop_when_grokked"] is False, f"Early stopping enabled: {record['run']}")
        require(int(training["seed"]) == record["seed"], f"Training seed mismatch: {record['run']}")

        run_path = root / run_dir
        metadata = read_json(run_path / "metadata.json")
        summary = read_json(run_path / "summary.json")
        require(metadata["optimizer"]["loss"] == "MSE", f"Loss mismatch: {record['run']}")
        require(
            metadata["optimizer"]["batching"] == "full_batch", f"Batching mismatch: {record['run']}"
        )
        require(
            metadata["optimizer"]["name"] == "adamw",
            f"Metadata optimizer mismatch: {record['run']}",
        )
        require(
            summary["final"]["step"] == record["max_steps"], f"Run stopped early: {record['run']}"
        )
        dynamics = summary["phase_metrics"]["dynamics"]
        classical = bool(summary["phase_metrics"]["is_classical_grokking"])
        require(summary["status"] == dynamics, f"Non-canonical status: {record['run']}")
        require(
            classical == (dynamics == "classical_grokking"),
            f"Classical flag mismatch: {record['run']}",
        )
        require(
            (summary["grok_step"] is not None) == classical,
            f"grok_step semantics mismatch: {record['run']}",
        )

        history = [
            json.loads(line) for line in (run_path / "history.jsonl").read_text().splitlines()
        ]
        expected_steps = [1, *range(100, int(record["max_steps"]) + 1, 100)]
        require(
            [int(row["step"]) for row in history] == expected_steps,
            f"History grid mismatch: {record['run']}",
        )

    require(observed_matrix == expected_matrix(), "Manifest does not match the fixed 80-run matrix")

    paired_rows = read_csv(root / "results" / "tables" / "paired_runs.csv")
    require(len(paired_rows) == 80, "paired_runs.csv must contain 80 rows")
    paired_by_run = {row["run"]: row for row in paired_rows}
    require(
        set(paired_by_run) == {record["run"] for record in records}, "Manifest/paired run mismatch"
    )
    return paired_by_run


def audit_phase_statistics(root: Path, paired_by_run: dict[str, dict[str, str]]) -> None:
    grouped: dict[tuple[str, int, float, float], list[dict[str, str]]] = defaultdict(list)
    for row in paired_by_run.values():
        grouped[
            (row["model"], int(row["p"]), float(row["train_fraction"]), float(row["weight_decay"]))
        ].append(row)
    require(
        len(grouped) == 16 and {len(rows) for rows in grouped.values()} == {5},
        "Expected 16 cells of five seeds",
    )

    summary_rows = read_csv(root / "results" / "tables" / "paired_summary.csv")
    require(len(summary_rows) == 16, "paired_summary.csv must contain 16 rows")
    for summary in summary_rows:
        key = (
            summary["model"],
            int(summary["p"]),
            float(summary["train_fraction"]),
            float(summary["weight_decay"]),
        )
        runs = grouped[key]
        phase_counts = Counter(row["dynamics"] for row in runs)
        require(sum(phase_counts.values()) == 5, f"Phase total mismatch: {key}")
        for phase in PHASES:
            count = phase_counts[phase]
            require(
                int(summary[f"{phase}_count"]) == count, f"Phase count mismatch: {key}, {phase}"
            )
            low, high = wilson_interval(count, 5)
            require(
                close(float(summary[f"{phase}_ci95_low"]), low),
                f"Wilson low mismatch: {key}, {phase}",
            )
            require(
                close(float(summary[f"{phase}_ci95_high"]), high),
                f"Wilson high mismatch: {key}, {phase}",
            )

        delays = [float(row["generalization_delay"]) for row in runs if row["generalization_delay"]]
        censored = sum(row["censored"].lower() == "true" for row in runs)
        require(
            int(summary["generalized_count"]) == len(delays),
            f"Generalized denominator mismatch: {key}",
        )
        require(int(summary["censored_count"]) == censored, f"Censor count mismatch: {key}")
        if delays:
            require(
                close(float(summary["delay_median"]), statistics.median(delays)),
                f"Delay median mismatch: {key}",
            )
            require(close(float(summary["delay_min"]), min(delays)), f"Delay min mismatch: {key}")
            require(close(float(summary["delay_max"]), max(delays)), f"Delay max mismatch: {key}")
        else:
            require(
                summary["delay_median"] == summary["delay_min"] == summary["delay_max"] == "",
                f"Empty-delay mismatch: {key}",
            )


def audit_mechanisms(root: Path, paired_by_run: dict[str, dict[str, str]]) -> None:
    selected = {
        run: row
        for run, row in paired_by_run.items()
        if (int(row["p"]), float(row["train_fraction"]), float(row["weight_decay"]))
        == MECHANISM_CONDITION
    }
    require(len(selected) == 10, "Mechanism condition must select ten models")
    core = root / "results" / "core_runs"
    require(
        {path.parent.name for path in core.glob("*/model.pt")} == set(selected),
        "Unexpected retained model.pt files",
    )
    require(
        {path.parent.name for path in core.glob("*/model_snapshots.pt")} == set(selected),
        "Unexpected retained model snapshots",
    )

    mechanism_rows = read_csv(root / "results" / "tables" / "mechanism_runs.csv")
    by_run: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in mechanism_rows:
        by_run[row["run"]][row["phase"]] = row
        for field in ("modular_diagonal_power_fraction", "hidden_participation_rank"):
            require(
                math.isfinite(float(row[field])),
                f"Non-finite mechanism metric: {row['run']}, {field}",
            )
    require(set(by_run) == set(selected), "Mechanism table run set mismatch")
    for run, paired in selected.items():
        expected_phases = {"memorization", "final"}
        if paired["generalization_step"]:
            expected_phases.add("generalization")
        require(set(by_run[run]) == expected_phases, f"Mechanism phase mismatch: {run}")
        require(
            int(by_run[run]["memorization"]["step"]) == int(paired["memorization_step"]),
            f"Memorization snapshot mismatch: {run}",
        )
        require(
            int(by_run[run]["final"]["step"]) == int(paired["max_steps"]),
            f"Final snapshot mismatch: {run}",
        )
        if "generalization" in expected_phases:
            require(
                int(by_run[run]["generalization"]["step"]) == int(paired["generalization_step"]),
                f"Generalization snapshot mismatch: {run}",
            )

    ablations = read_csv(root / "results" / "tables" / "edge_ablation_runs.csv")
    kan_runs = {run for run, row in selected.items() if row["model"] == "kan"}
    require({row["run"] for row in ablations} == kan_runs, "Ablation run set mismatch")
    cells: dict[tuple[str, float, str], list[dict[str, str]]] = defaultdict(list)
    for row in ablations:
        cells[(row["run"], float(row["fraction"]), row["strategy"])].append(row)
    for run, fraction in product(sorted(kan_runs), ABLATION_FRACTIONS):
        for strategy, expected_count in (("top", 1), ("bottom", 1), ("random", 10)):
            rows = cells[(run, fraction, strategy)]
            require(
                len(rows) == expected_count,
                f"Ablation repeat mismatch: {run}, {fraction}, {strategy}",
            )
            if strategy == "random":
                require(
                    {int(row["repeat"]) for row in rows} == set(range(10)),
                    f"Random repeat IDs mismatch: {run}, {fraction}",
                )

    symmetry = read_csv(root / "results" / "tables" / "symmetry_fourier_comparison.csv")
    require(
        Counter(row["split"] for row in symmetry) == {"random": 5, "symmetry_safe": 5},
        "Symmetry control must contain five runs per split",
    )
    symmetry_phases = read_csv(root / "results" / "tables" / "symmetry_phase_summary.csv")
    expected_symmetry = {
        ("random", "mlp", "0.35"): (0, 10),
        ("random", "kan", "0.35"): (8, 10),
        ("symmetry_safe", "kan", "0.35"): (0, 5),
        ("symmetry_safe", "mlp", "0.5"): (5, 5),
    }
    observed_symmetry = {
        (row["split"], row["model"], str(float(row["train_fraction"]))): (
            int(row["classical_grokking"]),
            int(row["seeds"]),
        )
        for row in symmetry_phases
    }
    require(observed_symmetry == expected_symmetry, "Symmetry phase summary mismatch")

    spectra = read_csv(root / "results" / "tables" / "fourier_spectra.csv")
    require({row["run"] for row in spectra} == set(selected), "Fourier spectrum run mismatch")
    spectrum_counts = Counter((row["run"], row["phase"]) for row in spectra)
    for run, phases in by_run.items():
        for phase in phases:
            require(
                spectrum_counts[(run, phase)] == 31 * 31,
                f"Fourier spectrum shape mismatch: {run}, {phase}",
            )
    for row in spectra:
        require(math.isfinite(float(row["power_fraction"])), "Non-finite Fourier power")

    top_modes = read_csv(root / "results" / "tables" / "fourier_top_modes.csv")
    top_counts = Counter((row["run"], row["phase"]) for row in top_modes)
    require(
        all(count == 12 for count in top_counts.values()),
        "Each run phase must retain twelve Fourier modes",
    )
    require(set(top_counts) == set(spectrum_counts), "Fourier top-mode phase mismatch")

    figures = root / "results" / "figures"
    for directory, expected in (("fourier_runs", 10), ("pca_runs", 10), ("kan_edge_functions", 5)):
        require(
            len(list((figures / directory).glob("*.png"))) == expected,
            f"Unexpected {directory} figure count",
        )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_artifacts_and_hashes(root: Path) -> None:
    required = (
        "results/figures/paired_phase_heatmap.png",
        "results/figures/paired_delays.png",
        "results/figures/fourier_multiseed.png",
        "results/figures/fourier_spectrum_heatmaps.png",
        "results/figures/participation_rank_multiseed.png",
        "results/figures/edge_ablation_multiseed.png",
        "results/figures/symmetry_fourier_comparison.png",
        "results/figures/symmetry_phase_comparison.png",
        "docs/project_summary.pdf",
        "docs/report.pdf",
        "docs/grokking_kan_study.pdf",
        "docs/grokking_kan_study.pptx",
    )
    for relative in required:
        require((root / relative).is_file(), f"Missing release artifact: {relative}")

    with zipfile.ZipFile(root / "docs" / "grokking_kan_study.pptx") as deck:
        slides = [
            name for name in deck.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
        ]
    require(len(slides) == 8, "Editable presentation must contain eight research slides")

    manifest_rows = read_csv(root / "RESULTS_MANIFEST.csv")
    actual_results = sorted(path for path in (root / "results").rglob("*") if path.is_file())
    require(len(manifest_rows) == len(actual_results), "RESULTS_MANIFEST row count mismatch")
    manifest_by_path = {row["path"]: row for row in manifest_rows}
    for path in actual_results:
        relative = path.relative_to(root).as_posix()
        row = manifest_by_path.get(relative)
        require(row is not None, f"Result missing from manifest: {relative}")
        require(int(row["bytes"]) == path.stat().st_size, f"Manifest size mismatch: {relative}")
        require(row["sha256"] == sha256(path), f"Manifest hash mismatch: {relative}")

    checksum_rows = {}
    for line in (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        checksum_rows[relative] = digest
    included = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "SHA256SUMS.txt"
        and not any(part in EXCLUDED_CHECKSUM_PARTS for part in path.relative_to(root).parts)
    )
    require(
        set(checksum_rows) == {path.relative_to(root).as_posix() for path in included},
        "SHA256SUMS file set mismatch",
    )
    for path in included:
        relative = path.relative_to(root).as_posix()
        require(
            checksum_rows[relative] == sha256(path), f"Repository checksum mismatch: {relative}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()

    paired_by_run = audit_manifest_and_runs(root)
    audit_phase_statistics(root, paired_by_run)
    audit_mechanisms(root, paired_by_run)
    audit_artifacts_and_hashes(root)
    print(
        "Study audit passed: fixed 80-run matrix, phase statistics, "
        "ten-model mechanisms, ablations, artifacts, and hashes."
    )


if __name__ == "__main__":
    main()
