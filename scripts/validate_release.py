#!/usr/bin/env python
"""Validate the compact research repository before release packaging."""

from __future__ import annotations

import argparse
import csv
import re
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASES = (
    "classical_grokking",
    "weak_delayed_generalization",
    "joint_generalization",
    "memorized_not_generalized",
    "not_memorized",
)
REQUIRED_RUN_FILES = ("config.yaml", "metadata.json", "history.jsonl", "summary.json")
FORBIDDEN_PARTS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "tmp",
}
ABSOLUTE_PATH = re.compile(r"(?:^|(?<=[\"'\s(]))[A-Za-z]:[\\/]|/Users/|/home/", re.MULTILINE)
SECRET = re.compile(r"(?i)(?:api[_-]?key|secret|token|password)\s*[:=]\s*[\"']?[A-Za-z0-9_\-]{16,}")
TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_tables(root: Path) -> None:
    runs = read_csv(root / "results" / "tables" / "paired_runs.csv")
    summary = read_csv(root / "results" / "tables" / "paired_summary.csv")
    if len(runs) != 80 or len({row["run"] for row in runs}) != 80:
        raise ValueError("paired_runs.csv must contain exactly 80 unique runs")
    groups = Counter(
        (row["model"], row["p"], row["train_fraction"], row["weight_decay"]) for row in runs
    )
    if len(groups) != 16 or set(groups.values()) != {5}:
        raise ValueError("paired_runs.csv must contain 16 cells with five seeds each")
    for row in summary:
        phase_total = sum(int(row[f"{phase}_count"]) for phase in PHASES)
        if phase_total != int(row["seeds"]):
            raise ValueError(f"Phase counts do not sum to seeds for {row}")
    if len(summary) != 16:
        raise ValueError("paired_summary.csv must contain exactly 16 rows")


def validate_core_runs(root: Path) -> None:
    core = root / "results" / "core_runs"
    run_dirs = sorted(path for path in core.iterdir() if path.is_dir())
    if len(run_dirs) != 80:
        raise ValueError(f"Expected 80 core run directories, found {len(run_dirs)}")
    for run_dir in run_dirs:
        missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file()]
        if missing:
            raise ValueError(f"{run_dir.name} is missing {', '.join(missing)}")
    model_files = list(core.glob("*/model.pt"))
    snapshot_files = list(core.glob("*/model_snapshots.pt"))
    if len(model_files) != 10 or len(snapshot_files) != 10:
        raise ValueError("Release must keep model.pt and model_snapshots.pt for exactly ten runs")


def validate_clean_tree(root: Path) -> None:
    violations: list[str] = []
    absolute_paths: list[str] = []
    secrets: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in FORBIDDEN_PARTS for part in relative.parts):
            violations.append(relative.as_posix())
            continue
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            if relative.as_posix() != "scripts/validate_release.py" and ABSOLUTE_PATH.search(text):
                absolute_paths.append(relative.as_posix())
            if SECRET.search(text):
                secrets.append(relative.as_posix())
        elif path.is_file() and path.suffix.lower() == ".pptx":
            with zipfile.ZipFile(path) as archive:
                office_text = "\n".join(
                    archive.read(name).decode("utf-8", errors="replace")
                    for name in archive.namelist()
                    if name.endswith((".xml", ".rels"))
                )
            if ABSOLUTE_PATH.search(office_text):
                absolute_paths.append(relative.as_posix())
            if SECRET.search(office_text):
                secrets.append(relative.as_posix())
    if violations:
        raise ValueError(f"Forbidden paths found: {violations[:10]}")
    if absolute_paths:
        raise ValueError(f"Absolute paths found in text files: {absolute_paths[:10]}")
    if secrets:
        raise ValueError(f"Possible secrets found: {secrets[:10]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    validate_tables(root)
    validate_core_runs(root)
    validate_clean_tree(root)
    print("Release validation passed: 80 runs, 16 cells, five seeds per cell.")


if __name__ == "__main__":
    main()
