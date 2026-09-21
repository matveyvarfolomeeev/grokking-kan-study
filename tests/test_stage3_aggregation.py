import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

from src.metrics import LEARNING_DYNAMICS


def _write_run(root: Path, name: str, status: str, is_classical: bool) -> None:
    run = root / name
    run.mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps(
            {
                "status": status,
                "final": {"test_accuracy": 0.995},
                "phase_metrics": {
                    "dynamics": "classical_grokking" if is_classical else status,
                    "is_classical_grokking": is_classical,
                    "memorization_step": 100,
                    "generalization_step": 1500 if is_classical else None,
                    "generalization_delay": 1400 if is_classical else None,
                },
            }
        ),
        encoding="utf-8",
    )
    (run / "metadata.json").write_text(
        json.dumps(
            {
                "seed": 0,
                "dataset": {
                    "p": 31,
                    "input_encoding": "one_hot",
                    "split_mode": "random",
                    "train_fraction_realized": 0.35,
                },
                "model": {"type": "spline_kan", "parameters": 11718},
            }
        ),
        encoding="utf-8",
    )
    (run / "config.yaml").write_text(
        yaml.safe_dump({"model": {"grid_size": 5, "spline_order": 3, "width": 14}}),
        encoding="utf-8",
    )


def test_stage3_summary_counts_every_seed(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_run(raw, "legacy", "grokked", True)
    tables = tmp_path / "tables"
    figures = tmp_path / "figures"
    subprocess.run(
        [
            sys.executable,
            "scripts/aggregate_stage3_controls.py",
            "--input",
            f"spline_default={raw}",
            "--tables-dir",
            str(tables),
            "--figures-dir",
            str(figures),
        ],
        check=True,
    )
    with (tables / "stage3_control_summary.csv").open(encoding="utf-8-sig") as handle:
        row = next(csv.DictReader(handle))
    assert int(row["classical_grokking"]) == 1
    assert sum(int(row[name]) for name in LEARNING_DYNAMICS) == int(row["seeds"])
