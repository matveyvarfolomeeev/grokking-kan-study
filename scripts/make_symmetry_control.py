#!/usr/bin/env python
"""Build an explicit random/symmetry-safe phase comparison from final tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PHASES = (
    "classical_grokking",
    "weak_delayed_generalization",
    "joint_generalization",
    "memorized_not_generalized",
    "not_memorized",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, default=Path("results/tables/paired_runs.csv"))
    parser.add_argument(
        "--historical", type=Path, default=Path("results/tables/historical_controls.csv")
    )
    parser.add_argument(
        "--table", type=Path, default=Path("results/tables/symmetry_phase_summary.csv")
    )
    parser.add_argument(
        "--figure", type=Path, default=Path("results/figures/symmetry_phase_comparison.png")
    )
    args = parser.parse_args()

    paired = read_csv(args.paired)
    rows: list[dict[str, Any]] = []
    for model in ("mlp", "kan"):
        selected = [
            row
            for row in paired
            if row["model"] == model
            and int(row["p"]) == 31
            and abs(float(row["train_fraction"]) - 0.35) < 1e-9
        ]
        counts = {phase: sum(row["dynamics"] == phase for row in selected) for phase in PHASES}
        rows.append(
            {
                "split": "random",
                "model": model,
                "p": 31,
                "train_fraction": 0.35,
                "seeds": len(selected),
                **counts,
            }
        )

    historical = read_csv(args.historical)
    for control, model, fraction in (
        ("symmetry_kan", "kan", 0.35),
        ("symmetry_mlp", "mlp", 0.50),
    ):
        source = next(row for row in historical if row["control"] == control)
        rows.append(
            {
                "split": "symmetry_safe",
                "model": model,
                "p": 31,
                "train_fraction": fraction,
                "seeds": int(source["seeds"]),
                **{phase: int(source[phase]) for phase in PHASES},
            }
        )

    args.table.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.table, rows)

    labels = [
        (
            f"{row['split']}\n{row['model'].upper()} · p={row['p']} · "
            f"{float(row['train_fraction']):.0%}"
        )
        for row in rows
    ]
    classical = [int(row["classical_grokking"]) for row in rows]
    totals = [int(row["seeds"]) for row in rows]
    other = [total - success for total, success in zip(totals, classical)]
    figure, axis = plt.subplots(figsize=(8.2, 4.8), layout="constrained")
    positions = range(len(rows))
    axis.bar(positions, classical, color="#15803d", label="classical grokking")
    axis.bar(positions, other, bottom=classical, color="#d1d5db", label="other phases")
    for position, success, total in zip(positions, classical, totals):
        axis.text(position, total + 0.15, f"{success}/{total}", ha="center", fontweight="bold")
    axis.set_xticks(list(positions), labels)
    axis.set(
        ylabel="число seed",
        title="Symmetry-safe контроль сужает вывод о KAN",
        ylim=(0, max(totals) + 1.5),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=200)
    plt.close(figure)
    print(f"Saved {args.table} and {args.figure}")


if __name__ == "__main__":
    main()
