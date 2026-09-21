#!/usr/bin/env python
"""Run one configured quadratic-MLP or spline-KAN experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train import train_experiment  # noqa: E402
from src.utils import load_config, set_config_values  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    config = set_config_values(load_config(args.config), args.set)
    run_dir = train_experiment(config, args.output_dir)
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
