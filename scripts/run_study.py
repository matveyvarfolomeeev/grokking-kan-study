#!/usr/bin/env python
"""Run the preregistered MLP/KAN matrix from one YAML manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train import train_experiment  # noqa: E402
from src.utils import load_config, set_config_values  # noqa: E402


def _config_hash(config: dict[str, Any]) -> str:
    serialized = yaml.safe_dump(config, allow_unicode=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _write_manifest(path: Path, study: str, expected: int, entries: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps(
            {"study": study, "expected_runs": expected, "runs": entries},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _execute(config: dict[str, Any], run_dir: str) -> dict[str, Any]:
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    train_experiment(config, Path(run_dir))
    return {
        "status": "completed",
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "wall_seconds": round(time.perf_counter() - started, 6),
    }


def _portable_run_dir(run_dir: Path) -> str:
    try:
        return run_dir.relative_to(ROOT).as_posix()
    except ValueError:
        return str(run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")

    specification = load_config(args.manifest)
    study_name = str(specification["study_name"])
    expected_runs = int(specification["expected_runs"])
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "study_manifest.json"
    entries: list[dict[str, Any]] = []
    pending: list[tuple[int, dict[str, Any], Path]] = []

    for condition in specification["conditions"]:
        p = int(condition["p"])
        max_steps = int(condition["max_steps"])
        for fraction in condition["fractions"]:
            fraction = float(fraction)
            for model_name, model_spec in specification["models"].items():
                for weight_decay in specification["weight_decays"]:
                    for seed in specification["seeds"]:
                        weight_decay = float(weight_decay)
                        seed = int(seed)
                        name = (
                            f"{model_name}_p{p}_f{int(round(fraction * 100)):02d}_"
                            f"wd{weight_decay:g}_seed{seed}"
                        )
                        run_dir = output_dir / name
                        assignments = [
                            f"run_name={name}",
                            f"data.p={p}",
                            f"data.train_fraction={fraction}",
                            f"data.split_seed={seed}",
                            f"training.seed={seed}",
                            f"training.weight_decay={weight_decay}",
                            f"training.max_steps={max_steps}",
                            *specification["common_overrides"],
                            *model_spec.get("overrides", []),
                        ]
                        config = set_config_values(load_config(model_spec["config"]), assignments)
                        entry: dict[str, Any] = {
                            "run": name,
                            "model": model_name,
                            "p": p,
                            "train_fraction": fraction,
                            "weight_decay": weight_decay,
                            "seed": seed,
                            "max_steps": max_steps,
                            "run_dir": _portable_run_dir(run_dir),
                            "config": config,
                            "config_sha256": _config_hash(config),
                        }
                        if args.dry_run:
                            entry["status"] = "planned"
                            entry["wall_seconds"] = 0.0
                        elif (run_dir / "summary.json").exists():
                            metadata = json.loads(
                                (run_dir / "metadata.json").read_text(encoding="utf-8")
                            )
                            summary = json.loads(
                                (run_dir / "summary.json").read_text(encoding="utf-8")
                            )
                            entry["status"] = "completed"
                            entry["resume_action"] = "skipped_existing"
                            entry["wall_seconds"] = float(summary["final"]["elapsed_seconds"])
                            entry["started_at"] = datetime.fromtimestamp(
                                float(metadata["started_at_unix"]), UTC
                            ).isoformat()
                            entry["finished_at"] = datetime.fromtimestamp(
                                float(metadata["started_at_unix"])
                                + float(summary["final"]["elapsed_seconds"]),
                                UTC,
                            ).isoformat()
                        else:
                            entry["status"] = "pending"
                            pending.append((len(entries), config, run_dir))
                        entries.append(entry)

    if len(entries) != expected_runs:
        raise ValueError(f"Manifest expected {expected_runs} runs, generated {len(entries)}")
    _write_manifest(manifest_path, study_name, expected_runs, entries)
    if pending:
        failures: list[str] = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_execute, config, str(run_dir)): index
                for index, config, run_dir in pending
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    entries[index].update(future.result())
                except Exception as error:
                    entries[index]["status"] = "failed"
                    entries[index]["error"] = f"{type(error).__name__}: {error}"
                    failures.append(entries[index]["run"])
                _write_manifest(manifest_path, study_name, expected_runs, entries)
                print(f"{entries[index]['status']}: {entries[index]['run']}", flush=True)
        if failures:
            raise RuntimeError(f"Failed runs: {', '.join(failures)}")
    print(f"Study manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
