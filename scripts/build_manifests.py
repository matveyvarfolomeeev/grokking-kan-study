#!/usr/bin/env python
"""Create compact result inventory and repository SHA-256 checksums."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {".git", ".venv", ".pytest_cache", ".ruff_cache", "__pycache__", "tmp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "SHA256SUMS.txt"
        and not any(part in EXCLUDED_PARTS for part in path.relative_to(root).parts)
    )


def result_category(relative: Path) -> str:
    if "core_runs" in relative.parts:
        return "paired_run"
    if "figures" in relative.parts:
        return "figure"
    if "tables" in relative.parts:
        return "table"
    return "result_metadata"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()

    result_files = sorted(path for path in (root / "results").rglob("*") if path.is_file())
    manifest_path = root / "RESULTS_MANIFEST.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "category", "bytes", "sha256"))
        writer.writeheader()
        for path in result_files:
            relative = path.relative_to(root)
            writer.writerow(
                {
                    "path": relative.as_posix(),
                    "category": result_category(relative),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )

    checksum_path = root / "SHA256SUMS.txt"
    lines = [
        f"{sha256(path)}  {path.relative_to(root).as_posix()}" for path in included_files(root)
    ]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {manifest_path.name} and {checksum_path.name}")


if __name__ == "__main__":
    main()
