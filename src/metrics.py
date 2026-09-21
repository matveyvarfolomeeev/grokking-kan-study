"""Metrics and grokking checkpoint definitions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

LEARNING_DYNAMICS = (
    "classical_grokking",
    "weak_delayed_generalization",
    "joint_generalization",
    "memorized_not_generalized",
    "not_memorized",
)


def canonical_learning_dynamics(summary: dict[str, Any]) -> str:
    """Return the phase label used by current reports.

    Early runs stored successful delayed generalization as ``grokked`` in the
    top-level status. Newer runs store the stricter classification inside
    ``phase_metrics``. The latter is authoritative because it also records the
    memorization gap and plateau duration.
    """

    phase = summary.get("phase_metrics") or {}
    if phase.get("is_classical_grokking") is True:
        return "classical_grokking"
    dynamics = phase.get("dynamics") or summary.get("status")
    if dynamics == "grokked":
        dynamics = "classical_grokking"
    if dynamics not in LEARNING_DYNAMICS:
        raise ValueError(f"Unknown learning dynamics: {dynamics!r}")
    return str(dynamics)


def find_grok_step(
    checkpoints: Iterable[dict[str, float]],
    threshold: float = 0.99,
    patience: int = 5,
) -> int | None:
    """Return first checkpoint in a consecutive threshold run, if present."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if patience < 1:
        raise ValueError("patience must be positive")
    consecutive = 0
    first_step: int | None = None
    for checkpoint in checkpoints:
        if checkpoint["test_accuracy"] >= threshold:
            consecutive += 1
            first_step = checkpoint["step"] if consecutive == 1 else first_step
            if consecutive >= patience:
                return int(first_step)
        else:
            consecutive = 0
            first_step = None
    return None


def find_sustained_step(
    checkpoints: Iterable[dict[str, float]],
    metric: str,
    threshold: float = 0.99,
    patience: int = 5,
) -> int | None:
    """Return the first step of a sustained threshold crossing for any metric."""

    rows = list(checkpoints)
    projected = [{"step": row["step"], "test_accuracy": row[metric]} for row in rows]
    return find_grok_step(projected, threshold=threshold, patience=patience)


def classify_learning_dynamics(
    checkpoints: Iterable[dict[str, float]],
    accuracy_threshold: float = 0.99,
    patience: int = 5,
    low_test_threshold: float = 0.20,
    plateau_end_threshold: float = 0.90,
    min_plateau_steps: int = 1000,
    min_gap_at_memorization: float = 0.50,
) -> dict[str, Any]:
    """Separate classical grokking from ordinary or weak delayed generalization.

    Classical grokking requires sustained train memorization, poor test accuracy
    at that point, a non-trivial overfitting plateau, and later sustained test
    generalization. Thresholds are explicit so phase-diagram studies can report
    sensitivity rather than hiding the definition.
    """

    rows = list(checkpoints)
    if not rows:
        raise ValueError("checkpoints must not be empty")
    memorization_step = find_sustained_step(rows, "train_accuracy", accuracy_threshold, patience)
    generalization_step = find_sustained_step(rows, "test_accuracy", accuracy_threshold, patience)
    row_by_step = {int(row["step"]): row for row in rows}
    test_at_memorization: float | None = None
    gap_at_memorization: float | None = None
    plateau_end_step: int | None = None
    plateau_duration: int | None = None
    delay: int | None = None
    if memorization_step is not None:
        memorization_row = row_by_step[memorization_step]
        test_at_memorization = float(memorization_row["test_accuracy"])
        gap_at_memorization = float(
            memorization_row["train_accuracy"] - memorization_row["test_accuracy"]
        )
        for row in rows:
            if row["step"] >= memorization_step and row["test_accuracy"] >= plateau_end_threshold:
                plateau_end_step = int(row["step"])
                break
        if plateau_end_step is None:
            plateau_duration = int(rows[-1]["step"] - memorization_step)
        else:
            plateau_duration = plateau_end_step - memorization_step
    if memorization_step is not None and generalization_step is not None:
        delay = generalization_step - memorization_step

    if memorization_step is None:
        dynamics = "not_memorized"
    elif generalization_step is None:
        dynamics = "memorized_not_generalized"
    elif generalization_step <= memorization_step or gap_at_memorization is None:
        dynamics = "joint_generalization"
    elif (
        test_at_memorization is not None
        and test_at_memorization <= low_test_threshold
        and gap_at_memorization >= min_gap_at_memorization
        and plateau_duration is not None
        and plateau_duration >= min_plateau_steps
    ):
        dynamics = "classical_grokking"
    elif gap_at_memorization >= 0.10:
        dynamics = "weak_delayed_generalization"
    else:
        dynamics = "joint_generalization"

    return {
        "dynamics": dynamics,
        "is_classical_grokking": dynamics == "classical_grokking",
        "memorization_step": memorization_step,
        "generalization_step": generalization_step,
        "generalization_delay": delay,
        "test_accuracy_at_memorization": test_at_memorization,
        "accuracy_gap_at_memorization": gap_at_memorization,
        "plateau_end_step": plateau_end_step,
        "plateau_duration_steps": plateau_duration,
        "definition": {
            "accuracy_threshold": accuracy_threshold,
            "patience": patience,
            "low_test_threshold": low_test_threshold,
            "plateau_end_threshold": plateau_end_threshold,
            "min_plateau_steps": min_plateau_steps,
            "min_gap_at_memorization": min_gap_at_memorization,
        },
    }
