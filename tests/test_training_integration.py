import json
from pathlib import Path

from src.train import train_experiment


def tiny_gromov_config() -> dict:
    return {
        "project": "test",
        "run_name": "integration",
        "data": {"p": 5, "train_fraction": 0.5, "split_seed": 3, "split_mode": "random"},
        "model": {"type": "gromov_quadratic", "width": 6, "activation": "quadratic"},
        "training": {
            "seed": 7,
            "optimizer": "sgd",
            "learning_rate": 0.01,
            "mean_field_lr_scaling": True,
            "weight_decay": 0.0,
            "max_steps": 4,
            "eval_every": 1,
            "grok_threshold": 0.99,
            "grok_patience": 5,
            "stop_when_grokked": False,
            "device": "cpu",
        },
    }


def metric_rows(path: Path) -> list[dict]:
    rows = []
    for line in (path / "history.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        row.pop("elapsed_seconds")
        rows.append(row)
    return rows


def test_complete_training_run_writes_expected_artifacts(tmp_path: Path) -> None:
    run_dir = train_experiment(tiny_gromov_config(), tmp_path / "run")
    expected = {
        "config.yaml",
        "metadata.json",
        "history.jsonl",
        "summary.json",
        "model.pt",
        "predictions.pt",
    }
    assert expected.issubset(path.name for path in run_dir.iterdir())
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["criterion"] == "test_accuracy >= 0.99 for 5 consecutive checkpoints"
    assert "sustained_99_step" in summary
    assert summary["grok_step"] is None
    assert summary["status"] == summary["phase_metrics"]["dynamics"]


def test_training_metrics_are_reproducible(tmp_path: Path) -> None:
    first = train_experiment(tiny_gromov_config(), tmp_path / "first")
    second = train_experiment(tiny_gromov_config(), tmp_path / "second")
    assert metric_rows(first) == metric_rows(second)
