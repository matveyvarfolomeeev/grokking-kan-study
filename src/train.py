"""Full-batch training and artifact persistence for the controlled study."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .data import GromovAdditionData, make_gromov_addition_data
from .metrics import classify_learning_dynamics, find_grok_step
from .models import GromovQuadraticNetwork, SplineKAN, count_parameters
from .utils import (
    package_versions,
    resolve_device,
    seed_everything,
    utc_run_id,
    write_json,
    write_yaml,
)


def squared_l2_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean squared output-vector distance, matching Gromov's objective."""

    if predictions.shape != targets.shape or predictions.ndim != 2:
        raise ValueError("predictions and targets must have the same [batch, output] shape")
    return (predictions - targets).square().sum(dim=1).mean()


def _build_model(
    model_config: dict[str, Any], input_dim: int, output_dim: int, device: torch.device
) -> GromovQuadraticNetwork | SplineKAN:
    model_type = str(model_config["type"])
    if model_type == "gromov_quadratic":
        return GromovQuadraticNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            width=int(model_config.get("width", 128)),
            activation=str(model_config.get("activation", "quadratic")),
            initialization_scale=float(model_config.get("initialization_scale", 1.0)),
        ).to(device)
    if model_type == "spline_kan":
        grid_range = tuple(float(value) for value in model_config.get("grid_range", [-2.0, 2.0]))
        return SplineKAN(
            input_dim=input_dim,
            output_dim=output_dim,
            width=int(model_config.get("width", 14)),
            grid_size=int(model_config.get("grid_size", 5)),
            spline_order=int(model_config.get("spline_order", 3)),
            grid_range=grid_range,
        ).to(device)
    raise ValueError(f"Unknown model type: {model_type}")


def _build_optimizer(
    model: GromovQuadraticNetwork | SplineKAN,
    training_config: dict[str, Any],
    input_dim: int,
) -> torch.optim.Optimizer:
    name = str(training_config.get("optimizer", "adamw")).lower()
    learning_rate = float(training_config.get("learning_rate", 0.001))
    weight_decay = float(training_config.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        mean_field = bool(training_config.get("mean_field_lr_scaling", False))
        if mean_field and isinstance(model, GromovQuadraticNetwork):
            return torch.optim.SGD(
                [
                    {"params": [model.first], "lr": learning_rate * input_dim**0.5},
                    {"params": [model.second], "lr": learning_rate * model.width},
                ],
                weight_decay=weight_decay,
            )
        return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def _evaluate(
    model: GromovQuadraticNetwork | SplineKAN,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        inputs, targets = next(iter(loader))
        predictions = model(inputs.to(device))
        targets = targets.to(device)
        loss = squared_l2_loss(predictions, targets).item()
        accuracy = (predictions.argmax(dim=1) == targets.argmax(dim=1)).float().mean().item()
    return {"loss": loss, "accuracy": accuracy}


def _snapshot(
    model: GromovQuadraticNetwork | SplineKAN,
    data: GromovAdditionData,
    device: torch.device,
    step: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    with torch.no_grad():
        predictions = model(data.inputs.to(device)).cpu()
    model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return (
        {"step": step, "predictions": predictions},
        {"step": step, "model": model_state},
    )


def train_experiment(config: dict[str, Any], output_dir: str | Path | None = None) -> Path:
    """Train one declared MLP or KAN experiment and save its full audit trail."""

    model_type = str(config["model"].get("type"))
    if model_type not in {"gromov_quadratic", "spline_kan"}:
        raise ValueError(f"Unsupported final-study model: {model_type}")

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    seed = int(training_config["seed"])
    seed_everything(seed)
    device = resolve_device(str(training_config.get("device", "auto")))
    data = make_gromov_addition_data(
        p=int(data_config["p"]),
        train_fraction=float(data_config["train_fraction"]),
        split_seed=int(data_config.get("split_seed", seed)),
        split_mode=str(data_config.get("split_mode", "random")),
        input_encoding=str(data_config.get("input_encoding", "one_hot")),
    )
    input_dim = int(data.inputs.shape[1])
    model = _build_model(model_config, input_dim, data.p, device)
    optimizer = _build_optimizer(model, training_config, input_dim)
    train_loader = DataLoader(data.train, batch_size=len(data.train), shuffle=False)
    test_loader = DataLoader(data.test, batch_size=len(data.test), shuffle=False)

    run_name = str(config.get("run_name", "run"))
    run_dir = Path(output_dir) if output_dir else Path("results/core_runs") / utc_run_id(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(run_dir / "config.yaml", config)
    architecture = (
        "bias-free quadratic MLP"
        if isinstance(model, GromovQuadraticNetwork)
        else "two-layer spline-KAN"
    )
    metadata: dict[str, Any] = {
        "project": config.get("project"),
        "run_name": run_name,
        "seed": seed,
        "device": str(device),
        "packages": package_versions(),
        "reference": "Gromov, arXiv:2301.02679",
        "dataset": {
            "p": data.p,
            "input_encoding": data.input_encoding,
            "input_dim": input_dim,
            "target_encoding": "one-hot(a+b mod p)",
            "total_examples": len(data.pairs),
            "train_examples": len(data.train),
            "test_examples": len(data.test),
            "train_fraction_realized": len(data.train) / len(data.pairs),
            "split_mode": data.split_mode,
        },
        "model": {
            "architecture": architecture,
            "type": model_type,
            "activation": str(model_config.get("activation", "spline")),
            "initialization_scale": float(model_config.get("initialization_scale", 1.0)),
            "width": int(
                model_config.get("width", 128 if isinstance(model, GromovQuadraticNetwork) else 14)
            ),
            "parameters": count_parameters(model),
        },
        "optimizer": {
            "name": str(training_config.get("optimizer", "adamw")).lower(),
            "learning_rate": float(training_config.get("learning_rate", 0.001)),
            "mean_field_lr_scaling": bool(training_config.get("mean_field_lr_scaling", False)),
            "weight_decay": float(training_config.get("weight_decay", 0.0)),
            "loss": "MSE",
            "batching": "full_batch",
        },
        "started_at_unix": time.time(),
    }
    if isinstance(model, SplineKAN):
        metadata["model"].update(
            {
                "grid_size": int(model_config.get("grid_size", 5)),
                "spline_order": int(model_config.get("spline_order", 3)),
                "grid_range": [
                    float(value) for value in model_config.get("grid_range", [-2.0, 2.0])
                ],
            }
        )
    write_json(run_dir / "metadata.json", metadata)

    history: list[dict[str, float]] = []
    prediction_snapshots: dict[str, dict[str, Any]] = {}
    model_snapshots: dict[str, dict[str, Any]] = {}
    candidate_snapshots: dict[str, tuple[dict[str, Any], dict[str, Any]] | None] = {
        "memorization": None,
        "generalization": None,
    }
    threshold_streaks = {"memorization": 0, "generalization": 0}
    max_steps = int(training_config.get("max_steps", 100000))
    eval_every = int(training_config.get("eval_every", 1000))
    accuracy_threshold = float(training_config.get("grok_threshold", 0.99))
    threshold_patience = int(training_config.get("grok_patience", 5))
    train_inputs, train_targets = next(iter(train_loader))
    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    kan_cache = (
        model.cache_first_layer_inputs(train_inputs) if isinstance(model, SplineKAN) else None
    )
    started = time.perf_counter()

    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        predictions = (
            model.forward_with_first_layer_cache(kan_cache)
            if isinstance(model, SplineKAN) and kan_cache is not None
            else model(train_inputs)
        )
        loss = squared_l2_loss(predictions, train_targets)
        loss.backward()
        optimizer.step()
        if step != 1 and step % eval_every != 0 and step != max_steps:
            continue

        train_metrics = _evaluate(model, train_loader, device)
        test_metrics = _evaluate(model, test_loader, device)
        history.append(
            {
                "step": step,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        for phase, observed_accuracy in (
            ("memorization", train_metrics["accuracy"]),
            ("generalization", test_metrics["accuracy"]),
        ):
            if phase in prediction_snapshots:
                continue
            if observed_accuracy >= accuracy_threshold:
                if threshold_streaks[phase] == 0:
                    candidate_snapshots[phase] = _snapshot(model, data, device, step)
                threshold_streaks[phase] += 1
                if threshold_streaks[phase] >= threshold_patience:
                    candidate = candidate_snapshots[phase]
                    if candidate is None:
                        raise RuntimeError(f"Missing candidate snapshot for {phase}")
                    prediction_snapshots[phase], model_snapshots[phase] = candidate
            else:
                threshold_streaks[phase] = 0
                candidate_snapshots[phase] = None
        sustained_step = find_grok_step(
            history,
            threshold=accuracy_threshold,
            patience=threshold_patience,
        )
        if sustained_step is not None and "generalization" not in prediction_snapshots:
            prediction, state = _snapshot(model, data, device, step)
            prediction_snapshots["generalization"] = prediction
            model_snapshots["generalization"] = state
        if sustained_step is not None and bool(training_config.get("stop_when_grokked", False)):
            break

    phase_metrics = classify_learning_dynamics(
        history,
        accuracy_threshold=accuracy_threshold,
        patience=threshold_patience,
        min_plateau_steps=int(training_config.get("min_plateau_steps", 1000)),
    )
    with (run_dir / "history.jsonl").open("w", encoding="utf-8") as handle:
        for row in history:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "sustained_99_step": find_grok_step(
            history,
            threshold=accuracy_threshold,
            patience=threshold_patience,
        ),
        "grok_step": (
            phase_metrics["generalization_step"] if phase_metrics["is_classical_grokking"] else None
        ),
        "status": phase_metrics["dynamics"],
        "criterion": (
            f"test_accuracy >= {accuracy_threshold} "
            f"for {threshold_patience} consecutive checkpoints"
        ),
        "final": history[-1],
        "checkpoints": len(history),
        "phase_metrics": phase_metrics,
    }
    write_json(run_dir / "summary.json", summary)
    model.eval()
    with torch.no_grad():
        all_predictions = model(data.inputs.to(device)).cpu()
    torch.save(
        {"pairs": data.pairs, "predictions": all_predictions, "targets": data.targets},
        run_dir / "predictions.pt",
    )
    if prediction_snapshots:
        torch.save(prediction_snapshots, run_dir / "prediction_snapshots.pt")
    if model_snapshots:
        torch.save(model_snapshots, run_dir / "model_snapshots.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": int(history[-1]["step"]),
        },
        run_dir / "model.pt",
    )
    return run_dir
