import torch

from src.analysis import mean_centered_fourier_spectrum
from src.metrics import (
    canonical_learning_dynamics,
    classify_learning_dynamics,
    find_grok_step,
)
from src.train import squared_l2_loss


def test_grok_step_requires_consecutive_checkpoints() -> None:
    history = [
        {"step": 10, "test_accuracy": 1.0},
        {"step": 20, "test_accuracy": 0.98},
        {"step": 30, "test_accuracy": 1.0},
        {"step": 40, "test_accuracy": 1.0},
        {"step": 50, "test_accuracy": 1.0},
        {"step": 60, "test_accuracy": 1.0},
    ]
    assert find_grok_step(history, threshold=0.99, patience=4) == 30
    assert find_grok_step(history, threshold=0.99, patience=5) is None


def test_squared_l2_loss_sums_outputs_and_means_examples() -> None:
    predictions = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
    targets = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    assert squared_l2_loss(predictions, targets).item() == 12.5


def test_classical_grokking_requires_memorization_plateau() -> None:
    history = []
    for step in range(0, 5001, 500):
        train = 1.0 if step >= 1000 else step / 1000
        test = 0.05 if step < 3500 else min(1.0, 0.05 + (step - 3000) / 1000)
        history.append({"step": step, "train_accuracy": train, "test_accuracy": test})
    phases = classify_learning_dynamics(history, patience=2, min_plateau_steps=1000)
    assert phases["dynamics"] == "classical_grokking"
    assert phases["memorization_step"] == 1000
    assert phases["generalization_step"] == 4000


def test_joint_growth_is_not_classical_grokking() -> None:
    history = [
        {"step": 0, "train_accuracy": 0.10, "test_accuracy": 0.08},
        {"step": 500, "train_accuracy": 0.50, "test_accuracy": 0.47},
        {"step": 1000, "train_accuracy": 0.995, "test_accuracy": 0.992},
        {"step": 1500, "train_accuracy": 1.0, "test_accuracy": 0.999},
        {"step": 2000, "train_accuracy": 1.0, "test_accuracy": 1.0},
    ]
    phases = classify_learning_dynamics(history, patience=2)
    assert phases["dynamics"] == "joint_generalization"
    assert not phases["is_classical_grokking"]


def test_legacy_grokked_status_is_normalized() -> None:
    assert canonical_learning_dynamics({"status": "grokked"}) == "classical_grokking"


def test_phase_metrics_override_legacy_status() -> None:
    summary = {
        "status": "grokked",
        "phase_metrics": {
            "dynamics": "memorized_not_generalized",
            "is_classical_grokking": False,
        },
    }
    assert canonical_learning_dynamics(summary) == "memorized_not_generalized"


def test_centered_fourier_spectrum_recovers_modular_wave() -> None:
    p = 7
    pairs = torch.tensor([(a, b) for a in range(p) for b in range(p)])
    phase = 2 * torch.pi * (pairs[:, 0] + pairs[:, 1]).float() / p
    wave = torch.cos(phase)
    logits = wave[:, None].repeat(1, p)

    spectrum = mean_centered_fourier_spectrum(logits, p)
    assert torch.isclose(spectrum.sum(), torch.tensor(1.0), atol=1e-6)

    frequencies = torch.fft.fftshift(torch.fft.fftfreq(p) * p).round().to(torch.int64)
    flat_indices = torch.topk(spectrum.flatten(), 2).indices
    coordinates = {
        (
            int(frequencies[index // p]),
            int(frequencies[index % p]),
        )
        for index in flat_indices
    }
    assert coordinates == {(-1, -1), (1, 1)}
