"""Spectral diagnostics for modular-arithmetic solutions."""

from __future__ import annotations

import torch


def validate_snapshot_steps(expected: dict[str, int | None], observed: dict[str, int]) -> None:
    """Require phase states to match the declared start of each sustained phase."""

    for phase, expected_step in expected.items():
        observed_step = observed.get(phase)
        if expected_step is None:
            continue
        if observed_step is None:
            raise ValueError(f"missing snapshot for {phase}")
        if observed_step != expected_step:
            raise ValueError(f"{phase} snapshot is step {observed_step}, expected {expected_step}")


def correct_class_score_table(predictions: torch.Tensor, p: int) -> torch.Tensor:
    """Return the score assigned to the correct class for every (a, b)."""

    if predictions.shape != (p * p, p):
        raise ValueError("predictions must have shape [p*p, p]")
    pairs = torch.tensor([(a, b) for a in range(p) for b in range(p)])
    labels = (pairs[:, 0] + pairs[:, 1]).remainder(p)
    return predictions[torch.arange(p * p), labels].reshape(p, p)


def fourier_power(table: torch.Tensor) -> torch.Tensor:
    """Centered normalized 2-D Fourier power spectrum."""

    if table.ndim != 2:
        raise ValueError("table must be two-dimensional")
    centered = table - table.mean()
    spectrum = torch.fft.fftshift(torch.fft.fft2(centered))
    power = spectrum.abs().square()
    return power / power.sum().clamp_min(torch.finfo(power.dtype).eps)


def classwise_fourier_power(logits: torch.Tensor, p: int) -> torch.Tensor:
    """Normalized 2-D Fourier power for every output class, [class, fa, fb]."""

    if logits.shape != (p * p, p):
        raise ValueError("logits must have shape [p*p, p]")
    tables = logits.reshape(p, p, p).permute(2, 0, 1)
    tables = tables - tables.mean(dim=(1, 2), keepdim=True)
    spectra = torch.fft.fft2(tables, dim=(-2, -1))
    power = spectra.abs().square()
    normalizer = power.sum(dim=(-2, -1), keepdim=True).clamp_min(torch.finfo(power.dtype).eps)
    return power / normalizer


def mean_centered_fourier_spectrum(logits: torch.Tensor, p: int) -> torch.Tensor:
    """Mean normalized classwise spectrum with signed frequencies centered."""

    return torch.fft.fftshift(classwise_fourier_power(logits, p).mean(dim=0))


def modular_diagonal_power_fraction(logits: torch.Tensor, p: int) -> float:
    """Average Fourier power on frequencies fa == fb, excluding DC."""

    power = classwise_fourier_power(logits, p)
    indices = torch.arange(p)
    diagonal = power[:, indices, indices].sum(dim=1) - power[:, 0, 0]
    non_dc = 1.0 - power[:, 0, 0]
    return float((diagonal / non_dc.clamp_min(1e-12)).mean().item())


def pca_summary(
    representations: torch.Tensor, components: int = 2
) -> dict[str, torch.Tensor | float]:
    """PCA projection, explained ratios, and effective participation rank."""

    if representations.ndim != 2 or representations.shape[0] < 2:
        raise ValueError("representations must be [examples, features]")
    centered = representations.float() - representations.float().mean(dim=0, keepdim=True)
    _, singular_values, right = torch.linalg.svd(centered, full_matrices=False)
    variances = singular_values.square()
    ratios = variances / variances.sum().clamp_min(1e-12)
    count = min(components, right.shape[0])
    projection = centered @ right[:count].t()
    participation_rank = variances.sum().square() / variances.square().sum().clamp_min(1e-12)
    return {
        "projection": projection,
        "explained_variance_ratio": ratios[:count],
        "participation_rank": float(participation_rank.item()),
    }
