import pytest
import torch

from src.analysis import (
    classwise_fourier_power,
    correct_class_score_table,
    fourier_power,
    modular_diagonal_power_fraction,
    pca_summary,
    validate_snapshot_steps,
)
from src.models import BSplineKANLayer, GromovQuadraticNetwork, SplineKAN, count_parameters


def test_b_spline_basis_is_partition_of_unity_inside_grid() -> None:
    layer = BSplineKANLayer(2, 3, grid_size=5, spline_order=3)
    inputs = torch.tensor([[-1.5, 0.0], [0.5, 1.5]])
    basis = layer.b_spline_basis(inputs)
    assert basis.shape == (2, 2, 8)
    assert torch.allclose(basis.sum(dim=-1), torch.ones(2, 2), atol=1e-5)


def test_kan_forward_and_parameter_matching() -> None:
    mlp = GromovQuadraticNetwork(194, 97, width=128)
    kan = SplineKAN(194, 97, width=14, grid_size=5, spline_order=3)
    output = kan(torch.zeros(4, 194))
    cached_output = kan.forward_with_first_layer_cache(
        kan.cache_first_layer_inputs(torch.zeros(4, 194))
    )
    assert output.shape == (4, 97)
    assert torch.allclose(output, cached_output)
    ratio = count_parameters(kan) / count_parameters(mlp)
    assert 0.9 <= ratio <= 1.1


def test_fourier_analysis_shapes_and_normalization() -> None:
    p = 5
    predictions = torch.randn(p * p, p)
    table = correct_class_score_table(predictions, p)
    power = fourier_power(table)
    assert table.shape == (p, p)
    assert power.shape == (p, p)
    assert torch.allclose(power.sum(), torch.tensor(1.0), atol=1e-5)


def test_classwise_fourier_detects_modular_diagonal() -> None:
    p = 7
    pairs = torch.tensor([(a, b) for a in range(p) for b in range(p)])
    labels = (pairs[:, 0] + pairs[:, 1]).remainder(p)
    logits = torch.nn.functional.one_hot(labels, p).float()
    power = classwise_fourier_power(logits, p)
    assert power.shape == (p, p, p)
    assert torch.allclose(power.sum(dim=(1, 2)), torch.ones(p), atol=1e-5)
    assert modular_diagonal_power_fraction(logits, p) > 0.999


def test_pca_summary_returns_projection_and_effective_rank() -> None:
    values = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    result = pca_summary(values)
    assert result["projection"].shape == (10, 2)
    assert result["explained_variance_ratio"][0] > 0.99
    assert 1.0 <= result["participation_rank"] <= 2.0


def test_phase_snapshot_must_match_sustained_phase_start() -> None:
    validate_snapshot_steps(
        {"memorization": 100, "generalization": 900},
        {"memorization": 100, "generalization": 900},
    )
    with pytest.raises(ValueError, match="generalization snapshot is step 1300"):
        validate_snapshot_steps(
            {"memorization": 100, "generalization": 900},
            {"memorization": 100, "generalization": 1300},
        )


def test_kan_edge_values_sum_to_layer_output() -> None:
    layer = BSplineKANLayer(3, 2)
    inputs = torch.randn(5, 3)
    assert torch.allclose(layer.edge_values(inputs).sum(dim=2), layer(inputs), atol=1e-5)
