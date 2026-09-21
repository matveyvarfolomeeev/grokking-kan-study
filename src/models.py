"""Neural architectures used in the controlled grokking study."""

from __future__ import annotations

import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class GromovQuadraticNetwork(nn.Module):
    """Bias-free quadratic MLP following Gromov's modular-addition setup."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        width: int = 128,
        activation: str = "quadratic",
        initialization_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        if activation not in {"quadratic", "relu", "gelu"}:
            raise ValueError(f"Unknown activation: {activation}")
        self.input_dim = input_dim
        self.width = width
        self.activation_name = activation
        self.initialization_scale = initialization_scale
        self.first = nn.Parameter(torch.empty(width, input_dim))
        self.second = nn.Parameter(torch.empty(output_dim, width))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Gromov initializes both matrices independently from N(0, 1).
        nn.init.normal_(self.first, mean=0.0, std=self.initialization_scale)
        nn.init.normal_(self.second, mean=0.0, std=self.initialization_scale)

    def hidden_representation(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs @ self.first.t() / (self.input_dim**0.5)
        if self.activation_name == "quadratic":
            return hidden.square()
        if self.activation_name == "relu":
            return torch.relu(hidden)
        return torch.nn.functional.gelu(hidden)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (self.hidden_representation(inputs) @ self.second.t()) / self.width


class BSplineKANLayer(nn.Module):
    """KAN layer with a learned B-spline function on every directed edge."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-2.0, 2.0),
    ) -> None:
        super().__init__()
        if grid_size < 1 or spline_order < 1:
            raise ValueError("grid_size and spline_order must be positive")
        if grid_range[0] >= grid_range[1]:
            raise ValueError("grid_range must be increasing")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.num_basis = grid_size + spline_order
        step = (grid_range[1] - grid_range[0]) / grid_size
        knots = (
            torch.arange(
                -spline_order,
                grid_size + spline_order + 1,
                dtype=torch.float32,
            )
            * step
            + grid_range[0]
        )
        self.register_buffer("knots", knots)
        self.base_weight = nn.Parameter(torch.empty(output_dim, input_dim))
        self.spline_weight = nn.Parameter(torch.empty(output_dim, input_dim, self.num_basis))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=5**0.5)
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.02)

    def b_spline_basis(self, inputs: torch.Tensor) -> torch.Tensor:
        """Evaluate the fixed basis, returning [batch, input, basis]."""

        values = inputs.unsqueeze(-1)
        basis = ((values >= self.knots[:-1]) & (values < self.knots[1:])).to(inputs.dtype)
        # Cox-de Boor raises the piecewise-constant basis to the requested order.
        for order in range(1, self.spline_order + 1):
            left_denominator = self.knots[order:-1] - self.knots[: -(order + 1)]
            right_denominator = self.knots[order + 1 :] - self.knots[1:-order]
            left = (values - self.knots[: -(order + 1)]) / left_denominator
            right = (self.knots[order + 1 :] - values) / right_denominator
            basis = left * basis[..., :-1] + right * basis[..., 1:]
        return basis

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        activated = torch.nn.functional.silu(inputs)
        return self.forward_from_basis(activated, self.b_spline_basis(inputs))

    def forward_from_basis(
        self, activated_inputs: torch.Tensor, spline_basis: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate trainable weights from cached input-only features."""

        base = torch.nn.functional.linear(activated_inputs, self.base_weight)
        spline = torch.einsum("bik,oik->bo", spline_basis, self.spline_weight)
        return base + spline

    def edge_values(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return every edge contribution with shape [batch, output, input]."""

        base = torch.nn.functional.silu(inputs).unsqueeze(1) * self.base_weight.unsqueeze(0)
        spline = torch.einsum("bik,oik->boi", self.b_spline_basis(inputs), self.spline_weight)
        return base + spline


class SplineKAN(nn.Module):
    """Two-layer spline-KAN used as the architecture-only extension."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        width: int = 14,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-2.0, 2.0),
    ) -> None:
        super().__init__()
        self.first = BSplineKANLayer(input_dim, width, grid_size, spline_order, grid_range)
        self.second = BSplineKANLayer(width, output_dim, grid_size, spline_order, grid_range)

    def hidden_representation(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.first(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.second(self.hidden_representation(inputs))

    def cache_first_layer_inputs(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.nn.functional.silu(inputs), self.first.b_spline_basis(inputs)

    def forward_with_first_layer_cache(
        self, cache: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        activated_inputs, spline_basis = cache
        return self.second(self.first.forward_from_basis(activated_inputs, spline_basis))
