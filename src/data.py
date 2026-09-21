"""Modular-addition data for faithful and controlled grokking experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch.utils.data import TensorDataset

GromovEncoding = Literal["one_hot", "circular"]
SplitMode = Literal["random", "symmetry_safe"]


@dataclass(frozen=True)
class GromovAdditionData:
    p: int
    input_encoding: GromovEncoding
    split_mode: SplitMode
    pairs: torch.Tensor
    inputs: torch.Tensor
    targets: torch.Tensor
    train_indices: torch.Tensor
    test_indices: torch.Tensor
    train: TensorDataset
    test: TensorDataset


def encode_circular_pairs(pairs: torch.Tensor, p: int) -> torch.Tensor:
    """Encode each residue by its first sine/cosine harmonic."""

    angles = 2.0 * torch.pi * pairs.to(torch.float32) / float(p)
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=1)


def _unordered_pair_groups(p: int) -> list[list[int]]:
    groups: list[list[int]] = []
    for a in range(p):
        for b in range(a, p):
            first = a * p + b
            groups.append([first] if a == b else [first, b * p + a])
    return groups


def symmetry_safe_split(
    p: int, train_fraction: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split pairs while keeping the commutative orbit (a,b)/(b,a) together."""

    if p < 2:
        raise ValueError("p must be at least 2")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    groups = _unordered_pair_groups(p)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(groups), generator=generator).tolist()
    target = round(p * p * train_fraction)
    train: list[int] = []
    test: list[int] = []
    for group_index in order:
        group = groups[group_index]
        (train if len(train) + len(group) <= target else test).extend(group)
    if len(train) < target:
        remaining = [group for group in groups if all(index not in train for index in group)]
        candidates = [group for group in remaining if len(train) + len(group) <= target]
        if candidates:
            chosen = min(candidates, key=lambda group: target - len(train) - len(group))
            for index in chosen:
                test.remove(index)
            train.extend(chosen)
    return (
        torch.tensor(sorted(train), dtype=torch.long),
        torch.tensor(sorted(test), dtype=torch.long),
    )


def random_split(
    total_examples: int, train_examples: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0 < train_examples < total_examples:
        raise ValueError("train_examples must be between 1 and total_examples-1")
    generator = torch.Generator().manual_seed(seed)
    train = torch.randperm(total_examples, generator=generator)[:train_examples].sort().values
    mask = torch.zeros(total_examples, dtype=torch.bool)
    mask[train] = True
    test = torch.arange(total_examples, dtype=torch.long)[~mask]
    return train, test


def make_gromov_addition_data(
    p: int = 97,
    train_fraction: float = 0.5,
    split_seed: int = 0,
    split_mode: SplitMode = "random",
    input_encoding: GromovEncoding = "one_hot",
) -> GromovAdditionData:
    """Create all pairs in Z_p x Z_p with one-hot output targets."""

    if p < 2:
        raise ValueError("p must be at least 2")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    pairs = torch.tensor([(a, b) for a in range(p) for b in range(p)], dtype=torch.long)
    labels = (pairs[:, 0] + pairs[:, 1]).remainder(p)
    if split_mode == "random":
        train_indices, test_indices = random_split(
            len(pairs), round(len(pairs) * train_fraction), split_seed
        )
    elif split_mode == "symmetry_safe":
        train_indices, test_indices = symmetry_safe_split(p, train_fraction, split_seed)
    else:
        raise ValueError(f"Unknown split mode: {split_mode}")

    if input_encoding == "one_hot":
        inputs = torch.zeros((len(pairs), 2 * p), dtype=torch.float32)
        rows = torch.arange(len(pairs))
        inputs[rows, pairs[:, 0]] = 1.0
        inputs[rows, p + pairs[:, 1]] = 1.0
    elif input_encoding == "circular":
        inputs = encode_circular_pairs(pairs, p)
    else:
        raise ValueError(f"Unknown input encoding: {input_encoding}")

    targets = torch.zeros((len(pairs), p), dtype=torch.float32)
    targets[torch.arange(len(pairs)), labels] = 1.0
    return GromovAdditionData(
        p=p,
        input_encoding=input_encoding,
        split_mode=split_mode,
        pairs=pairs,
        inputs=inputs,
        targets=targets,
        train_indices=train_indices,
        test_indices=test_indices,
        train=TensorDataset(inputs[train_indices], targets[train_indices]),
        test=TensorDataset(inputs[test_indices], targets[test_indices]),
    )
