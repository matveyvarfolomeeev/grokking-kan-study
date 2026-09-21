import math

import torch

from src.data import (
    encode_circular_pairs,
    make_gromov_addition_data,
    random_split,
    symmetry_safe_split,
)


def test_circular_encoding_shape() -> None:
    encoded = encode_circular_pairs(torch.tensor([[0, 1]]), p=7)
    assert encoded.shape == (1, 4)
    assert torch.allclose(encoded[0, :2], torch.tensor([1.0, math.cos(2 * math.pi / 7)]))


def test_symmetry_safe_split_is_partition() -> None:
    train, test = symmetry_safe_split(p=11, train_fraction=0.5, seed=0)
    assert set(train.tolist()).isdisjoint(test.tolist())
    assert sorted(train.tolist() + test.tolist()) == list(range(121))
    train_set = set(train.tolist())
    for a in range(11):
        for b in range(11):
            left = a * 11 + b in train_set
            right = b * 11 + a in train_set
            assert left == right


def test_split_is_reproducible_and_seeded() -> None:
    first = symmetry_safe_split(13, 0.3, 42)
    second = symmetry_safe_split(13, 0.3, 42)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_random_split_is_reproducible_partition() -> None:
    first = random_split(total_examples=49, train_examples=17, seed=4)
    second = random_split(total_examples=49, train_examples=17, seed=4)
    assert torch.equal(first[0], second[0])
    assert set(first[0].tolist()).isdisjoint(first[1].tolist())
    assert sorted(first[0].tolist() + first[1].tolist()) == list(range(49))


def test_gromov_data_is_one_hot() -> None:
    data = make_gromov_addition_data(p=5, train_fraction=0.5, split_seed=0)
    inputs, targets = data.train.tensors
    assert inputs.shape[1] == 10
    assert targets.shape[1] == 5
    assert torch.all(inputs.sum(dim=1) == 2)
    assert torch.all(targets.sum(dim=1) == 1)


def test_gromov_circular_encoding_keeps_same_task_and_split() -> None:
    one_hot = make_gromov_addition_data(p=7, train_fraction=0.5, split_seed=3)
    circular = make_gromov_addition_data(
        p=7, train_fraction=0.5, split_seed=3, input_encoding="circular"
    )
    assert circular.inputs.shape == (49, 4)
    assert torch.equal(circular.targets, one_hot.targets)
    assert torch.equal(circular.train_indices, one_hot.train_indices)


def test_gromov_symmetry_safe_mode_keeps_mirrors_together() -> None:
    data = make_gromov_addition_data(
        p=7, train_fraction=0.5, split_seed=2, split_mode="symmetry_safe"
    )
    train = set(data.train_indices.tolist())
    for a in range(7):
        for b in range(7):
            assert (a * 7 + b in train) == (b * 7 + a in train)
