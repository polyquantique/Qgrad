import numpy as np
import pytest

from qgrad.clements import (
    clements_indices,
    random_parameters,
    target_value,
)


def test_clements_indices_dim_2():
    assert clements_indices(2) == [0]


def test_clements_indices_dim_3():
    assert clements_indices(3) == [0, 1, 0]


def test_clements_indices_length():
    for dim in range(2, 10):
        indices = clements_indices(dim)

        # Clements decomposition contains n(n-1)/2 MZIs
        expected = dim * (dim - 1) // 2
        assert len(indices) == expected


def test_clements_indices_range():
    for dim in range(2, 10):
        indices = clements_indices(dim)

        assert all(0 <= m < dim - 1 for m in indices)


def test_clements_indices_non_integer():
    with pytest.raises(TypeError):
        clements_indices(3.5)


def test_clements_indices_too_small():
    with pytest.raises(ValueError):
        clements_indices(1)


def test_random_parameters_length():
    dim = 5
    decomposition = random_parameters(dim)

    expected = dim * (dim - 1) // 2
    assert len(decomposition) == expected


def test_random_parameters_structure():
    dim = 4
    decomposition = random_parameters(dim)

    for m, phi, theta in decomposition:
        assert isinstance(m, int)
        assert isinstance(phi, float)
        assert isinstance(theta, float)


def test_random_parameters_indices_match_clements():
    dim = 6

    decomposition = random_parameters(dim)
    expected_indices = clements_indices(dim)

    actual_indices = [m for m, _, _ in decomposition]

    assert actual_indices == expected_indices


def test_random_parameters_angle_range():
    dim = 5

    decomposition = random_parameters(dim)

    for _, phi, theta in decomposition:
        assert 0 <= phi <= 2 * np.pi
        assert 0 <= theta <= 2 * np.pi


@pytest.fixture
def sample_decomposition():
    return [
        (0, 1.0, 2.0),
        (1, 3.0, 4.0),
        (0, 5.0, 6.0),
    ]


def test_target_value_phi(sample_decomposition):
    assert target_value(sample_decomposition, (1, 0)) == 3.0


def test_target_value_theta(sample_decomposition):
    assert target_value(sample_decomposition, (1, 1)) == 4.0


def test_target_value_first_element(sample_decomposition):
    assert target_value(sample_decomposition, (0, 0)) == 1.0
    assert target_value(sample_decomposition, (0, 1)) == 2.0


def test_target_value_last_element(sample_decomposition):
    assert target_value(sample_decomposition, (2, 0)) == 5.0
    assert target_value(sample_decomposition, (2, 1)) == 6.0


def test_target_value_invalid_decomposition_type():
    with pytest.raises(TypeError):
        target_value("not a decomposition", (0, 0))


def test_target_value_index_out_of_range(sample_decomposition):
    with pytest.raises(IndexError):
        target_value(sample_decomposition, (10, 0))
