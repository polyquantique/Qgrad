import numpy as np
import perceval as pcvl
import pytest

from qgrad.clements import (
    target_value,
    random_parameters,
)
from qgrad.qdiff_theo import (
    build_component,
    generate_angles,
    generate_coefs,
    compute_psr,
    finite_diff,
)


def test_generate_angles_basic():
    angles = generate_angles(3)

    assert len(angles) == 3
    assert all(isinstance(a, float) for a in angles)


def test_generate_angles_values():
    n = 2
    angles = generate_angles(n)

    expected = [
        2 * np.pi * 1 / (2 * n + 1),
        2 * np.pi * 2 / (2 * n + 1),
    ]

    np.testing.assert_allclose(angles, expected)


def test_generate_angles_invalid():
    with pytest.raises(TypeError):
        generate_angles(2.5)

    with pytest.raises(ValueError):
        generate_angles(0)


def test_generate_coefs_shape():
    n = 3
    coefs = generate_coefs(n)

    assert len(coefs) == n


def test_generate_coefs_deterministic():
    # This should be deterministic given fixed n
    coefs1 = generate_coefs(3)
    coefs2 = generate_coefs(3)

    np.testing.assert_allclose(coefs1, coefs2)


def test_generate_coefs_invalid():
    with pytest.raises(TypeError):
        generate_coefs("3")

    with pytest.raises(ValueError):
        generate_coefs(-1)


def test_build_component_invalid_dim():
    with pytest.raises(TypeError):
        build_component(" 3", [])

    with pytest.raises(ValueError):
        build_component(1, [])


def test_build_component_loss_bounds():
    with pytest.raises(ValueError):
        build_component(3, [], BS_loss=1.5)

    with pytest.raises(ValueError):
        build_component(3, [], Flat_loss=-0.1)

    with pytest.raises(ValueError):
        build_component(3, [], PS_error=-0.1)


def test_build_component_invalid_target():
    with pytest.raises(ValueError):
        build_component(3, [], target="bad")

    with pytest.raises(ValueError):
        build_component(3, [], target=(0, 2))  # invalid parameter index


def test_build_component_requires_param():
    with pytest.raises(ValueError):
        build_component(3, [], target=(0, 0))


def test_build_component_runs_simple_case():
    dim = 3

    decomposition = [
        (0, 0.1, 0.2),
        (1, 0.3, 0.4),
        (0, 0.5, 0.6),
    ]

    proc = build_component(dim, decomposition)

    assert proc is not None


@pytest.mark.parametrize(
    "dim, i, j, decomposition, target, BS_loss, Flat_loss, PS_error",
    [
        (3, (1, 2, 1), (0, 1, 2), random_parameters(3), (1, 1), 0.05, 0.01, 0.0),
        (4, (1, 1, 1, 1), (2, 0, 2, 0), random_parameters(4), (2, 0), 0.0, 0.0, 0.0),
    ],
)
def test_compute_psr(dim, i, j, decomposition, target, BS_loss, Flat_loss, PS_error):
    psr = compute_psr(dim, i, j, decomposition, target, BS_loss, Flat_loss, PS_error)
    finite = finite_diff(dim, i, j, decomposition, target, BS_loss, Flat_loss, PS_error)

    assert np.isclose(psr, finite, atol=1e-4)
