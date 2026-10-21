import numpy as np
import perceval as pcvl
import random
from clements import target_value, random_parameters


def build_component(
    dim: int,
    decomposition: list,
    BS_loss: float = 0.0,
    Flat_loss: float = 0.0,
    PS_error: float = 0.0,
    target=None,
    param=None,
) -> pcvl.Processor:
    """
    Builds the lossy component corresponding to a Clements decomposition.
    If a target is specified, the corresponding parameter is replaced by param.

    Args:
        dim (int): The dimension of the circuit.
        decomposition (list): The Clements decomposition as a list of tuples (m, phi, theta).
        BS_loss (float): The loss in the beam splitters (between 0 and 1).
        Flat_loss (float): The loss in the flat components (between 0 and 1).
        PS_error (float): The standard deviation of the Gaussian noise added to the phase shifters (between 0 and π).
        target (tuple): A tuple (index, parameter) specifying which parameter to replace.
        param (float): The value to replace the target parameter with.

    Returns:
        pcvl.Processor: The constructed processor with the specified losses and errors.
    """

    if not isinstance(dim, int):
        raise TypeError("dim must be an integer")
    if dim < 2:
        raise ValueError("dim must be at least 2")

    if not (0 <= BS_loss < 1):
        raise ValueError("BS_loss must be between 0 and 1")

    if not (0 <= Flat_loss < 1):
        raise ValueError("Flat_loss must be between 0 and 1")

    if not (0 <= PS_error <= np.pi):
        raise ValueError("PS_error must be between 0 and π")

    if target is not None:
        if not isinstance(target, (tuple, list)) or len(target) != 2:
            raise ValueError("target must be of the form (index, parameter)")

        if target[1] not in (0, 1):
            raise ValueError("target parameter must be 0 (phi) or 1 (theta)")

        if param is None:
            raise ValueError("param must be provided when target is specified")

    proc = pcvl.Processor("SLOS", pcvl.Circuit(dim))

    for k, (m, phi, theta) in enumerate(decomposition):

        if not (0 <= m < dim - 1):
            raise ValueError(f"Invalid mode index {m} in decomposition")

        effective_phi = phi
        effective_theta = theta

        if target is not None and k == target[0]:
            if target[1] == 0:
                effective_phi = param
            else:
                effective_theta = param

        proc.add(m, pcvl.PS(phi=effective_phi + random.gauss(0, PS_error**2)))

        proc.add(m, pcvl.BS(convention=pcvl.BSConvention.H))
        proc.add(m, pcvl.LC(loss=BS_loss))
        proc.add(m + 1, pcvl.LC(loss=BS_loss))

        proc.add(m, pcvl.PS(phi=effective_theta + random.gauss(0, PS_error**2)))

        proc.add(m, pcvl.BS(convention=pcvl.BSConvention.H))
        proc.add(m, pcvl.LC(loss=BS_loss))
        proc.add(m + 1, pcvl.LC(loss=BS_loss))

    for m in range(dim):
        proc.add(m, pcvl.LC(loss=1 - (1 - Flat_loss) ** dim))

    return proc


def generate_angles(n: int) -> list[float]:
    """
    Generates the angles required for the parameter shift rule.

    Args:
        n (int): The number of photons in the input state.

    Returns:
        list[float]: A list of angles in radians.
    """

    if not isinstance(n, int):
        raise TypeError("n must be an integer")

    if n <= 0:
        raise ValueError("n must be positive")

    return [2 * np.pi * k / (2 * n + 1) for k in range(1, n + 1)]


def generate_coefs(n: int) -> list[float]:
    """
    Generates the coefficients for the parameter shift rule.

    Args:
        n (int): The number of photons in the input state.

    Returns:
        list[float]: A list of coefficients.
    """

    if not isinstance(n, int):
        raise TypeError("n must be an integer")

    if n <= 0:
        raise ValueError("n must be positive")

    mus = generate_angles(n)

    A = np.array([[np.sin(m * mu) for mu in mus] for m in range(1, n + 1)], dtype=float)

    b = 0.5 * np.arange(1, n + 1)

    return np.linalg.solve(A, b)


def compute_psr(
    dim, i, j, decomposition, target, BS_loss=0.0, Flat_loss=0.0, PS_error=0.0
):
    """
    Computes the parameter-shift derivative of a transition probability.
    """

    if not isinstance(i, (list, tuple)) or not isinstance(j, (list, tuple)):
        raise TypeError("i and j must be lists or tuples")

    if len(i) != dim:
        raise ValueError(f"Input state must have length {dim}")
    if len(j) != dim:
        raise ValueError(f"Output state must have length {dim}")

    if any((not isinstance(x, int)) or x < 0 for x in i):
        raise ValueError("Input state must contain non-negative integers")
    if any((not isinstance(x, int)) or x < 0 for x in j):
        raise ValueError("Output state must contain non-negative integers")

    n = sum(i)

    value = target_value(decomposition, target)

    coefs = generate_coefs(n)

    input_state = pcvl.BasicState(i)
    output_state = pcvl.BasicState(j)

    psr = 0.0

    for coef, m in zip(coefs, range(1, n + 1)):

        shift = 2 * np.pi * m / (2 * n + 1)

        proc_plus = pcvl.Processor("SLOS", pcvl.Circuit(dim))
        proc_minus = pcvl.Processor("SLOS", pcvl.Circuit(dim))

        component_plus = build_component(
            dim, decomposition, BS_loss, Flat_loss, PS_error, target, value + shift
        )
        component_minus = build_component(
            dim, decomposition, BS_loss, Flat_loss, PS_error, target, value - shift
        )

        proc_plus.add(0, component_plus)
        proc_minus.add(0, component_minus)

        proc_plus.min_detected_photons_filter(0)
        proc_minus.min_detected_photons_filter(0)

        proc_plus.with_input(input_state)
        proc_minus.with_input(input_state)

        result_plus = proc_plus.probs(1e-4)
        result_minus = proc_minus.probs(1e-4)

        psr_plus = result_plus["results"][output_state]
        psr_minus = result_minus["results"][output_state]

        psr += coef * (psr_plus - psr_minus)

    return psr


def finite_diff(
    dim,
    i,
    j,
    decomposition,
    target,
    BS_loss=0.0,
    Flat_loss=0.0,
    PS_error=0.0,
    delta=1e-4,
):
    """Computes the finite difference approximation of the derivative of the transition probability"""
    value = target_value(decomposition, target)
    input_state = pcvl.BasicState(i)
    output_state = pcvl.BasicState(j)
    proc_plus = pcvl.Processor("SLOS", pcvl.Circuit(dim))
    proc_minus = pcvl.Processor("SLOS", pcvl.Circuit(dim))
    component_plus = build_component(
        dim, decomposition, BS_loss, Flat_loss, PS_error, target, value + delta
    )
    component_minus = build_component(
        dim, decomposition, BS_loss, Flat_loss, PS_error, target, value - delta
    )
    proc_plus.add(0, component_plus)
    proc_minus.add(0, component_minus)

    proc_plus.min_detected_photons_filter(0)
    proc_minus.min_detected_photons_filter(0)

    proc_plus.with_input(input_state)
    proc_minus.with_input(input_state)

    result_plus = proc_plus.probs(1e-4)
    result_minus = proc_minus.probs(1e-4)

    finite_plus = result_plus["results"][output_state]
    finite_minus = result_minus["results"][output_state]

    return (finite_plus - finite_minus) / (2 * delta)
