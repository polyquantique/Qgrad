TOKEN = "Valid Token"
PLATFORM = "Valid QPU Platform"

from unittest import result

import numpy as np
import perceval as pcvl
from qdiff_theo import generate_coefs, finite_diff


def build_parameterized_component(
    dim: int, decomposition: list[tuple], target: tuple
) -> tuple[pcvl.Circuit, pcvl.Parameter]:
    """
    Builds a parameterized Clements component.

    Args:
        dim (int): The dimension of the component.
        decomposition (list[tuple]): The decomposition of the component.
        target (tuple): The target parameter.
    Returns:
        tuple: A tuple containing the parameterized component and the parameter.

    """

    circuit = pcvl.Circuit(dim)

    parameter = None

    for k, (m, phi, theta) in enumerate(decomposition):

        effective_phi = phi
        effective_theta = theta

        if k == target[0]:

            if target[1] == 0:
                parameter = pcvl.Parameter("phi_param")
                effective_phi = parameter

            elif target[1] == 1:
                parameter = pcvl.Parameter("theta_param")
                effective_theta = parameter

            else:
                raise ValueError("target parameter must be 0 (phi) or 1 (theta)")

        circuit.add(m, pcvl.PS(phi=effective_phi))

        circuit.add(m, pcvl.BS(convention=pcvl.BSConvention.H))

        circuit.add(m, pcvl.PS(phi=effective_theta))

        circuit.add(m, pcvl.BS(convention=pcvl.BSConvention.H))

    if parameter is None:
        raise ValueError("Target parameter was not found in decomposition")

    return circuit, parameter


def probability_from_sampler(
    sampler: pcvl.algorithm.Sampler, output_state: pcvl.BasicState, shots: int = 1000
) -> float:
    """
    Computes the probability of measuring the output_state using the sampler.

    Args:
        sampler: The sampler object.
        output_state: The output state to measure.
        shots (int): The number of shots to use for the measurement.

    Returns:
        float: The probability of measuring the output_state.
    """
    result = sampler.sample_count(shots)

    if result is None:
        raise RuntimeError("Sampler returned no result.")

    counts = result["results"]
    target_counts = counts[output_state] if output_state in counts else 0
    total_counts = sum(counts.values())

    if total_counts == 0:
        return 0.0

    return target_counts / total_counts


def parameter_shift_gradient(
    sampler: pcvl.algorithm.Sampler,
    parameter: pcvl.Parameter,
    parameter_value: float,
    output_state: pcvl.BasicState,
    n_photons: int,
    shots: int = 1000,
) -> float:
    """
    Generalized parameter shift rule.

    Args:
        sampler: The sampler object.
        parameter: The parameter object.
        parameter_value: The current value of the parameter.
        output_state: The output state to measure.
        n_photons: The number of photons in the input state.
        shots: The number of shots to use for the measurement.

    Returns:
        float: The gradient of the probability with respect to the parameter.
    """

    coefs = generate_coefs(n_photons)

    gradient = 0.0

    for coef, m in zip(coefs, range(1, n_photons + 1)):

        shift = 2 * np.pi * m / (2 * n_photons + 1)

        # positive shift
        parameter.set_value(parameter_value + shift)

        p_plus = probability_from_sampler(sampler, output_state, shots)

        # negative shift
        parameter.set_value(parameter_value - shift)

        p_minus = probability_from_sampler(sampler, output_state, shots)

        gradient += coef * (p_plus - p_minus)

    parameter.set_value(parameter_value)

    return gradient


def gradient_descent_parameter_shift(
    dim: int,
    i: tuple[int, ...],
    j: tuple[int, ...],
    decomposition: pcvl.Circuit,
    target: tuple[int, int],
    initial_parameter: float,
    learning_rate: float = 0.1,
    max_iter: int = 50,
    shots: int = 1000,
) -> tuple[float, np.ndarray]:
    """
    Performs gradient descent using the generalized parameter shift rule.

    Args:
        dim (int): The dimension of the component.
        i (tuple[int, ...]): The input state.
        j (tuple[int, ...]): The output state.
        decomposition (pcvl.Circuit): The circuit decomposition.
        target (tuple[int, int]): The target state.
        initial_parameter (float): The initial value of the parameter.
        learning_rate (float, optional): The learning rate. Defaults to 0.1.
        max_iter (int, optional): The maximum number of iterations. Defaults to 50.
        shots (int, optional): The number of shots to use for the measurement. Defaults to 1000.

    Returns:
        tuple[float, np.ndarray]: The optimized parameter value and the cost function values.
    """

    input_state = pcvl.BasicState(i)
    output_state = pcvl.BasicState(j)

    component, parameter = build_parameterized_component(dim, decomposition, target)

    proc = pcvl.RemoteProcessor(PLATFORM, TOKEN)

    proc.add(0, component)

    proc.min_detected_photons_filter(0)

    proc.with_input(input_state)

    sampler = pcvl.algorithm.Sampler(proc, max_shots_per_call=1000000)

    parameter_value = initial_parameter

    parameter.set_value(parameter_value)

    cost = []

    n_photons = sum(i)

    for k in range(max_iter):

        grad = parameter_shift_gradient(
            sampler, parameter, parameter_value, output_state, n_photons, shots
        )

        parameter_value -= learning_rate * grad

        parameter_value = parameter_value % (2 * np.pi)

        parameter.set_value(parameter_value)

        prob = probability_from_sampler(sampler, output_state, shots)

        cost.append(prob)

        print(
            f"Iteration {k:3d} | "
            f"P={prob:.6f} | "
            f"grad={grad:.6f} | "
            f"theta={parameter_value:.6f}"
        )

    return parameter_value, np.array(cost)


def optimize_delta(
    dim: int,
    i: tuple[int, ...],
    j: tuple[int, ...],
    decomposition: pcvl.Circuit,
    target: tuple[int, int],
    parameter_value: float,
    deltas: list[float],
    shots: int = 1000,
) -> list[float]:
    """
    Optimizes the delta value for finite difference gradient estimation.

    Args:
        dim (int): The dimension of the component.
        i (tuple[int, ...]): The input state.
        j (tuple[int, ...]): The output state.
        decomposition (pcvl.Circuit): The circuit decomposition.
        target (tuple[int, int]): The target state.
        parameter_value (float): The current value of the parameter.
        deltas (list[float]): The list of delta values to test.
        shots (int, optional): The number of shots to use for the measurement. Defaults to 1000.

    Returns:
        list[float]: The list of absolute errors for each delta value.
    """
    input_state = pcvl.BasicState(i)
    output_state = pcvl.BasicState(j)

    component, parameter = build_parameterized_component(dim, decomposition, target)

    expected = finite_diff(
        dim,
        i,
        j,
        decomposition,
        target,
        BS_loss=0.0,
        Flat_loss=0.0,
        PS_error=0.0,
        delta=1e-7,
    )

    proc = pcvl.RemoteProcessor(PLATFORM, TOKEN)

    proc.add(0, component)

    proc.min_detected_photons_filter(0)

    proc.with_input(input_state)

    sampler = pcvl.algorithm.Sampler(proc, max_shots_per_call=1000000)

    abs_err = []
    for delta in deltas:
        # ---- P(theta + delta)
        parameter.set_value(parameter_value + delta)
        prob_plus = probability_from_sampler(sampler, output_state, shots)

        # ---- P(theta - delta)
        parameter.set_value(parameter_value - delta)
        prob_minus = probability_from_sampler(sampler, output_state, shots)

        # Central finite difference
        grad = (prob_plus - prob_minus) / (2 * delta)

        abs_err.append(abs(expected - grad))

    return abs_err


def gradient_descent_finite_difference(
    dim: int,
    i: tuple[int, ...],
    j: tuple[int, ...],
    decomposition: pcvl.Circuit,
    target: tuple[int, int],
    initial_parameter: float,
    learning_rate: float = 0.1,
    max_iter: int = 50,
    shots: int = 2000,
    delta: float = 1e-4,
) -> tuple[float, np.ndarray]:
    """
    Performs gradient descent using finite difference gradient estimation.

    Args:
        dim (int): The dimension of the component.
        i (tuple[int, ...]): The input state.
        j (tuple[int, ...]): The output state.
        decomposition (pcvl.Circuit): The circuit decomposition.
        target (tuple[int, int]): The target state.
        initial_parameter (float): The initial value of the parameter.
        learning_rate (float, optional): The learning rate for gradient descent. Defaults to 0.1.
        max_iter (int, optional): The maximum number of iterations. Defaults to 50.
        shots (int, optional): The number of shots to use for the measurement. Defaults to 2000.
        delta (float, optional): The delta value for finite difference estimation. Defaults to 1e-4.

    Returns:
        tuple[float, np.ndarray]: The optimized parameter value and the cost function values over iterations.
    """

    input_state = pcvl.BasicState(i)
    output_state = pcvl.BasicState(j)

    component, parameter = build_parameterized_component(dim, decomposition, target)

    proc = pcvl.RemoteProcessor(PLATFORM, TOKEN)

    proc.add(0, component)

    proc.min_detected_photons_filter(0)

    proc.with_input(input_state)

    sampler = pcvl.algorithm.Sampler(proc, max_shots_per_call=1000000)

    parameter_value = initial_parameter

    parameter.set_value(parameter_value)

    cost = []

    for k in range(max_iter):

        parameter.set_value(parameter_value + delta)
        prob_plus = probability_from_sampler(sampler, output_state, shots)

        parameter.set_value(parameter_value - delta)
        prob_minus = probability_from_sampler(sampler, output_state, shots)

        # Central finite difference
        grad = (prob_plus - prob_minus) / (2 * delta)

        # Gradient descent step
        parameter_value -= learning_rate * grad

        # Normalize angle
        parameter_value = parameter_value % (2 * np.pi)

        parameter.set_value(parameter_value)

        prob = probability_from_sampler(sampler, output_state, shots)

        cost.append(prob)

        print(
            f"Iteration {k:3d} | "
            f"P={prob:.6f} | "
            f"grad={grad:.6f} | "
            f"theta={parameter_value:.6f}"
        )

    return parameter_value, np.array(cost)
