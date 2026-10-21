import numpy as np


def clements_indices(dim: int) -> list:
    """
    Generates the indices of the Clements decomposition for a given dimension.

    Args:
        dim (int): The dimension of the circuit.

    Returns:
        list: A list of indices for the Clements decomposition.
    """

    if not isinstance(dim, int):
        raise TypeError("dim must be an integer")
    if dim < 2:
        raise ValueError("dim must be at least 2")

    right_sequence = []
    left_sequence = []

    for k in range(1, dim):
        if k % 2 == 1:
            for i, j in zip(range(dim - 1, dim - k - 1, -1), range(k - 1, -1, -1)):
                right_sequence.append(j)
        else:
            for i, j in zip(range(dim - k, dim), range(0, k)):
                left_sequence.append(i - 1)

    return right_sequence + left_sequence[::-1]


def random_parameters(dim: int) -> list:
    """
    Generates a random Clements decomposition for a given dimension.

    Args:
        dim (int): The dimension of the circuit.

    Returns:
        list: A list of tuples representing the Clements decomposition.
    """

    indices = clements_indices(dim)

    decomposition = []
    for i in indices:
        phi = np.random.uniform(0, 2 * np.pi)
        theta = np.random.uniform(0, 2 * np.pi)
        decomposition.append((i, phi, theta))

    return decomposition


def target_value(decomposition: list, target: tuple) -> float:
    """
    Returns the value of the target parameter in the decomposition.

    Args:
        decomposition (list): A list of tuples representing the Clements decomposition.
        target (tuple): A tuple (index, parameter) specifying which parameter to return.

    Returns:
        float: The value of the target parameter.
    """
    if not isinstance(decomposition, (list, tuple)):
        raise TypeError("decomposition must be a list of tuples")

    idx, param = target

    _, phi, theta = decomposition[idx]

    return phi if param == 0 else theta
