from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def solve_tridiagonal(
    lower: ArrayLike,
    diagonal: ArrayLike,
    upper: ArrayLike,
    right_hand_side: ArrayLike,
) -> NDArray[np.float64]:
    """
    Solve a tridiagonal linear system using the Thomas algorithm.

    For a system containing n unknowns:

    - lower has length n - 1
    - diagonal has length n
    - upper has length n - 1
    - right_hand_side has length n
    """

    lower_values = np.asarray(lower, dtype=np.float64)
    diagonal_values = np.asarray(diagonal, dtype=np.float64)
    upper_values = np.asarray(upper, dtype=np.float64)
    rhs_values = np.asarray(
        right_hand_side,
        dtype=np.float64,
    )

    size = diagonal_values.size

    if size < 1:
        raise ValueError(
            "Tridiagonal system must contain at least one unknown."
        )

    if lower_values.shape != (size - 1,):
        raise ValueError(
            "Lower diagonal must have length n - 1."
        )

    if upper_values.shape != (size - 1,):
        raise ValueError(
            "Upper diagonal must have length n - 1."
        )

    if rhs_values.shape != (size,):
        raise ValueError(
            "Right-hand side must have length n."
        )

    arrays = (
        lower_values,
        diagonal_values,
        upper_values,
        rhs_values,
    )

    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError(
            "Tridiagonal system must contain only finite values."
        )

    modified_diagonal = diagonal_values.copy()
    modified_rhs = rhs_values.copy()

    pivot_tolerance = np.finfo(np.float64).eps

    if abs(modified_diagonal[0]) <= pivot_tolerance:
        raise ValueError(
            "Tridiagonal system contains a zero leading pivot."
        )

    for index in range(1, size):
        multiplier = (
            lower_values[index - 1]
            / modified_diagonal[index - 1]
        )

        modified_diagonal[index] -= (
            multiplier
            * upper_values[index - 1]
        )

        modified_rhs[index] -= (
            multiplier
            * modified_rhs[index - 1]
        )

        if abs(modified_diagonal[index]) <= pivot_tolerance:
            raise ValueError(
                "Tridiagonal system is singular or ill-conditioned."
            )

    solution = np.empty(size, dtype=np.float64)
    solution[-1] = (
        modified_rhs[-1]
        / modified_diagonal[-1]
    )

    for index in range(size - 2, -1, -1):
        solution[index] = (
            modified_rhs[index]
            - upper_values[index]
            * solution[index + 1]
        ) / modified_diagonal[index]

    return solution