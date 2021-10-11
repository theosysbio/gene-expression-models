"""
Implement finite state projection algorithm
"""

from time import perf_counter as pc

import numpy as np
import scipy.sparse as sps
from scipy.linalg import null_space


def fsp_twostate(parameters: list, N: int) -> list:
    """Steady-state distribution for a two-state model evaluated using the FSP.

    Args:
        parameters: list of the four rate parameters: v12,v21,k1,k2
        N: maximal mRNA copy number.

    Returns:
        probability distribution for mRNA copy numbers for n=0:N-1.
    """

    t0 = pc()
    v12, v21, k1, k2 = parameters
    A = np.array([[-v12, v21], [v12, -v21]])
    T = np.diag([k1, k2])
    D = np.eye(2)
    AG = sps.lil_matrix((2 * N, 2 * N), dtype=np.float64)

    for i in range(1, N + 1):
        if i < N:
            AG[(i - 1) * 2 : i * 2, (i - 1) * 2 : i * 2] = (
                AG[(i - 1) * 2 : i * 2, (i - 1) * 2 : i * 2] + A - T - (i - 1) * D
            )
            AG[i * 2 : (i + 1) * 2, (i - 1) * 2 : i * 2] = T
        else:
            AG[(i - 1) * 2 : i * 2, (i - 1) * 2 : i * 2] = (
                AG[(i - 1) * 2 : i * 2, (i - 1) * 2 : i * 2] + A - (i - 1) * D
            )
        if i - 1 > 0:
            AG[(i - 2) * 2 : (i - 1) * 2, (i - 1) * 2 : i * 2] = (i - 1) * D

    matrix_time = pc() - t0
    t1 = pc()

    P = null_space(AG.toarray())
    null_time = pc() - t1
    t2 = pc()

    P = np.squeeze(P)
    P = P / P.sum()
    L = 2 * N + 1
    P = P[0:L:2] + P[1:L:2]

    return P  # , matrix_time, null_time


def fsp_threestate(parameters: list, N: int) -> list:
    """Steady state distribution for a three-state model evaluated using the FSP.

    Args:
        parameters: list of the nine rate parameters: v12,v13,v21,v23,v31,v32,k1,k2,k3
        N: maximal mRNA copy number.

    Returns:
        probability distribution for mRNA copy numbers for n=0:N-1.
    """

    v12, v13, v21, v23, v31, v32, k1, k2, k3 = parameters
    A = np.array(
        [[-v12 - v13, v21, v31], [v12, -v21 - v23, v32], [v13, v23, -v31 - v32]]
    )
    T = np.diag([k1, k2, k3])
    D = np.eye(3)
    AG = sps.lil_matrix((3 * N, 3 * N), dtype=np.float64)

    for i in range(1, N + 1):
        if i < N:
            AG[(i - 1) * 3 : i * 3, (i - 1) * 3 : i * 3] = (
                AG[(i - 1) * 3 : i * 3, (i - 1) * 3 : i * 3] + A - T - (i - 1) * D
            )
            AG[i * 3 : (i + 1) * 3, (i - 1) * 3 : i * 3] = T
        else:
            AG[(i - 1) * 3 : i * 3, (i - 1) * 3 : i * 3] = (
                AG[(i - 1) * 3 : i * 3, (i - 1) * 3 : i * 3] + A - (i - 1) * D
            )
        if i - 1 > 0:
            AG[(i - 2) * 3 : (i - 1) * 3, (i - 1) * 3 : i * 3] = (i - 1) * D

    P = null_space(AG.toarray())
    P = np.squeeze(P)
    P = P / P.sum()
    L = 3 * N + 1
    P = P[0:L:3] + P[1:L:3] + P[2:L:3]
    return P
