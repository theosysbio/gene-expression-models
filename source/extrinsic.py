"""
General utility functions for models with extrinsic noise
"""

from copy import deepcopy
from decimal import Decimal

import numpy as np
import scipy.stats as st

from recurrence import invgenfunc


def solve_compound(
    sol_func,
    parameters: list,
    hyperparameter: float,
    N: int,
    parIdx: int = 3,
    distribution: str = "normal",
    recurrence: bool = False,
    precision: int = 50,
) -> list:
    """Obtain recursion coefficients h_i for compound distribution for model in solfunc.

    Args:
        solFunc: the solution function over which to compound
        parameters: list of parameters accepted by solFunc
        hyperparameter: standard deviation of the compounding distribution
        N: maximal mRNA copy number. The distribution is evaluated for n=0:N-1
        parIdx: index of the parameter over which the solution is compounded
        distribution: string specifying the type of compounding distribution
        recurrence: boolean specifying if compounding is over recurrence terms
        precision: integer specifying the precision used by the Decimal class

    Returns:
        recursion coefficients for mRNa copy numbers n=0:N-1.

    Raises:
        AssertionError: distribution given not supported
    """

    assert distribution in ["normal", "lognormal", "gamma"]

    # Specify some hyperparameters governing integration accuracy
    cdfMax = 0.999
    nTheta = 200

    # Set up the parameter distribution
    m, s = parameters[parIdx], hyperparameter
    if distribution == "normal":
        a, b = (0 - m) / s, 10000
        dist = st.truncnorm(a, b, m, s)
    elif distribution == "gamma":
        theta = s ** 2 / m
        k = (m / s) ** 2
        dist = st.gamma(k, scale=theta)
    elif distribution == "lognormal":
        mu = np.log(m / np.sqrt(1 + (s / m) ** 2))
        sg = np.sqrt(np.log(1 + (s / m) ** 2))
        dist = st.lognorm(s=sg, scale=np.exp(mu))
    else:
        print("Invalid distribution selected")
        return

    # Set up parameter vector
    thetMax = dist.ppf(cdfMax)
    thetMin = dist.ppf(1 - cdfMax)
    thetVec = np.linspace(thetMin, thetMax, nTheta)
    dThet = thetVec[1] - thetVec[0]
    P = np.zeros(N)
    parMod = deepcopy(parameters)

    # If operating on the recurrence terms, need to convert to Decimal
    if recurrence:
        P = np.array([Decimal(p) for p in P])
        dThet = Decimal(dThet)

    # Evaluate distribution for each theta and add contribution
    for thet in thetVec:
        parMod[parIdx] = thet
        if recurrence:
            P += np.array(sol_func(parMod, N, precision=precision)) * Decimal(
                dist.pdf(thet)
            )
        else:
            P += sol_func(parMod, N) * dist.pdf(thet)

    P *= dThet
    return P


def solve_compound_rec(
    recFunc,
    parameters: list,
    hyperparameter: float,
    N: int,
    M: int,
    parIdx: int = 3,
    distribution: str = "normal",
    precision: int = 100,
) -> list:
    """Compound distribution.

    Calls solve_compound() to obtain recurrence coefficients h_i and computes probability distribution using  invgenfunc()

    Arguments:
        recFunc: the recurrence relation function over which to compound
        parameters: list of parameters accepted by solFunc
        hyperparameter: standard deviation of the compounding distribution
        N: maximal mRNA copy number. The distribution is evaluated for n=0:N-1
        M: recursion length. The number of terms evaluated recursively
        parIdx: index of the parameter over which the solution is compunded
        distribution: string specifying the type of compounding distribution
        precision: integer specifying the precision used by the Decimal class

    Returns:
        probability distribution for mRNa copy numbers n=0:N-1.

    Raises:
        AssertionError: distribution given not supported
    """

    assert distribution in ["normal", "lognormal", "gamma"]

    H = solve_compound(
        recFunc,
        parameters,
        hyperparameter,
        M,
        parIdx,
        distribution,
        recurrence=True,
        precision=precision,
    )
    return [invgenfunc(H, n, precision=precision) for n in range(0, N)]
