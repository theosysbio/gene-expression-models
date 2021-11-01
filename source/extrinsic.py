"""
Functions for generating recurrence solutions for models with extrinsic noise

Function used to compute recurrence solution: solve_compound().
"""

from copy import deepcopy
from decimal import Decimal
from typing import Callable, List

import numpy as np
import scipy.stats as st

from recurrence import invgenfunc


def solve_compound_rec(
    recurrence_func: Callable,
    parameter_list: List[float],
    std_of_compound_dist: float,
    max_mRNA_copy_number: int,
    recursion_length: int,
    index_compound_parameter: int = 3,
    compounding_distribution: str = "normal",
    decimal_precision: int = 100,
) -> List[float]:
    """Compound distribution.

    Calls solve_compound() to obtain recurrence coefficients h_i and computes probability distribution using  invgenfunc()

    Arguments:
        recurrence_func: the recurrence relation function over which to compound
        parameter_list: list of parameters accepted by solFunc
        std_of_compound_dist: standard deviation of the compounding distribution
        max_mRNA_copy_number: maximal mRNA copy number. The distribution is evaluated for n=0:N-1
        recursion_length: recursion length. The number of terms evaluated recursively
        index_compound_parameter: index of the parameter over which the solution is compunded
        compounding_distribution: string specifying the type of compounding distribution
        decimal_precision: integer specifying the precision used by the Decimal class

    Returns:
        probability distribution for mRNa copy numbers n=0:N-1.

    Raises:
        AssertionError: distribution given not supported
    """

    assert compounding_distribution in ["normal", "lognormal", "gamma"]

    H = solve_compound(
        recurrence_func,
        parameter_list,
        std_of_compound_dist,
        recursion_length,
        index_compound_parameter,
        compounding_distribution,
        compound_over_recurrence_terms=True,
        decimal_precision=decimal_precision,
    )
    return [
        invgenfunc(H, n, precision=decimal_precision)
        for n in range(0, max_mRNA_copy_number)
    ]


def solve_compound(
    sol_func: Callable,
    parameter_list: List[float],
    std_of_compound_dist: float,
    max_mRNA_copy_number: int,
    index_compound_parameter: int = 3,
    compounding_distribution: str = "normal",
    compound_over_recurrence_terms: bool = False,
    decimal_precision: int = 50,
) -> List[float]:
    """Obtain recursion coefficients h_i for compound distribution for model in sol_func.

    Args:
        sol_func: the solution function over which to compound
        parameter_list: list of parameters accepted by solFunc
        std_of_compound_dist: standard deviation of the compounding distribution
        max_mRNA_copy_number: maximal mRNA copy number. The distribution is evaluated for n=0:N-1
        index_compound_parameter: index of the parameter over which the solution is compounded
        compounding_distribution: string specifying the type of compounding distribution
        compound_over_recurrence_terms: boolean specifying if compounding is over recurrence terms
        decimal_precision: integer specifying the precision used by the Decimal class

    Returns:
        recursion coefficients for mRNa copy numbers n=0:N-1.

    Raises:
        AssertionError: distribution given not supported
    """

    assert compounding_distribution in ["normal", "lognormal", "gamma"]

    # Specify some hyperparameters governing integration accuracy
    cdfMax = 0.999
    nTheta = 200

    # Set up the parameter distribution

    dist = set_up_parameter_distribution(
        parameter_list,
        std_of_compound_dist,
        index_compound_parameter,
        compounding_distribution,
    )

    # Set up parameter vector
    thetMax = dist.ppf(cdfMax)
    thetMin = dist.ppf(1 - cdfMax)
    thetVec = np.linspace(thetMin, thetMax, nTheta)
    dThet = thetVec[1] - thetVec[0]
    P = np.zeros(max_mRNA_copy_number)
    parMod = deepcopy(parameter_list)

    # If operating on the recurrence terms, need to convert to Decimal
    if compound_over_recurrence_terms:
        P = np.array([Decimal(p) for p in P])
        dThet = Decimal(dThet)

    # Evaluate distribution for each theta and add contribution
    for thet in thetVec:
        parMod[index_compound_parameter] = thet
        if compound_over_recurrence_terms:
            P += np.array(
                sol_func(parMod, max_mRNA_copy_number, precision=decimal_precision)
            ) * Decimal(dist.pdf(thet))
        else:
            P += sol_func(parMod, max_mRNA_copy_number) * dist.pdf(thet)

    P *= dThet
    return P


def set_up_parameter_distribution(
    parameter_list: List[float],
    std_of_compound_dist: float,
    index_compound_parameter: int = 3,
    compounding_distribution: str = "normal",
) -> Callable:
    """Set up parameter distribution used needed in solve_compound()

    Args:
        parameter_list: list of parameters accepted by solFunc
        std_of_compound_dist: standard deviation of the compounding distribution
        index_compound_parameter: index of the parameter over which the solution is compounded
        compounding_distribution: string specifying the type of compounding distribution

    Returns:
        distribution dist

    """
    m, s = parameter_list[index_compound_parameter], std_of_compound_dist
    dist = None
    if compounding_distribution == "normal":
        a, b = (0 - m) / s, 10000
        dist = st.truncnorm(a, b, m, s)
    elif compounding_distribution == "gamma":
        theta = s ** 2 / m
        k = (m / s) ** 2
        dist = st.gamma(k, scale=theta)
    elif compounding_distribution == "lognormal":
        mu = np.log(m / np.sqrt(1 + (s / m) ** 2))
        sg = np.sqrt(np.log(1 + (s / m) ** 2))
        dist = st.lognorm(s=sg, scale=np.exp(mu))
    return dist
