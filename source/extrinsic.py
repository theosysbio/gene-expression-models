# General utility functions used elsewhere

import scipy.stats as st
import numpy as np
from copy import deepcopy
from recurrence import invgenfunc
from decimal import Decimal, getcontext


def solve_compound(
    solFunc,
    parameters,
    hyperparameter,
    N,
    parIdx=3,
    distribution="normal",
    recurrence=False,
    precision=50,
):
    """Obtain a compound distribution for the model in solfunc.

    Arguments:
    solFunc -- The solution function over which to compound
    parameters -- List of parameters accepted by solFunc
    hyperparameter -- Standard deviation of the compounding distribution
    N -- Maximal mRNA copy number. The distribution is evaluated for n=0:N-1

    Keyword arguments:
    parIdx -- Index of the parameter over which the solution is compunded
    distribution -- String specifying the type of compounding distribution
    recurrence -- Boolean specifying if compounding is over recurrence terms
    precision -- Integer specifying the precision used by the Decimal class
    """

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

    # If operating on the recurrence terms, need to comnvert to Decimal
    if recurrence:
        P = np.array([Decimal(p) for p in P])
        dThet = Decimal(dThet)

    # Evaluate distribution for each theta and add contribution
    for thet in thetVec:
        parMod[parIdx] = thet
        if recurrence:
            P += np.array(solFunc(parMod, N, precision=precision)) * Decimal(
                dist.pdf(thet)
            )
        else:
            P += solFunc(parMod, N) * dist.pdf(thet)

    P *= dThet
    return P


def solve_compound_rec(
    recFunc,
    parameters,
    hyperparameter,
    N,
    M,
    parIdx=3,
    distribution="normal",
    precision=100,
):
    """Obtain the coefficients h_i of the recurrence method for a compound
    distribution.

    Acts as a wrapper for solve_compound as the underlying operation is
    the same. Subsequently obtains the probability mass function.

    Arguments:
    recFunc -- The recurrence relation function over which to compound
    parameters -- List of parameters accepted by solFunc
    hyperparameter -- Standard deviation of the compounding distribution
    N -- Maximal mRNA copy number. The distribution is evaluated for n=0:N-1
    M -- Recursion length. The number of terms evaluated recursively

    Keyword arguments:
    parIdx -- Index of the parameter over which the solution is compunded
    distribution -- String specifying the type of compounding distribution
    precision -- Integer specifying the precision used by the Decimal class
    """

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
