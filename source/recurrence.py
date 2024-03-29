"""
Functions for evaluating the recurrence relations

Functions available for evaluatind recurrence relations for different models:
    - recurrence_two_switch
    - recurrence_three_switch
    - recurrence_feedback

Generate distribution from recurrence terms:
    - invgenfunc
"""

import math
from decimal import Decimal, getcontext
from typing import List

import mpmath as mpm
import numpy as np

dfltPrec = 50  # Default precision for Decimal class


def recurrence_two_switch(
    parameter_list: List[float], N: int, M: int, precision: int = dfltPrec
) -> List[float]:
    """Compute prodability distribution for two-state leaky gene model via recurrence method.

    First calculates recurrence terms using recurrence_step_two_switch() and subsequently the distribution using
    invgenfunc()

    Args:
        parameter_list: list of the four rate parameters: v12,v21,K1,K2
        N: maximal mRNA copy number. The distribution is evaluated for n=0:N-1
        M: number of terms evaluated by the recursion relation
        precision: numerical precision used by the Decimal class

    Returns:
        probability distribution for mRNa copy numbers n=0:N-1.
    """

    G = recurrence_step_two_switch(parameter_list, M, precision)
    return [invgenfunc(G, n, precision) for n in range(0, N)]


def recurrence_three_switch(
    parameter_list: List[float], N: int, M: int, precision: int = dfltPrec
) -> List[float]:
    """Compute prodability distribution for three-state leaky gene model via recurrence method.

    Args:
        parameter_list: list of the nine rate parameters: v12,v13,v21,v23,v31,v32,k1,k2,k3
        N: maximal mRNA copy number. The distribution is evaluated for n=0:N-1
        M: number of terms evaluated by the recursion relation
        precision: numerical precision used by the Decimal class

    Returns:
        probability distribution for mRNa copy numbers n=0:N-1.
    """

    G = recurrence_step_three_switch(parameter_list, M, precision)
    return [invgenfunc(G, n, precision) for n in range(0, N)]


def recurrence_feedback(
    parameter_list: List[float], N: int, M: int, precision: int = dfltPrec
):
    """Compute prodability distribution for feedback model via recurrence method.

    Args:
        parameter_list: list of the five rate parameters: ru, rb, th, su, sb
        N: maximal mRNA copy number. The distribution is evaluated for n=0:N-1
        M: number of terms evaluated by the recursion relation
        precision: numerical precision used by the Decimal class

    Returns:
        probability distribution for mRNa copy numbers n=0:N-1.
    """

    G = recurrence_step_feedback(parameter_list, M, precision)
    return [invgenfunc(G, n, precision) for n in range(0, N)]


def invgenfunc(G: list, n: int, precision: int = dfltPrec):
    """Back-calculate the distribution from the recurrence terms

    several functions such as recurrence_step_two_switch() compute the recurrence coefficiencts h_i. invgenfunc() uses these to compute the probability distribution.

    Args:
        G: list of recurrence terms
        n: copy number at which to evaluate the distribution
        precision: numerical precision used by the Decimal class

    Returns:
        probability of mRNA copy number n
    """

    getcontext().prec = precision
    M = len(G)
    s = Decimal(0)
    fac = Decimal(math.factorial(n))
    i = n + 0 - 1
    ds = fac * G[i + 1] * (-1) ** (0)
    s += ds

    for k in range(1, M - n):
        i = n + k - 1
        ds = G[i + 1] * (-1) ** (k)
        fac *= Decimal(k + n) / Decimal(k)
        ds *= fac
        s += ds

    return s / Decimal(math.factorial(n))


def recurrence_step_two_switch(parameter_list: list, M: int, precision: int = dfltPrec):
    """Compute recurrence terms for leaky gene model.

    Args:
        parameter_list: list of the four rate parameters: v12,v21,K1,K2
        M: number of terms evaluated by the recursion relation
        precision: numerical precision used by the Decimal class

    Returns:
         Recurrence terms up to order M
    """

    getcontext().prec = precision

    # Set up parameters
    lam = Decimal(parameter_list[0])
    v = Decimal(parameter_list[1])
    K_I = Decimal(parameter_list[2])
    K_A = Decimal(parameter_list[3])

    def gg0(n, x, y):  # calculate interates for g_0
        a = (((n + v) * (n + lam)) / v) - lam
        b = (n + v) / v
        return (1 / a) * ((b * K_I * x) + (K_A * y))

    def gg1(n, x, y):  # calculate interates for g_1
        c = (((n + v) * (n + lam)) / lam) - v
        d = (n + lam) / lam
        return (1 / c) * ((d * K_A * y) + (K_I * x))

    LL = [None] * M
    KK = [None] * M
    LL[0] = v / (lam + v)  # Initial condition for g_0
    KK[0] = lam / (lam + v)  # Initial condition for g_1

    for n in range(1, M):
        LL[n] = gg0(n, LL[n - 1], KK[n - 1])  # calculates iterates for g0
        KK[n] = gg1(n, LL[n - 1], KK[n - 1])  # calculates iterates for g1

    G = [a + b for a, b in zip(LL, KK)]

    return G


def recurrence_step_three_switch(
    parameter_list: list, M: int, precision: int = dfltPrec
) -> list:
    """Compute recurrence terms for three-state leaky gene model.

    Arguments:
        parameter_list: list of the nine rate parameters: v12,v13,v21,v23,v31,v32,k1,k2,k3
        M: number of terms evaluated by the recursion relation
        precision: numerical precision used by the Decimal class

    Returns:
        recurrence terms up to order M
    """

    getcontext().prec = precision

    def d(n: Decimal) -> Decimal:
        return (
            n ** Decimal(2)
            + n * (v12 + v13 + v21 + v23 + v31 + v32)
            + v12 * (v23 + v31 + v32)
            + v13 * (v21 + v23 + v32)
            + v21 * (v31 + v32)
            + v23 * v31
        )

    def update(nn: int, ss: list) -> list:
        n = Decimal(nn)
        up = [None] * 3
        for i in range(3):
            aa = list(range(3))
            aa.pop(i)
            j = aa[0]
            k = aa[1]

            up[i] = (
                kvec[j]
                * ss[j]
                * (
                    Decimal(n) * vvec[j, i]
                    + vvec[j, k] * vvec[k, i]
                    + vvec[j, i] * (vvec[k, i] + vvec[k, j])
                )
                + kvec[k]
                * ss[k]
                * (
                    Decimal(n) * vvec[k, i]
                    + vvec[j, k] * vvec[k, i]
                    + vvec[j, i] * (vvec[k, i] + vvec[k, j])
                )
                + kvec[i]
                * ss[i]
                * (
                    Decimal(n ** 2)
                    + vvec[j, i] * vvec[k, i]
                    + vvec[j, k] * vvec[k, i]
                    + vvec[j, i] * vvec[k, j]
                    + Decimal(n) * (vvec[j, i] + vvec[j, k] + vvec[k, i] + vvec[k, j])
                )
            )

            up[i] *= Decimal(1) / (n * d(n))
        return up

    # Set up parameters
    v12 = Decimal(parameter_list[0])
    v13 = Decimal(parameter_list[1])
    v21 = Decimal(parameter_list[2])
    v23 = Decimal(parameter_list[3])
    v31 = Decimal(parameter_list[4])
    v32 = Decimal(parameter_list[5])
    k1 = Decimal(parameter_list[6])
    k2 = Decimal(parameter_list[7])
    k3 = Decimal(parameter_list[8])

    vvec = np.array([[0, v12, v13], [v21, 0, v23], [v31, v32, 0]])
    kvec = [k1, k2, k3]

    ini = [
        (vvec[1, 0] * vvec[2, 0] + vvec[1, 2] * vvec[2, 0] + vvec[1, 0] * vvec[2, 1])
        / d(Decimal(0)),
        (vvec[0, 1] * vvec[2, 0] + vvec[0, 1] * vvec[2, 1] + vvec[0, 2] * vvec[2, 1])
        / d(Decimal(0)),
        (vvec[0, 2] * vvec[1, 0] + vvec[0, 1] * vvec[1, 2] + vvec[0, 2] * vvec[1, 2])
        / d(Decimal(0)),
    ]

    SS = [ini]

    for n in range(1, M):
        SS.append(update(n, SS[n - 1]))

    G = np.sum(SS, 1)

    return G


def recurrence_step_feedback(
    parameter_list: list, M: int, precision: int = 500
) -> list:
    """Compute recurrence terms for feedback model.

    Arguments:
        parameter_list: list of the five rate parameters: ru, rb, th, su, sb
        M: number of terms evaluated by the recursion relation
        precision: decimal precision used by the mpmath mpf type

    Returns:
        recurrence terms up to order M
    """

    mpm.mp.dps = precision

    def update(nn: int, h0m2: float, h0m1: float, h1m2: float, h1m1: float) -> list:
        n = mpm.mpf(nn)

        u1 = (
            mpm.mpf(1)
            / (sb * n)
            * (
                (th + su) * h1m1
                + su * h1m2
                - (sb + mpm.mpf(1)) * (n - mpm.mpf(1)) * h0m1
                + ru * h0m2
            )
        )
        u2 = (rb + su) / n * h1m1 - (sb + mpm.mpf(1)) * u1 + ru / n * h0m1
        return [u1, u2]

    # Set up parameters
    ru = mpm.mpf(parameter_list[0])
    rb = mpm.mpf(parameter_list[1])
    th = mpm.mpf(parameter_list[2])
    su = mpm.mpf(parameter_list[3])
    sb = mpm.mpf(parameter_list[4])

    Sb = mpm.mpf(1) + sb
    Rr = ru - rb * Sb
    Cc = (Sb - mpm.mpf(1)) / Sb
    al = th + su / Rr * (ru - rb)
    bet = mpm.mpf(1) + th + mpm.mpf(1) / Sb * (su + ru * Cc)
    w_1 = Rr * (Sb - mpm.mpf(1)) / (Sb ** mpm.mpf(2))

    Aa = mpm.mpf(1) / (
        Sb * al / (sb * ru) * mpm.hyp1f1((al + 1), (bet), (w_1))
        + (mpm.mpf(1) + (th - al) / (ru - rb)) * mpm.hyp1f1((al), (bet), (w_1))
    )
    h0Ini = Aa * (
        Sb * al / (sb * ru) * mpm.hyp1f1((al + 1), (bet), (w_1))
        + ((th - al) / (ru - rb)) * mpm.hyp1f1((al), (bet), (w_1))
    )
    h1Ini = Aa * mpm.hyp1f1((al), (bet), (w_1))
    h01Ini = mpm.mpf(1) / sb * (th + su) * h1Ini

    hh = [[h0Ini, h1Ini]]
    u2 = (rb + su) * h1Ini - (sb + mpm.mpf(1)) * h01Ini + ru * h0Ini
    hh.append([h01Ini, u2])

    for n in range(2, M):
        hh.append(update(n, hh[n - 2][0], hh[n - 1][0], hh[n - 2][1], hh[n - 1][1]))

    G = np.sum(hh, 1)

    return G
