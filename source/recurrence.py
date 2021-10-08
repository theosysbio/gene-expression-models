"""
Functions for evaluating the recurrence relations
"""

import math
import numpy as np
from decimal import Decimal, getcontext
import scipy.special as sp
import mpmath as mpm

dfltPrec = 50  # Default precision for Decimal class


def recurrence_two_switch(prms, N, M, precision=dfltPrec):
    """Compute leaky gene distribution via the recurrence method.

    Arguments:
    prms -- List of the four rate parameters: v12,v21,K1,K2
    N -- Maximal mRNA copy number. The distribution is evaluated for n=0:N-1
    M -- Number of terms evaluated by the recursion relation
    precision -- Numerical precision used by the Decimal class"""

    G = recurrence_step_two_switch(prms, M, precision)
    return [invgenfunc(G, n, precision) for n in range(0, N)]


def recurrence_three_switch(prms, N, M, precision=dfltPrec):
    """Compute leaky gene distribution via the recurrence method.

    Arguments:
    parameters -- List of the nine rate parameters: v12,v13,v21,v23,v31,v32,k1,k2,k3
    N -- Maximal mRNA copy number. The distribution is evaluated for n=0:N-1
    M -- Number of terms evaluated by the recursion relation
    precision -- Numerical precision used by the Decimal class"""

    G = recurrence_step_three_switch(prms, M, precision)
    return [invgenfunc(G, n, precision) for n in range(0, N)]


def recurrence_feedback(prms, N, M, precision=dfltPrec):
    """Compute feedback model solution via the recurrence method.

    Arguments:
    parameters -- List of the five rate parameters: ru, rb, th, su, sb
    N -- Maximal mRNA copy number. The distribution is evaluated for n=0:N-1
    M -- Number of terms evaluated by the recursion relation
    precision -- Numerical precision used by the Decimal class"""

    G = recurrence_step_feedback(prms, M, precision)
    return [invgenfunc(G, n, precision) for n in range(0, N)]


def invgenfunc(G, n, precision=dfltPrec):
    """Back-calculate the distribution from the recurrence terms

    Arguments:
    G -- List of recurrence terms
    n -- Copy number at which to evaluate the distribution
    precision -- Numerical precision used by the Decimal class"""

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


def recurrence_step_two_switch(prms, M, precision=dfltPrec):
    """Compute leaky gene recurrence terms.

    Arguments:
    prms -- List of the four rate parameters: v12,v21,K1,K2
    M -- Number of terms evaluated by the recursion relation
    precision -- Numerical precision used by the Decimal class"""

    getcontext().prec = precision

    # Set up parameters
    lam = Decimal(prms[0])
    v = Decimal(prms[1])
    K_I = Decimal(prms[2])
    K_A = Decimal(prms[3])

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


def recurrence_step_three_switch(prms, M, precision=dfltPrec):
    """Compute leaky gene distribution via the recurrence method.

    Arguments:
    parameters -- List of the nine rate parameters: v12,v13,v21,v23,v31,v32,k1,k2,k3
    M -- Number of terms evaluated by the recursion relation
    precision -- Numerical precision used by the Decimal class"""

    getcontext().prec = precision

    def d(n):
        return (
            n ** Decimal(2)
            + n * (v12 + v13 + v21 + v23 + v31 + v32)
            + v12 * (v23 + v31 + v32)
            + v13 * (v21 + v23 + v32)
            + v21 * (v31 + v32)
            + v23 * v31
        )

    def update(nn, ss):
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
    v12 = Decimal(prms[0])
    v13 = Decimal(prms[1])
    v21 = Decimal(prms[2])
    v23 = Decimal(prms[3])
    v31 = Decimal(prms[4])
    v32 = Decimal(prms[5])
    k1 = Decimal(prms[6])
    k2 = Decimal(prms[7])
    k3 = Decimal(prms[8])

    vvec = np.array([[0, v12, v13], [v21, 0, v23], [v31, v32, 0]])
    kvec = [k1, k2, k3]

    ini = [
        (vvec[1, 0] * vvec[2, 0] + vvec[1, 2] * vvec[2, 0] + vvec[1, 0] * vvec[2, 1])
        / d(0),
        (vvec[0, 1] * vvec[2, 0] + vvec[0, 1] * vvec[2, 1] + vvec[0, 2] * vvec[2, 1])
        / d(0),
        (vvec[0, 2] * vvec[1, 0] + vvec[0, 1] * vvec[1, 2] + vvec[0, 2] * vvec[1, 2])
        / d(0),
    ]

    SS = [ini]

    for n in range(1, M):
        SS.append(update(n, SS[n - 1]))

    G = np.sum(SS, 1)

    return G


def recurrence_step_feedback(prms, M, precision=500):
    """Compute feedback model solution via the recurrence method.

    Arguments:
    parameters -- List of the five rate parameters: ru, rb, th, su, sb
    M -- Number of terms evaluated by the recursion relation
    precision -- Decimal precision used by the mpmath mpf type"""

    mpm.mp.dps = precision

    def update(nn, h0m2, h0m1, h1m2, h1m1):
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
    ru = mpm.mpf(prms[0])
    rb = mpm.mpf(prms[1])
    th = mpm.mpf(prms[2])
    su = mpm.mpf(prms[3])
    sb = mpm.mpf(prms[4])

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
