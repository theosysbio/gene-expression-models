"""
Functions for implementing analytic solutions for certain gene models

Available functions:
    - analytic_twostate
    - analytic_telegraph
    - analytic_twotwo
    - analytic_feedback
"""

import math
from typing import List

import mpmath as mpm
import numpy as np
import scipy.special as sp

import utility_functions as ut

mpm.mp.dps = 100  # Set precision for mpmath computations


def analytic_twostate(
    parameter_list: List[float], max_mRNA_copy_number: int
) -> List[float]:
    """Analytic steady state distribution for a two-state model (leaky telegraph).

    Requires computation at high precision via mpm for accurate convergence of
    the summation.

    Args:
        parameter_list: list of the four rate parameters: v12,v21,K0,K1
        max_mRNA_copy_number: maximal mRNA copy number. The distribution is evaluated for n=0:N-1

    Returns:
          probability distribution for mRNa copy numbers n=0:(max_mRNA_copy_number-1).
    """

    v12 = mpm.mpf(parameter_list[0])
    v21 = mpm.mpf(parameter_list[1])
    K0 = mpm.mpf(parameter_list[2])
    K1 = mpm.mpf(parameter_list[3])

    P = mpm.matrix(np.zeros(max_mRNA_copy_number))  # Preallocate at high precision
    for n in range(max_mRNA_copy_number):
        for r in range(n + 1):
            mpmCalc = (
                mpm.power(K1, n - r)
                * mpm.power(K0 - K1, r)
                * mpm.exp(-K0)
                * mpm.hyp1f1(v12, v12 + v21 + r, K0 - K1)
                / (mpm.factorial(n - r) * mpm.factorial(r))
            )
            P[n] += mpmCalc * ut.fracrise_mpm(v21, v21 + v12, r)

    P = np.array([float(p) for p in P])
    return P / P.sum()


def analytic_telegraph(
    parameter_list: List[float], max_mRNA_copy_number: int
) -> List[float]:
    """Analytic steady state distribution for the Telegraph model.

    Args:
        parameter_list: list of the three rate parameters: v12,v21,K
        max_mRNA_copy_number: maximal mRNA copy number.

    Returns
        probability distribution for mRNa copy numbers n=0:(max_mRNA_copy_number-1).
    """

    v12, v21, c = parameter_list
    prob_dist = np.zeros(max_mRNA_copy_number)

    a = v12
    b = v12 + v21

    for n in range(max_mRNA_copy_number):
        prob_dist[n] = c ** n * sp.hyp1f1(a + n, b + n, -c) / math.factorial(n)
        if np.isinf(prob_dist[n]):  # Use Stirling's approximation for n!
            prob_dist[n] = (
                sp.hyp1f1(a + n, b + n, -c)
                * (c * np.exp(1) / n) ** n
                / np.sqrt(2 * n * math.pi)
            )
        prob_dist[n] *= ut.fracrise(a, b, n - 1)

    return prob_dist / prob_dist.sum()


def analytic_twotwo(
    parameter_list: List[float], max_mRNA_copy_number: int
) -> List[float]:
    """Analytic solution to the 2^2 multistate model.

    Args:
        parameter_list: list of the seven rate parameters: lamda0,mu0,lamda1,mu1, KB,k0,k1
        max_mRNA_copy_number: maximal mRNA copy number.

    Returns
        probability distribution for mRNa copy numbers n=0:(max_mRNA_copy_number-1).
    """

    return np.array(
        [twotwo_n(parameter_list, n) for n in range(0, max_mRNA_copy_number)]
    )


def twotwo_n(parameter_list: List[float], mRNA_copy_number: int) -> float:
    """Analytic solution to the 2^2 multistate model for a single copy number n.

    Args:
        parameter_list: list of the seven rate parameters: lamda0,mu0,lamda1,mu1, KB,k0,k1
        mRNA_copy_number: copy number at which the distribution is evaluated.

    Returns:
        probability of mRNA copy number mRNA_copy_number
    """

    # Set up the parameters
    lamda0, mu0, lamda1, mu1, KB, k0, k1 = parameter_list
    n = mRNA_copy_number

    # Form list of all possible r_i combinations
    r_combinations = [(i, j, n - i - j) for i in range(n + 1) for j in range(n + 1 - i)]

    prob = 0
    for rc, r0, r1 in r_combinations:
        prob += (
            ut.multinomial_memo(rc, r0, r1)
            * KB ** rc
            * np.exp(-KB)
            * k0 ** r0
            * ut.fracrise_memo(lamda0, mu0, r0)
            * ut.hyp_memo(lamda0, mu0, k0, r0)
            * k1 ** r1
            * ut.fracrise_memo(lamda1, mu1, r1)
            * ut.hyp_memo(lamda1, mu1, k1, r1)
        )

    return prob / math.factorial(n)


def analytic_twothree(
    parameter_list: List[float], max_mRNA_copy_number: int
) -> List[float]:
    """Analytic solution to the 2^3 multistate model.

    Args:
        parameter_list: list of the ten rate parameters: lamda0,mu0,lamda1,mu1,lamda2,mu2, KB,k0,k1,k2.
        max_mRNA_copy_number: maximal mRNA copy number.

    Returns
        probability distribution for mRNa copy numbers n=0:(max_mRNA_copy_number-1).
    """

    return [twothree_n(parameter_list, n) for n in range(0, max_mRNA_copy_number)]


def twothree_n(parameter_list: List[float], mRNA_copy_number: int) -> float:
    """Analytic solution to the 2^3 multistate model for a single copy number n.

    Args:
        parameter_list: list of the ten rate parameters: lamda0,mu0,lamda1,mu1,lamda2,mu2, KB,k0,k1,k2
        mRNA_copy_number: copy number at which the distribution is evaluated.

    Returns:
        probability of mRNA copy number mRNA_copy_number
    """

    # Set up the parameters
    lamda0, mu0, lamda1, mu1, lamda2, mu2, KB, k0, k1, k2 = parameter_list
    n = mRNA_copy_number

    # Obtain list of all possible combinations of r_i
    r_combinations = [
        (i, j, k, n - i - j - k)
        for i in range(n + 1)
        for j in range(n + 1 - i)
        for k in range(n + 1 - i - j)
    ]

    prob = mpm.mpf(0)
    for rc, r0, r1, r2 in r_combinations:
        prob += (
            ut.multinomial_mpm((rc, r0, r1, r2))
            * ut.power_mpm_memo(KB, rc)
            * ut.exp_mpm_memo(-KB)
            * ut.power_mpm_memo(k0, r0)
            * ut.fracrise_mpm_memo(lamda0, mu0, r0)
            * ut.hyp_mpm_memo(lamda0, mu0, k0, r0)
            * ut.power_mpm_memo(k1, r1)
            * ut.fracrise_mpm_memo(lamda1, mu1, r1)
            * ut.hyp_mpm_memo(lamda1, mu1, k1, r1)
            * ut.power_mpm_memo(k2, r2)
            * ut.fracrise_mpm_memo(lamda2, mu2, r2)
            * ut.hyp_mpm_memo(lamda2, mu2, k2, r2)
        )

    return prob / mpm.factorial(n)


def analytic_feedback(
    parameter_list: List[float], max_mRNA_copy_number: int
) -> List[float]:
    """Analytic solution to the feedback model.

    Solution originally published in Grima et al, JCP 137, 035104 (2012)

    Args:
        parameter_list: list of the five rate parameter_list: ru, rb, th, su, sb
        max_mRNA_copy_number: maximal mRNA copy number.

    Returns
        probability distribution for mRNa copy numbers n=0:(max_mRNA_copy_number-1).
    """

    # Define some useful values
    ru, rb, th, su, sb = parameter_list
    Sb = 1 + sb
    R = ru - rb * Sb
    a = th + su * (ru - rb) / R
    b = 1 + th + (su + ru - ru / Sb) / Sb
    w0 = -R / (Sb ** 2)

    def P1(n):
        prob = 0.0
        for m in range(n + 1):
            prob += (
                sp.comb(n, m)
                * rb ** (n - m)
                * (R / Sb) ** m
                * ut.fracrise(a, b, m)
                * sp.hyp1f1(a + m, b + m, w0)
            )

        return prob / math.factorial(n)

    def P0(n):
        prob = 0.0
        for m in range(n):
            prob += (
                sp.comb(n - 1, m)
                * rb ** (n - 1 - m)
                * (R / Sb) ** m
                * ut.fracrise(a, b, m)
                * (
                    Sb * (m + a) * sp.hyp1f1(a + m + 1, b + m, w0) / (Sb - 1)
                    - (m + a + su * rb / R) * sp.hyp1f1(a + m, b + m, w0)
                )
            )

        return prob / math.factorial(n)

    # Evaluate P then normalise
    p00 = (
        Sb * a * sp.hyp1f1(a + 1, b, w0) / ru / (Sb - 1) - su * sp.hyp1f1(a, b, w0) / R
    )

    prob_dist = [P1(n) + P0(n) for n in range(1, max_mRNA_copy_number)]
    prob_dist.insert(0, p00 + P1(0))
    return prob_dist / sum(prob_dist)
