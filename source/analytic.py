"""
Functions for implementing analytic solutions for certain models
"""

import math

import mpmath as mpm
import numpy as np
import scipy.special as sp

mpm.mp.dps = 100  # Set precision for mpmath computations


# First a collection of utility functions used subsequently
def fracrise(x: float, y: float, r: int) -> float:
    """Compute ratio of rising factorials.

    Utility function for use within other analytic solutions.
    Computes the ratio of the rising factorials of x and y to order r
    q = x^{(r)}/y^{(r)}

    Args:
        x: combination of transition rates
        y: combination of transition rates
        r: order of rising factorial

    Returns
        Ratio of rising factorials as float
    """

    q = 1.0
    for m in range(r):
        q *= (x + m) / (y + m)

    return q


def fracrise_mpm(x: float, y: float, r: int) -> float:
    """High precision version of fracrise()"""
    q = mpm.mpf(1)
    for m in range(r):
        m = mpm.mpf(m)
        q *= (x + m) / (y + m)
    return q


def multinomial(integer_tuple: tuple) -> int:
    """Compute multinomial of tuple of parameters.

    Recursive implementation of multinomial of tuple of integers w.r.t their sum.

    Args:
        integer_tuple: tuple of integers

    Returns
        int
    """
    if len(integer_tuple) == 1:
        return 1
    return sp.binom(sum(integer_tuple), integer_tuple[-1]) * multinomial(
        integer_tuple[:-1]
    )


def multinomial_mpm(params: list) -> float:
    """High precision version of the multinomial function"""
    if len(params) == 1:
        return 1
    return mpm.binomial(sum(params), params[-1]) * multinomial_mpm(params[:-1])


# memoize functions that enable previous function calls to be stored in a dictionary.
# Provides improved performance when computationally expenesive evaluations are performed many times.
def memoize(f):
    """Creates a dictionary that stores all previous evaluations of a given function f

    Args:
        f: function

    Returns:
        dictionary of previous evaluations of f
    """
    memo = {}

    def helper(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]

    return helper


def memoize_nargs(f):
    """Same as memoize but for variable number of arguments"""
    memo = {}

    def helper(*nargs):
        if nargs not in memo:
            memo[nargs] = f(*nargs)
        return memo[nargs]

    return helper


@memoize_nargs
def multinomial_memo(rc, r0, r1):
    return multinomial((rc, r0, r1))


@memoize
def exp_mpm_memo(a):
    return mpm.exp(a)


@memoize_nargs
def power_mpm_memo(a, b):
    return mpm.power(a, b)


@memoize_nargs
def hyp_memo(lamda, mu, k, r):
    return sp.hyp1f1(lamda + r, mu + lamda + r, -k)


@memoize_nargs
def hyp_mpm_memo(lamda, mu, k, r):
    return mpm.hyp1f1(lamda + r, mu + lamda + r, -k)


@memoize_nargs
def fracrise_mpm_memo(lamda, mu, r):
    return fracrise_mpm(lamda, lamda + mu, r)


@memoize_nargs
def fracrise_memo(lamda, mu, r):
    return fracrise(lamda, lamda + mu, r)


def analytic_twostate(parameters: list, N: int) -> list:
    """Analytic steady state distribution for a two-state model (leaky telegraph).

    Requires computation at high precision via mpm for accurate convergence of
    the summation.

    Args:
        parameters: list of the four rate parameters: v12,v21,K0,K1
        N: maximal mRNA copy number. The distribution is evaluated for n=0:N-1

    Returns:
          probability distribution for mRNa copy numbers n=0:N-1.
    """

    v12 = mpm.mpf(parameters[0])
    v21 = mpm.mpf(parameters[1])
    K0 = mpm.mpf(parameters[2])
    K1 = mpm.mpf(parameters[3])

    P = mpm.matrix(np.zeros(N))  # Preallocate at high precision
    for n in range(N):
        for r in range(n + 1):
            mpmCalc = (
                mpm.power(K1, n - r)
                * mpm.power(K0 - K1, r)
                * mpm.exp(-K0)
                * mpm.hyp1f1(v12, v12 + v21 + r, K0 - K1)
                / (mpm.factorial(n - r) * mpm.factorial(r))
            )
            P[n] += mpmCalc * fracrise_mpm(v21, v21 + v12, r)

    P = np.array([float(p) for p in P])
    return P / P.sum()


def analytic_telegraph(parameters: list, N: int) -> list:
    """Analytic steady state distribution for the Telegraph model.

    Args:
        parameters: list of the three rate parameters: v12,v21,K
        N: maximal mRNA copy number.

    Returns
        probability distribution for mRNa copy numbers n=0:N-1.
    """

    v12, v21, K = parameters
    P = np.zeros(N)

    a = v12
    b = v12 + v21
    c = K

    for n in range(N):
        P[n] = c ** n * sp.hyp1f1(a + n, b + n, -c) / math.factorial(n)
        if np.isinf(P[n]):  # Use Stirling's approximation for n!
            P[n] = (
                sp.hyp1f1(a + n, b + n, -c)
                * (c * np.exp(1) / n) ** n
                / np.sqrt(2 * n * math.pi)
            )
        P[n] *= fracrise(a, b, n - 1)

    return P / P.sum()


def analytic_twotwo(parameters: list, N: int) -> list:
    """Analytic solution to the 2^2 multistate model.

    Args:
        parameters: list of the seven rate parameters: lamda0,mu0,lamda1,mu1, KB,k0,k1
        N: maximal mRNA copy number.

    Returns
        probability distribution for mRNa copy numbers n=0:N-1.
    """

    return np.array([twotwo_n(parameters, n) for n in range(0, N)])


def analytic_twothree(parameters: list, N: int) -> list:
    """Analytic solution to the 2^3 multistate model.

    Args:
        parameters: list of the ten rate parameters: lamda0,mu0,lamda1,mu1,lamda2,mu2, KB,k0,k1,k2.
        N: maximal mRNA copy number.

    Returns
        probability distribution for mRNa copy numbers n=0:N-1.
     """

    return [twothree_n(parameters, n) for n in range(0, N)]


def twotwo_n(parameters: list, n: int) -> list:
    """Analytic solution to the 2^2 multistate model for a single copy number n.

    Args:
        parameters: list of the seven rate parameters: lamda0,mu0,lamda1,mu1, KB,k0,k1
        n: copy number at which the distribution is evaluated.

    Returns:
        probability of mRNA copy number n
    """

    # Set up the parameters
    lamda0, mu0, lamda1, mu1, KB, k0, k1 = parameters
    k0A, k1A = KB, k0 + KB
    k0B, k1B = 0, k1

    # Form list of all possible r_i combinations
    r_combinations = [(i, j, n - i - j) for i in range(n + 1) for j in range(n + 1 - i)]

    p = 0
    for rc, r0, r1 in r_combinations:
        p += (
            multinomial_memo(rc, r0, r1)
            * KB ** rc
            * np.exp(-KB)
            * k0 ** r0
            * fracrise_memo(lamda0, mu0, r0)
            * hyp_memo(lamda0, mu0, k0, r0)
            * k1 ** r1
            * fracrise_memo(lamda1, mu1, r1)
            * hyp_memo(lamda1, mu1, k1, r1)
        )

    return p / math.factorial(n)


def twothree_n(parameters: list, n: int) -> list:
    """Analytic solution to the 2^3 multistate model for a single copy number n.

    Args:
        parameters: list of the ten rate parameters: lamda0,mu0,lamda1,mu1,lamda2,mu2, KB,k0,k1,k2
        n: copy number at which the distribution is evaluated.

    Returns:
        probability of mRNA copy number n
    """

    # Set up the parameters
    lamda0, mu0, lamda1, mu1, lamda2, mu2, KB, k0, k1, k2 = parameters
    k0A, k1A = mpm.mpf(KB), mpm.mpf(k0 + KB)
    k0B, k1B = mpm.mpf(0), mpm.mpf(k1)
    k0C, k1C = mpm.mpf(0), mpm.mpf(k2)

    # Obtain list of all possible combinations of r_i
    r_combinations = [
        (i, j, k, n - i - j - k)
        for i in range(n + 1)
        for j in range(n + 1 - i)
        for k in range(n + 1 - i - j)
    ]

    p = mpm.mpf(0)
    for rc, r0, r1, r2 in r_combinations:
        p += (
            multinomial_mpm(r_combinations)
            * power_mpm_memo(KB, rc)
            * exp_mpm_memo(-KB)
            * power_mpm_memo(k0, r0)
            * fracrise_mpm_memo(lamda0, mu0, r0)
            * hyp_mpm_memo(lamda0, mu0, k0, r0)
            * power_mpm_memo(k1, r1)
            * fracrise_mpm_memo(lamda1, mu1, r1)
            * hyp_mpm_memo(lamda1, mu1, k1, r1)
            * power_mpm_memo(k2, r2)
            * fracrise_mpm_memo(lamda2, mu2, r2)
            * hyp_mpm_memo(lamda2, mu2, k2, r2)
        )

    return p / mpm.factorial(n)


def analytic_feedback(parameters: list, N: int) -> list:
    """Analytic solution to the feedback model.

    Solution originally published in Grima et al, JCP 137, 035104 (2012)

    Args:
        parameters: list of the five rate parameters: ru, rb, th, su, sb
        N: maximal mRNA copy number.

    Returns
        probability distribution for mRNa copy numbers n=0:N-1.
    """

    # Define some useful values
    ru, rb, th, su, sb = parameters
    Sb = 1 + sb
    R = ru - rb * Sb
    a = th + su * (ru - rb) / R
    b = 1 + th + (su + ru - ru / Sb) / Sb
    w0 = -R / (Sb ** 2)

    def P1(n):
        p = 0.0
        for m in range(n + 1):
            p += (
                sp.comb(n, m)
                * rb ** (n - m)
                * (R / Sb) ** m
                * fracrise(a, b, m)
                * sp.hyp1f1(a + m, b + m, w0)
            )

        return p / math.factorial(n)

    def P0(n):
        p = 0.0
        for m in range(n):
            p += (
                sp.comb(n - 1, m)
                * rb ** (n - 1 - m)
                * (R / Sb) ** m
                * fracrise(a, b, m)
                * (
                    Sb * (m + a) * sp.hyp1f1(a + m + 1, b + m, w0) / (Sb - 1)
                    - (m + a + su * rb / R) * sp.hyp1f1(a + m, b + m, w0)
                )
            )

        return p / math.factorial(n)

    # Evaluate P then normalise
    p00 = (
        Sb * a * sp.hyp1f1(a + 1, b, w0) / ru / (Sb - 1) - su * sp.hyp1f1(a, b, w0) / R
    )

    P = [P1(n) + P0(n) for n in range(1, N)]
    P.insert(0, p00 + P1(0))
    return P / sum(P)
