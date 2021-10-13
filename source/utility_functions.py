"""
Memoized utility functions used in analytic.py

Implemented functions
    - fracrise: ratio of rising factorials
    - fracrise_mpm: high-precision version of fracrise
    - multinomial
    - multinomial_mpm: high-precision version of multinomial
"""

import mpmath as mpm
import scipy.special as sp

mpm.mp.dps = 100  # Set precision for mpmath computations


# First a collection of utility functions used subsequently
def fracrise(x: float, y: float, order: int) -> float:
    """Compute ratio of rising factorials.

    Utility function for use within other analytic solutions.
    Computes the ratio of the rising factorials of x and y to order r
    q = x^{(r)}/y^{(r)}

    Args:
        x: combination of transition rates
        y: combination of transition rates
        order: order of rising factorial

    Returns
        Ratio of rising factorials as float
    """

    q = 1.0
    for m in range(order):
        q *= (x + m) / (y + m)

    return q


def fracrise_mpm(x: float, y: float, order: int) -> float:
    """High precision version of fracrise()"""
    q = mpm.mpf(1)
    for m in range(order):
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


def multinomial_mpm(integer_combinations: tuple) -> float:
    """High precision version of the multinomial function"""
    if len(integer_combinations) == 1:
        return 1
    return mpm.binomial(
        sum(integer_combinations), integer_combinations[-1]
    ) * multinomial_mpm(integer_combinations[:-1])


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


def multinomial_mpm_memo(rc, r0, r1, r2):
    return multinomial_mpm((rc, r0, r1, r2))


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
