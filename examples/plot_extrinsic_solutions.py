"""
Compare evaluation of compound distributions
"""

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")
import scipy.stats as st

# Load our modules
import source.recurrence as rec
import source.extrinsic as ext
import source.analytic as an

# Plot compound poisson distribution, comparing with analytic solution of the
# negative binomial to verify integration accuracy
def plot_leaky_telegraph_extrinsic():
    N = 80
    pVec = [100.0, 0.0001, 0, 20]  # [v12, v21, K1, K2]
    pStd = 10.0
    dNB = st.nbinom((pVec[3] / pStd) ** 2, 1 - pStd ** 2 / (pVec[3] + pStd ** 2))
    print("Solving compound two-state model using recurrence method")
    P = [dNB.pmf(n) for n in range(N)]
    Q = ext.solve_compound_rec(
        rec.recurrence_step_two_switch,
        pVec,
        pStd,
        N,
        400,
        compounding_distribution="gamma",
        decimal_precision=60,
    )
    fig = plt.figure()
    fig.suptitle("Compound leaky Telegraph model")
    plt.plot(range(N), P, label="Analytic solution")
    plt.plot(range(N), Q, "o", color="red", label="Recursion compound")
    plt.xlabel(r"Copy no., $n$")
    plt.ylabel(r"Probability distribution, $p(n)$")
    plt.legend()
    plt.show()
    # plt.savefig("Compound_Telegraph.pdf", format="pdf")


# Three state model with extrinsic noise on K3 as features in Figure 9 (b)
def plot_three_switch_extrinsic():
    prms = [0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 5.0, 20.0, 40.0]
    # [v12, v13, v21, v23, v31, v32, k1, k2 ,k3]
    kstd = 10.0
    N = 100
    print("Solving compound three-switch model using recurrence method")
    P = rec.recurrence_three_switch(prms, N, M=300)
    Q = ext.solve_compound_rec(
        rec.recurrence_step_three_switch,
        prms,
        kstd,
        N,
        recursion_length=391,
        index_compound_parameter=8,
        compounding_distribution="lognormal",
        decimal_precision=73,
    )

    plt.clf()
    fig, ax = plt.subplots()
    fig.suptitle("Compound three-state model")
    ax.plot(range(N), P, label=r"No noise", linestyle="--", color="black")
    ax.plot(
        range(N), Q, label=r"Recursion compound", linewidth=0, marker="o", markersize=2
    )
    ax.legend(loc="lower right")
    ax.set_xlabel(r"Copy no., $n$")
    ax.set_ylabel(r"Probability distribution, $p(n)$")

    # Inset axes showing the distribution tail
    axin = ax.inset_axes([0.6, 0.5, 0.39, 0.49])
    axin.semilogy(range(N), P, linestyle="--", color="black")
    axin.semilogy(range(N), Q, linewidth=0, marker="o", markersize=2)
    axin.set_xlim(prms[-1], N)
    axin.set_xlabel(r"$n$")
    axin.set_ylabel(r"$p(n)$")
    plt.show()
    # plt.savefig("Compound_ThreeState.pdf", format="pdf")


# 2^2 state model with extrinsic noise as features in Figure 9 (a)
def plot_twotwo_multistate_extrinsic():
    prms = [0.01, 0.01, 0.01, 0.01, 5, 16, 28]
    # [lamda0, mu0, lamda1, mu1, KB, k0, k1]
    kstd = 10.0
    N = 100
    print("Solving compound four-state model using analytic method")
    P = an.analytic_twotwo(prms, N)
    Q = ext.solve_compound(
        an.analytic_twotwo,
        prms,
        kstd,
        N,
        index_compound_parameter=6,
        compounding_distribution="lognormal",
    )

    plt.clf()
    fig, ax = plt.subplots()
    fig.suptitle(r"Compound $2^2$ state model")
    ax.plot(range(N), P, label=r"No noise", linestyle="--", color="black")
    ax.plot(range(N), Q, label=r"Compound")
    ax.legend(loc="lower right")
    ax.set_xlabel(r"Copy no., $n$")
    ax.set_ylabel(r"Probability distribution, $p(n)$")

    axin = ax.inset_axes([0.5, 0.4, 0.45, 0.55])
    axin.semilogy(range(N), P, linestyle="--", color="black")
    axin.semilogy(range(N), Q)
    axin.set_xlim(50, N)
    axin.set_xlabel(r"$n$")
    axin.set_ylabel(r"$p(n)$")
    plt.show()
    # plt.savefig("Compound_FourState.pdf", format="pdf")
