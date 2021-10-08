# Compare evaluation of compound distributions
import scipy.stats as st
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../source"))
)

# Load our modules
import FSP as fsp
import analytic as an
import recurrence as rec
import extrinsic as ext

# Plot compound poisson distribution, comparing with analytic solution of the
# negative binomial to verify integration accuracy
N = 80
pVec = [100.0, 0.0001, 0, 20]
pStd = 10.0
dNB = st.nbinom((pVec[3] / pStd) ** 2, 1 - pStd ** 2 / (pVec[3] + pStd ** 2))
print("Calculating compound two-state")
P = [dNB.pmf(n) for n in range(N)]
Q = ext.solve_compound_rec(
    rec.recurrence_step_two_switch,
    pVec,
    pStd,
    N,
    400,
    distribution="gamma",
    precision=60,
)
fig = plt.figure()
fig.suptitle("Compound leaky Telegraph model")
plt.plot(range(N), P, label="Analytic solution")
plt.plot(range(N), Q, "o", color="red", label="Recursion compound")
plt.xlabel(r"Copy no., $n$")
plt.ylabel(r"Probability distribution, $p(n)$")
plt.legend()
plt.savefig("Compound_Telegraph.pdf", format="pdf")

# Three state model with extrinsic noise on K3 as features in Figure 9 (b)
prms = [0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 5.0, 20.0, 40.0]
kstd = 10.0
N = 100
print("Calculating compound three-state")
P = rec.recurrence_three_switch(prms, N, M=300)
Q = ext.solve_compound_rec(
    rec.recurrence_step_three_switch,
    prms,
    kstd,
    N,
    M=391,
    parIdx=8,
    distribution="lognormal",
    precision=73,
)

plt.clf()
fig, ax = plt.subplots()
fig.suptitle("Compound three-state model")
ax.plot(range(N), P, label=r"No noise", linestyle="--", color="black")
ax.plot(range(N), Q, label=r"Recursion compound", linewidth=0, marker="o", markersize=2)
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
plt.savefig("Compound_ThreeState.pdf", format="pdf")

# 2^2 state model with extrinsic noise as features in Figure 9 (a)
prms = [0.01, 0.01, 0.01, 0.01, 5, 16, 28]
kstd = 10.0
N = 100
print("Calculating compound four-state")
P = an.analytic_twotwo(prms, N)
Q = ext.solve_compound(
    an.analytic_twotwo, prms, kstd, N, parIdx=6, distribution="lognormal"
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
plt.savefig("Compound_FourState.pdf", format="pdf")
