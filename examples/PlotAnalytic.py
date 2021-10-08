# Plot analytic solutions
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../source"))
)
import analytic as an

# Plot the leaky Telegraph model
prms = [0.1, 0.1, 50.0, 5.0]
print("Calculating two-state")
p_analytic = an.analytic_twostate(prms, 80)
fig = plt.figure()
fig.suptitle("Leaky Telegraph Model")
plt.plot(p_analytic)
plt.xlabel(r"Copy no., $n$")
plt.ylabel(r"Probability distribution, $p(n)$")
plt.savefig("Analytic_TwoState.pdf", format="pdf")

# Plot the 2^2-multistate model as featured in Figure 5(b)
plt.clf()
prms = [0.01, 0.01, 0.01, 0.01, 5.0, 16.0, 28.0]
print("Calculating four-state")
p_analytic = an.analytic_twotwo(prms, 100)
fig = plt.figure()
fig.suptitle(r"$2^2$-Multistate Model")
plt.plot(p_analytic)
plt.xlabel(r"Copy no., $n$")
plt.ylabel(r"Probability distribution, $p(n)$")
plt.savefig("Analytic_FourState.pdf", format="pdf")


# Uncomment this to solve example for 2^3 state model - takes several hours to run
# # Plot the 2^3 state model as featured in Figure 6 (b)
# plt.clf()
# prms = [0.01,0.01,0.01,0.01,0.01,0.01, 5.,20.,65.,115.]
# print('Calculating eight-state. May take around 10 hours.')
# p_analytic = an.analytic_twothree(prms, 250)
# fig = plt.figure()
# fig.suptitle(r'$2^3$-Multistate Model')
# plt.plot(p_analytic)
# plt.xlabel(r'Copy no., $n$')
# plt.ylabel(r'Probability distribution, $p(n)$')
# plt.savefig("Analytic_EightState.pdf", format='pdf')
# plt.show()
