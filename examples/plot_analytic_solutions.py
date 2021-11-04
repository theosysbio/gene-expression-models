"""
Plot analytic solutions of a few example systems for pre-defined parameters

Available functions:
    - plot_leaky_telegraph()
    - plot_2_2_multistate()
    - plot_2_3_multistate()
"""

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

import source.analytic as an

# Plot the leaky Telegraph model
def plot_leaky_telegraph():
    parameter_list = [0.1, 0.1, 50.0, 5.0]  # [v12, v21, K0, K1]
    print("Solving Leaky Telegraph (two-state) model using analytic method")
    p_analytic = an.analytic_twostate(parameter_list, 80)
    fig = plt.figure()
    fig.suptitle("Leaky Telegraph Model")
    plt.plot(p_analytic)
    plt.xlabel(r"Copy no., $n$")
    plt.ylabel(r"Probability distribution, $p(n)$")
    plt.show()
    # plt.savefig("Analytic_TwoState.pdf", format="pdf")


# Plot the 2^2-multistate model as featured in Figure 5(b)
def plot_twotwo_multistate():
    plt.clf()
    parameter_list = [0.01, 0.01, 0.01, 0.01, 5.0, 16.0, 28.0]
    # [lamda0, mu0, lamda1, mu1, KB, k0, k1]
    print("Solving four-state model using analytic method")
    p_analytic = an.analytic_twotwo(parameter_list, 100)
    fig = plt.figure()
    fig.suptitle(r"$2^2$-Multistate Model")
    plt.plot(p_analytic)
    plt.xlabel(r"Copy no., $n$")
    plt.ylabel(r"Probability distribution, $p(n)$")
    plt.show()
    # plt.savefig("Analytic_FourState.pdf", format="pdf")


# Plot the 2^3 state model as featured in Figure 6 (b)
# takes several hours to run
def plot_twothree_multistate():
    plt.clf()
    parameter_list = [
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        5.0,
        20.0,
        65.0,
        115.0,
    ]
    # [lamda0, mu0, lamda1, mu1, lamda2, mu2, KB, k0, k1, k2]
    print("Solving eight-state model using analytic method.")
    print("May take around 10 hours!")
    p_analytic = an.analytic_twothree(parameter_list, 250)
    fig = plt.figure()
    fig.suptitle(r"$2^3$-Multistate Model")
    plt.plot(p_analytic)
    plt.xlabel(r"Copy no., $n$")
    plt.ylabel(r"Probability distribution, $p(n)$")
    plt.show()
    # plt.savefig("Analytic_EightState.pdf", format='pdf')
