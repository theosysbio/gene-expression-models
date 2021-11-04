"""
Plot solutions computed with recurrence method for a few example systems with pre-defined parameters

Available functions:
    - plot_leaky_telegraph_recurrence()
    - plot_three_switch_recurrence()
    - plot_feedback_model_recurrence()
"""

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

# Load our modules
import source.fsp as fsp
import source.analytic as an
import source.recurrence as rec

# Plot recurrence approximation and analytic solution for Leaky Telegraph model
def plot_leaky_telegraph_recurrence():
    parameter_list = [0.1, 0.1, 50, 5] # [v12, v21, K1, K2]
    print("Solving Leaky Telegraph (two-state) model using recurrence method")
    p_recurrence = rec.recurrence_two_switch(parameter_list, 80, 260, 50)
    p_analytic = an.analytic_twostate(parameter_list, 80)
    fig = plt.figure()
    fig.suptitle("Leaky Telegraph Model")
    plt.plot(p_analytic, label="analytic")
    plt.plot(p_recurrence, "o", color="red", label="recurrence")
    plt.xlabel(r"Copy number, $n$")
    plt.ylabel(r"Probability distribution, $p(n)$")
    plt.legend()
    plt.show()
    # plt.savefig("Recurrence_TwoState.pdf", format="pdf")

# Plot recurrence and FSP approximation for three-switch model as featured in Figure 7 (b)
def plot_three_switch_recurrence():
    parameter_list = [
        0.045,
        0.045,
        0.035,
        0.015,
        0.015,
        0.035,
        5,
        30,
        60,
    ]  # [v12, v13, v21, v23, v31, v32, k1, k2 ,k3]
    print("Solving three-state model using recurrence method")
    p_recurrence = rec.recurrence_three_switch(parameter_list, 100, 305, 60)
    p_fsp = fsp.fsp_threestate(parameter_list, 100)
    plt.clf()
    fig = plt.figure()
    fig.suptitle("Three-Switch Model")
    plt.plot(p_fsp, label="FSP")
    plt.plot(p_recurrence, "o", color="red", label="recurrence")
    plt.xlabel(r"Copy number, $n$")
    plt.ylabel(r"Probability distribution, $p(n)$")
    plt.legend()
    plt.show()
    # plt.savefig("Recurrence_ThreeSwitch.pdf", format="pdf")

# Plot recurrence approximation and analytic solution for feedback model as featured in Figure 8 (b)
def plot_feedback_model_recurrence():
    parameter_list = [50, 5, 0, 0.5, 0.004]  # [ru, rb, th, su, sb]
    print("Solving feedback model using recurrence method")
    p_recurrence = rec.recurrence_feedback(parameter_list, 80, 300, 2000)
    p_analytic = an.analytic_feedback(parameter_list, 80)
    plt.clf()
    fig = plt.figure()
    fig.suptitle("Feedback Model")
    plt.plot(p_analytic, label="analytic")
    plt.plot(p_recurrence, "o", color="red", label="recurrence")
    plt.xlabel(r"Copy number, $n$")
    plt.ylabel(r"Probability distribution, $p(n)$")
    plt.legend()
    plt.show()
    # plt.savefig("Recurrence_Feedback.pdf", format="pdf")
