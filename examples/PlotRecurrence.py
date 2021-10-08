# Plot examples of the recurrence method solutions
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

# Plot recurrence approximation and analytic solution for leaky Telegraph model
prms = [0.1, 0.1, 50, 5]
print("Calculating two-state")
p_recurrence = rec.recurrence_two_switch(prms, 80, 260, 50)
p_analytic = an.analytic_twostate(prms, 80)
fig = plt.figure()
fig.suptitle("Leaky Telegraph Model")
plt.plot(p_analytic, label="analytic")
plt.plot(p_recurrence, "o", color="red", label="recurrence")
plt.xlabel(r"Copy number, $n$")
plt.ylabel(r"Probability distribution, $p(n)$")
plt.legend()
plt.savefig("Recurrence_TwoState.pdf", format="pdf")

# Plot recurrence and FSP approximation for three-switch model as featured in Figure 7 (b)
prms = [
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
print("Calculating three-state")
p_recurrence = rec.recurrence_three_switch(prms, 100, 305, 60)
p_fsp = fsp.FSP_threestate(prms, 100)
plt.clf()
fig.suptitle("Three-Switch Model")
plt.plot(p_fsp, label="FSP")
plt.plot(p_recurrence, "o", color="red", label="recurrence")
plt.xlabel(r"Copy number, $n$")
plt.ylabel(r"Probability distribution, $p(n)$")
plt.legend()
plt.savefig("Recurrence_ThreeSwitch.pdf", format="pdf")

# Plot recurrence approximation and analytic solution for feedback model as featured in Figure 8 (b)
prms = [50, 5, 0, 0.5, 0.004]  # [ru, rb, th, su, sb]
print("Calculating feedback model")
p_recurrence = rec.recurrence_feedback(prms, 80, 300, 2000)
p_analytic = an.analytic_feedback(prms, 80)
plt.clf()
fig.suptitle("Feedback Model")
plt.plot(p_analytic, label="analytic")
plt.plot(p_recurrence, "o", color="red", label="recurrence")
plt.xlabel(r"Copy number, $n$")
plt.ylabel(r"Probability distribution, $p(n)$")
plt.legend()
plt.savefig("Recurrence_Feedback.pdf", format="pdf")
