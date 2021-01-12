import numpy as np
import matplotlib.pyplot as plt

optic = 1
alp = 0.85
eps = 0.7
I = 1000
C = 150
sig = 5.6703729*10**-8
A_spot = np.pi*25**2
eta = 0.3

def func(optic, alp, eps, I, C, sig, T0):
    T_opt = np.arange(273, (I * C / sig) ** 0.25, 1)
    eta = optic * (alp - eps * sig * T_opt ** 4 / (I * C)) * (1 - T0 / T_opt)
    optimal = np.where(eta == max(eta))
    optimal = T_opt[optimal]
    plt.plot(T_opt, eta)
    eta_max = max(eta)
    return eta_max, optimal

i = 0
while i < 10:
    P = 100*10**6/eta
    S = P/A_spot
    C = S/1000
    eta, T = func(optic, alp, eps, 103, C, sig, 323)
    i += 1

T = int(T)
print("Concentration: ", C//1, "Suns\t\tEfficiency: ", eta//0.001 / 10, "%\t\tTemperature: ", T, "K")

#plt.show()

"""
A single reservoir is a rectangular box with a square top face of area A_res (= s²) and a depth of h. The total surface area is thus
S_res = 2 * A + 4 * h * sqrt(A) = 2 * s² + 4 * h * s

P_out should equal P_in: P_net = P_out - P_in = 0

P_in is mostly our radiation beam: P = C * I * A_res
and also some ambient intake: P = S_res * sigma * epsilon * T_amb**4

P_out is modelled as blackbody radiation: P = S_res * sigma * epsilon * T_op**4
with T_op being the operational temperature obtained from 'func()'

so P_net = S_res * sigma * epsilon * T_op**4 - S_res * sigma * epsilon * T_amb**4 - C * I * A_res = 0
    <=> S_res * sigma * epsilon * (T_op**4 - T_amb**4) - C * I * A_res = 0
    <=> (2 * s² + 4 * h * s) * sigma * epsilon * (T_op**4 - T_amb**4) - s² * C * I = 0
    <=> s² * (2 * (1 + 2 * h/s) * sigma * epsilon * (T_op**4 - T_amb**4) - C * I) = 0

We try to find the ratio between s and h for which P_net is zero, rework:
    <=> (C * I) / (2 * sigma * epsilon * (T_op**4 - T_amb**4)) = 1 + 2 * h/s
    <=> h/s = ((C * I) / (2 * sigma * epsilon * (T_op**4 - T_amb**4)) - 1) / 2
"""

f = ((C * I) / (2 * sig * eps * (T**4 - 293**4)) - 1) / 2

print("h/s-ratio: ", f//0.1 / 10)

