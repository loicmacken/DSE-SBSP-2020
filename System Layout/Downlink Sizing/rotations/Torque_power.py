import numpy as np

# import MMOI of sting and relay
I_stingy = 0.25 * 15 * np.pi * 37.19 ** 3 * 26 # ellipse
I_relayy = 0.25 * 15 * np.pi * 231.19**3 * 26 # ellipse around short axis
I_relayx = 0.25 * 15 * np.pi * 231.19 * 26**3 # ellipse around long axis

#print(I_stingy / 10 ** 6, I_relayy / 10 ** 6)

# import important angles in degrees for readability
gamma = 49.13
rho = 95.52
beta = 48.13
alpha = 64.79
delta = alpha - beta

# define downtime
downtime_yr = 0.15 # days/yr
downtime_day = (downtime_yr/365) * 24 * 60 * 60 # sec/day

#print(downtime_day)

# omega_0
w0 = 360/(24*60*60)

# --------------------------
# -- STING --
# --------------------------
# define 'quick' turns
turn1 = abs(alpha/2 - -rho/2) # turn from -rho/2 to alpha/2
turn2 = abs(rho/2 - alpha/2) # turn from alpha/2 to rho/2
turn3 = abs(delta/2) # turn from beta/2 to alpha/2, is the same as a rotation of delta/2
turn4 = abs(-beta/2 - alpha/2) # turn from alpha/2 tot -beta/2

angles = np.array([turn1, turn2, turn3, turn4])
#print(angles)

# give each turn a turning time proportional to the required turn
total_turn = sum(angles)
times = 2/3 * downtime_day * (angles/total_turn)**(2/3)
print(times)
print(sum(times), downtime_day)

# angular accelerations, aim to make them the same
accs = angles * (2/times)**2
#print(accs)

# angular velocities
vels = accs * times/2
#print(vels)

# Torque & Power Sting
T = np.radians(accs) * I_stingy
P = np.radians(vels) * T
print("STING:\nMax Torque: ", round(max(T), 2), "Nm\nPower: ", round(np.average(P), 2), "Watts")

# ---------------------
# -- RELAY --
# ---------------------
# define 'quick' turns
turn5 = abs((180 - delta)/2 + (180 - rho + alpha)/2)
#print(turn5)

# omega required
time5 = (rho - beta)/360 * (24*60*60)
a = turn5 * (2/time5)**2
w = time5/2 * a
#print(a, w)

# Torque and power
T5 = I_relayy * np.radians(a)
P5 = np.radians(w) * T5
print("\nRELAY:\nTorque: ", round(T5, 2), "Nm\nPower: ", round(P5, 2), "Watts")
