from turtle import color
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import scipy as sc

r = 0.31
mu = 0.4
sigma = 0.55
a4 = 0.3019
inputs = np.array([55, 30, 40, 35, 55])
# Novo experimento: 40, 35, 30, 55, 40

simulation_time = 12500
simulation_step = 1e-3

iterations = int(simulation_time / simulation_step)

time = np.linspace(.0, simulation_time, iterations)
time_span = [time[0], time[-1]]

# Derivation:
def dSdt(t, S):
    h3, h4 = S

    _t = int(t)
    
    if _t < 2500:
        u = inputs[0]
    elif _t > 2500 and _t < 5000:
        u = inputs[1]
    elif _t > 5000 and _t < 7500:
        u = inputs[2]
    elif _t > 7500 and _t < 10000:
        u = inputs[3]
    else:
        u = inputs[4]

    if h3 < 0:
        h3 = 0

    # Modelo experimental
    # qout = (185.48*np.sqrt(h3) - 167.01)*10e-5
    # qin = (16.46*u - 156.93)*10e-5
    # q34 = (32.4*(h4-h3) - 83.93)*10e-5

    # Modelo experimental ajustado
    qout = 0.93*(185.48*np.sqrt(h3) - 167.01)*10e-5
    qin = 1.04*(16.46*u - 156.93)*10e-5
    q34 = 0.95*(32.4*(h4-h3) - 83.93)*10e-5

    z1 = np.sqrt(h3)
    z2 = np.cos(2.5*np.pi * (h4 - mu)) / (sigma * np.sqrt(2 * np.pi))
    z3 = np.exp(-((h4 - mu)*2) / (2 * sigma*2))
    a3 = ((3*r)/5) * (2.7*r - (z2 * z3))

    h3_dot = (q34-qout)/a3
    h4_dot = (qin-q34)/a4

    return [h3_dot, h4_dot]

# Initial condition:
S0 = np.array([0, 0])

# Integration:
solution = solve_ivp(
    dSdt,
    time_span,
    S0,
)

# Simulated
t = solution.t
h3 = solution.y[0]
h4 = solution.y[1]

# Real
time = np.load('time.npy')
h3_r = np.load('level3.npy')[0]
h4_r = np.load('level4.npy')[0]

# Plot
fig, ax = plt.subplots(2, 1)
ax[0].plot(time, h3_r, label='real', color='y')
ax[0].plot(t, h3, label='simulated', color='b')
ax[0].set_title('h3 [cm]')
ax[0].legend(['real','simulated'])
ax[1].plot(time, h4_r, label='real', color='y')
ax[1].plot(t, h4, label='simulated', color='b')
ax[1].set_title('h4 [cm]')
ax[1].legend(['real','simulated'])
plt.savefig('simulation.png')
plt.show()