import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
import control
from non_linear_system import operation_points, state_space_non_linear_system, solid_area
from linearization import system_linearization, get_response_discrete_system


# system variables
h1 = sym.var('h1')
h2 = sym.var('h2')
u = sym.var('u')

h1_point = 0.2

op_points = operation_points({'h1': h1_point})

# state space system
h1_p, h2_p = state_space_non_linear_system()

T = 5.6
_, discrete_sys = system_linearization(h1_p, h2_p, op_points['h1'], op_points['h2'], h1, h2, u, T)


t0 = 0
t1 = T * 2000
nt = int(np.ceil(t1 / T))
t = np.linspace(t0, t1, nt)
delta_u = 0.2
F = np.array([delta_u if i < len(t) / 2 else -delta_u for i in range(len(t))])

h1_0 = 0
h2_0 = 0
h0 = np.array([[h2_0], [h1_0]])

h = get_response_discrete_system(discrete_sys, t, F, h0)

# x = control.forced_response(discrete_sys, t, F, h0)
h2 = h[0] + op_points['h2']
h1 = h[1] + op_points['h1']

op_points_u0 = operation_points({'u': op_points['u'] + delta_u})
op_points_u1 = operation_points({'u': op_points['u'] - delta_u})

ref_h1 = np.array([op_points_u0['h1'] if i < len(t) / 2 else op_points_u1['h1'] for i in range(len(t))])
ref_h2 = np.array([op_points_u0['h2'] if i < len(t) / 2 else op_points_u1['h2'] for i in range(len(t))])

plt.close('all')
plt.figure(1, figsize=(12, 9))
plt.plot(t, h1, 'blue')
plt.plot(t, ref_h1, 'red')
plt.grid()
plt.legend(labels=('h1 [ m ]', 'ref_h1'))
plt.xlabel('t [ s ]')

plt.figure(2, figsize=(12, 9))
plt.plot(t, h2, 'blue')
plt.plot(t, ref_h2, 'red')
plt.grid()
plt.legend(labels=('x2 [ m / s ]', 'ref_h2'))
plt.xlabel('t [ s ]')

plt.show()
