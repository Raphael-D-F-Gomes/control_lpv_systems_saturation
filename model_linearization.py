import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
import control
from non_linear_system import operation_points, state_space_non_linear_system, solid_area
from linearization import system_linearization


# system variables
h1 = sym.var('h1')
h2 = sym.var('h2')
u = sym.var('u')

# operation points
h1_point = 0.28
h2_point, u_point = operation_points(h1_point)

# state space system
h1_p, h2_p = state_space_non_linear_system()

_, discrete_system = system_linearization(h1_p, h2_p, h1_point, h2_point, h1, h2, u)

n = 2



t0 = 0
t1 = T * 1000
nt = int(t1 / T) + 1
t = np.linspace(t0, t1, nt)
F = np.array([-0.04 if i < nt / 2 else 0.04 for i in range(nt)])

x1_0 = 0
x2_0 = 0
x0 = np.array([[x1_0], [x2_0]])

x = control.forced_response(continuos_sys, t, F, x0)
x1 = x.states[0, :] + h1_point
x2 = x.states[1, :] + h1_point

plt.close('all')
plt.figure(1, figsize=(12, 9))
plt.subplot(3, 1, 1)
plt.plot(t, x1, 'blue')
plt.grid()
plt.legend(labels=('x1 [ m ]',))
plt.subplot(3, 1, 2)
plt.plot(t, x2, 'green')
plt.grid()
plt.legend(labels=('x2 [ m / s ]',))
plt.subplot(3, 1, 3)
plt.plot(t, F, 'red')
plt.grid()
plt.legend(labels=('F[ N ]',))
plt.xlabel('t [ s ]')
plt.show()




