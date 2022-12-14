import sympy as sym
import numpy as np
import control
from scipy.signal import cont2discrete, lti, dlti, dstep


def system_linearization(h1_p, h2_p, h1_point, h2_point, h1, h2, u, dt):

    A = np.array([[h2_p, h2_p], [h1_p, h1_p]])
    B = np.array([[h2_p], [h1_p]])

    A[0][0] = sym.diff(A[0][0], h2).subs({h1: h1_point, h2: h2_point})
    A[0][1] = sym.diff(A[0][1], h1).subs({h1: h1_point, h2: h2_point})
    A[1][0] = sym.diff(A[1][0], h2).subs({h1: h1_point, h2: h2_point})
    A[1][1] = sym.diff(A[1][1], h1).subs({h1: h1_point, h2: h2_point})

    B[0][0] = sym.diff(B[0][0], u).subs({h1: h1_point, h2: h2_point})
    B[1][0] = sym.diff(B[1][0], u).subs({h1: h1_point, h2: h2_point})

    C = np.array([[0, 1]])
    D = np.array([[0]])

    continuos_sys = control.StateSpace(A, B, C, D)
    # discrete_sys = continuos_sys.sample(T)
    methods = ['zoh', 'bilinear', 'euler', 'backward_diff', 'foh', 'impulse']
    discrete_sys = cont2discrete((A, B, C, D), dt, method=methods[4])
    A = discrete_sys[0]
    B = discrete_sys[1]
    C = discrete_sys[2]
    D = discrete_sys[3]

    return A, B, C, D


def get_response_discrete_system(A, B, t, u, x0):

    x = np.array(x0)

    for i in range(1, len(t)):
        uk = u[i-1]
        x0 = [[x[0][-1]], [x[1][-1]]]
        x_aux = np.dot(A, x0) + np.dot(B, uk)
        x = np.append(x, x_aux, axis=1)

    return x
