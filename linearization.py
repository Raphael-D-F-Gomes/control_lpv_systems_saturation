import sympy as sym
import numpy as np
import control


def system_linearization(h1_p, h2_p, h1_point, h2_point, h1, h2, u, T):

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
    discrete_sys = continuos_sys.sample(T)

    return continuos_sys, discrete_sys


def get_response_discrete_system(sys, t, u, x0):

    x = np.array(x0)

    for i in range(1, len(t)):
        uk = u[i-1]
        x0 = [[x[0][-1]], [x[1][-1]]]
        x_aux = np.dot(sys.A, x0) + np.dot(sys.B, uk)
        x = np.append(x, x_aux, axis=1)

    return x
