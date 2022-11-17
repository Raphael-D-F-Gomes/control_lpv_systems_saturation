import sympy as sym
import numpy as np
from non_linear_system import operation_points, state_space_non_linear_system, solid_area
from linearization import system_linearization


def lpv_system(h1_min, h1_max):
    # system variables
    h1 = sym.var('h1')
    h2 = sym.var('h2')
    u = sym.var('u')

    alpha_1, alpha2 = parameters_behavior(h1_min, h1_max)

    # operation points
    system_info = {'operation_points': [], 'non_linear_system': [],
                   'linear_system': [], 'parameters': (alpha_1, alpha2)}
    for h1_point in [h1_min, h1_max]:

        h2_point, u_point = operation_points(h1_point)
        system_info['operation_points'].append((h1_point, h2_point, u_point))

        # state space system
        h1_p, h2_p = state_space_non_linear_system()
        system_info['non_linear_system'].append((h1_p, h2_p))

        _, sys = system_linearization(h1_p, h2_p, h1_point, h2_point, h1, h2, u)
        system_info['linear_system'].append(sys)

    return system_info


def parameters_behavior(h1_min, h1_max):

    """

    alpha_1 + alpha_2 = 1
    alpha_1 = a1 * h1 + b1
    alpha_2 = a2 * h2 + b2
    """

    a1 = np.array([[h1_min, 1], [h1_max, 1]])
    b1 = np.array([1, 0])
    a2 = np.array([[h1_min, 1], [h1_max, 1]]);
    b2 = np.array([0, 1])

    x1 = np.linalg.solve(a1, b1)
    x2 = np.linalg.solve(a2, b2)

    return x1, x2
