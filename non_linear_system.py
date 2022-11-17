import sympy as sym
from sympy import Poly, roots
import numpy as np
from sympy.solvers import solve


def operation_points(op_point: dict):
    """
    This method calculates the operation points of the non-linear systems. Considering the system in
    balance, the mass flow is equal at the input, between tanks and output
    Args:
        op_point (dict): operation point ate tank 1.
    Returns:
        (float): operation point ate tank 2.
        (float): input.
    """
    # system variables
    h1 = sym.var('h1')
    h2 = sym.var('h2')
    u = sym.var('u')

    # system non-linear equations
    qin, q12, qout = system_non_linear_equations(h1, h2, u)
    op_points = {}
    if list(op_point)[0] == 'h1':
        q12 = q12.subs({h1: op_point['h1']})
        qout = qout.subs({h1: op_point['h1']})

        op_points['u'] = solve(qin - qout)[0]
        op_points['h2'] = solve(q12 - qout)[0]
        op_points['h1'] = op_point['h1']

    elif list(op_point)[0] == 'h2':
        q12 = q12.subs({h2: op_point['h2']})

        op_points['u'] = solve(qin - q12)[0]
        op_points['h1'] = solve(q12 - qout)[0]
        op_points['h2'] = op_point['h2']

    if list(op_point)[0] == 'u':
        qin = qin.subs({u: op_point['u']})
        op_points['h1'] = solve(qout - qin)[0]
        q12 = q12.subs({h1: op_points['h1']})

        op_points['u'] = op_point['u']
        op_points['h2'] = solve(q12 - qin)[0]

    return op_points


def state_space_non_linear_system():
    """
    This method calculates the solid surface area given h1 point.

    Args:
        h1_point (float): operation point ate tank 1.
    Returns:
        (float): area of the solid
    """

    # system variables
    h1 = sym.var('h1')
    h2 = sym.var('h2')
    u = sym.var('u')

    # system non-linear equations
    qin, q12, qout = system_non_linear_equations(h1, h2, u)

    # solid surface parameters
    mu = 0.4
    sigma = 0.55
    r = 0.31
    z = 2.5 * np.pi * (h1 - mu)
    w = -(h1 - mu) ** 2 / (2 * sigma ** 2)

    # z and w decomposed with taylor series approach
    z_part = 1 - z ** 2 / 2 + z ** 4 / 24 - z ** 6 / 720 + z ** 8 / 40320
    w_part = 1 + w + w ** 2 / 2 - w ** 3 / 6 + w ** 4 / 24

    # solid surface area decomposed
    solid_area = (3 * r / 5) * (2.7 * r - (1 / (sigma * np.sqrt(2 * np.pi))) * z_part * w_part)

    # state space non-linear system
    h1_p = (q12 - qout) / solid_area
    h2_p = (qin - q12) / 0.3019

    return h1_p, h2_p


def solid_area(h1_point):
    mu = 0.4
    sigma = 0.55
    r = 0.31
    z = 2.5 * np.pi * (h1_point - mu)
    w = -(h1_point - mu) ** 2 / (2 * sigma ** 2)
    area = (3 * r / 5) * (2.7 * r - (np.cos(float(z)) / (sigma * np.sqrt(2 * np.pi))) * np.exp(float(w)))

    return area


def system_non_linear_equations(h1, h2, u):
    """

    Returns:

    """

    # system non-linear equations (Larissa)
    '''qin = (1.64 * u + 35.7) * 10 ** -5
    q12 = (33.5 * (h2 - h1) + 4.31) * 10 ** -4
    qout = (8.71 * h1 ** (1 / 2) + 3.1) * 10 ** -4'''

    qout = 0.93 * (185.48 * h1 ** (1 / 2) - 167.01) * 10e-5
    qin = 1.04 * (16.46 * u - 156.93) * 10e-5
    q12 = 0.95 * (32.4 * (h2 - h1) - 83.93) * 10e-5

    return qin, q12, qout
