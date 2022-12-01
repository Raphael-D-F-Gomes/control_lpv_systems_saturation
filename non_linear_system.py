import sympy as sym
import numpy as np
from sympy.solvers import solve
import matplotlib.pyplot as plt
import control
from ortools.linear_solver import pywraplp
from scipy.signal import cont2discrete


def parameters_behavior_by_interpolation(alpha_points, h1, plot=False):

    alpha_1 = np.polyfit(h1, alpha_points[0], 5)
    alpha_2 = np.polyfit(h1, alpha_points[1], 5)
    alpha_3 = np.polyfit(h1, alpha_points[2], 5)
    alpha_4 = np.polyfit(h1, alpha_points[3], 5)
    poly_alpha_1 = np.poly1d(alpha_1)
    poly_alpha_2 = np.poly1d(alpha_2)
    poly_alpha_3 = np.poly1d(alpha_3)
    poly_alpha_4 = np.poly1d(alpha_4)

    if plot:
        plt.figure(0, figsize=(12, 9))
        plt.plot(h1, alpha_points[0], 'b')
        plt.plot(h1, poly_alpha_1(h1), 'b--')
        plt.grid()

        plt.plot(h1, alpha_points[1], 'g')
        plt.plot(h1, poly_alpha_2(h1), 'g--')

        plt.plot(h1, alpha_points[2], 'c')
        plt.plot(h1, poly_alpha_3(h1), 'c--')

        plt.plot(h1, alpha_points[3], 'k')
        plt.plot(h1, poly_alpha_4(h1), 'k--')

        plt.legend(labels=('alpha_1 (points)', 'alpha_1 (equation)',
                           'alpha_2 (points)', 'alpha_2 (equation)',
                           'alpha_3 (points)', 'alpha_3 (equation)',
                           'alpha_4 (points)', 'alpha_4 (equation)'))
        plt.xlabel('h1 [cm]')
        plt.ylabel('alphas')

        plt.show()

    return alpha_1, alpha_2, alpha_3, alpha_4


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
        op_points['h1'] = solve(q12 - qout)[0]
        q12 = q12.subs({h1: op_points['h1']})

        op_points['u'] = solve(qin - q12)[0]
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
    z = 2.5 * np.pi * (h1/100 - mu)
    w = -(h1/100 - mu) ** 2 / (2 * sigma ** 2)

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
    h1_point = h1_point / 100
    mu = 0.4
    sigma = 0.55
    r = 0.31
    z = 2.5 * np.pi * (h1_point - mu)
    w = -((h1_point - mu) ** 2) / (2 * sigma ** 2)
    area = (3 * r / 5) * (2.7 * r - (np.cos(float(z)) / (sigma * np.sqrt(2 * np.pi))) * np.exp(float(w)))

    return area


def get_non_linear_system_by_point(h1_point, period):

    area = solid_area(h1_point)
    a11 = (-30.78 / 0.3019) * 10e-5
    a12 = (30.78 / 0.3019) * 10e-5
    a21 = (30.78 / area) * 10e-5
    a22 = (-30.78 / area + (-172.4964 * np.sqrt(h1_point)) / (area * h1_point)) * 10e-5
    #b11 = 17.1184 * 10e-5
    b11 = 0
    b21 = 0
    # - 83.4737 / (0.3019 * op_points['u'])
    # 75.5858 / (area * op_points['u'])

    C = np.array([[0, 0]])
    D = np.array([[0]])
    A = np.array([[a11, a12], [a21, a22]])
    B = np.array([[b11], [b21]])

    methods = ['zoh', 'bilinear', 'euler', 'backward_diff', 'foh', 'impulse']
    discrete_sys = cont2discrete((A, B, C, D), period, method=methods[4])
    A = discrete_sys[0]
    C = discrete_sys[2]
    D = discrete_sys[3]

    op_points = operation_points({'h1': h1_point})

    xk_plus1 = np.array([[float(op_points['h2'])], [float(op_points['h1'])]])
    xk = xk_plus1
    B = (xk_plus1 - np.dot(A, xk)) / op_points['u']

    return A, B, C, D


def non_linear_system_variation(h1_min, h1_max, n_points, period, plot=False, verbose=False):

    # operation points
    system_info = {'A11': [], 'A12': [], 'A21': [], 'A22': [], 'B11': [], 'B21': []}
    h1_range = np.linspace(h1_min, h1_max, n_points)

    for h1_point in h1_range:

        sys = get_non_linear_system_by_point(h1_point, period)
        system_info['A11'].append(sys.A[0][0])
        system_info['A12'].append(sys.A[0][1])
        system_info['A21'].append(sys.A[1][0])
        system_info['A22'].append(sys.A[1][1])
        system_info['B11'].append(sys.B[0][0])
        system_info['B21'].append(sys.B[1][0])

    nv = 4
    parameters = {'A': np.zeros([nv, len(h1_range)])}

    for i, h1_point in enumerate(h1_range):

        solver = pywraplp.Solver.CreateSolver('GLOP')

        alpha1 = solver.NumVar(0, 1, 'alpha1')
        alpha2 = solver.NumVar(0, 1, 'alpha2')
        alpha3 = solver.NumVar(0, 1, 'alpha3')
        alpha4 = solver.NumVar(0, 1, 'alpha4')

        solver.Add(system_info['A21'][-1] * 2 * alpha1 == system_info['A21'][i])
        solver.Add(system_info['A22'][-1] * 2 * alpha2 == system_info['A22'][i])
        solver.Add(alpha1 + alpha2 + alpha3 + alpha4 == 1)

        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            alpha = [alpha1.solution_value(), alpha2.solution_value(), alpha3.solution_value(), alpha4.solution_value()]
            parameters['A'][0][i] = alpha[0]
            parameters['A'][1][i] = alpha[1]
            parameters['A'][2][i] = alpha[2]
            parameters['A'][3][i] = alpha[3]
        else:
            print('The problem does not have an optimal solution.')

    system_remaining_elements = {}
    for element in ['A11', 'A12', 'B11', 'B21']:

        solver = pywraplp.Solver.CreateSolver('GLOP')

        a1 = solver.NumVar(-max(system_info[element]), max(system_info[element]), 'a1')
        a2 = solver.NumVar(-max(system_info[element]), max(system_info[element]), 'a2')
        a3 = solver.NumVar(-max(system_info[element]), max(system_info[element]), 'a3')
        a4 = solver.NumVar(-max(system_info[element]), max(system_info[element]), 'a4')
        s = [solver.NumVar(0, solver.infinity(), f's{i}') for i in range(len(h1_range))]

        for i, h1_point in enumerate(h1_range):
            solver.Add(a1 * parameters['A'][0][i] + a2 * parameters['A'][1][i]
                       + a3 * parameters['A'][2][i] + a4 * parameters['A'][3][i]
                       - system_info[element][i] <= s[i])
            solver.Add(a1 * parameters['A'][0][i] + a2 * parameters['A'][1][i]
                       + a3 * parameters['A'][2][i] + a4 * parameters['A'][3][i]
                       - system_info[element][i] >= -s[i])

        solver.Minimize(sum(s))

        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            system_remaining_elements[element] = [a1.solution_value(), a2.solution_value(),
                                                  a3.solution_value(), a4.solution_value()]
        else:
            print('The problem does not have an optimal solution.')

    parameters_equation = parameters_behavior_by_interpolation(parameters['A'], h1_range, True)
    alpha_1 = np.poly1d(parameters_equation[0])
    alpha_2 = np.poly1d(parameters_equation[1])
    alpha_3 = np.poly1d(parameters_equation[2])
    alpha_4 = np.poly1d(parameters_equation[3])

    lpv_system = {}

    lpv_system['A21'] = [system_info['A21'][-1] * 2 * alpha_1(h1)
                         for h1 in h1_range]
    lpv_system['A22'] = [system_info['A22'][-1] * 2 * alpha_2(h1)
                         for h1 in h1_range]

    lpv_system['A11'] = [system_remaining_elements['A11'][0] * alpha_1(h1)
                         + system_remaining_elements['A11'][1] * alpha_2(h1)
                         + system_remaining_elements['A11'][2] * alpha_3(h1)
                         + system_remaining_elements['A11'][3] * alpha_4(h1) for h1 in h1_range]
    lpv_system['A12'] = [system_remaining_elements['A12'][0] * alpha_1(h1)
                         + system_remaining_elements['A12'][1] * alpha_2(h1)
                         + system_remaining_elements['A12'][2] * alpha_3(h1)
                         + system_remaining_elements['A12'][3] * alpha_4(h1) for h1 in h1_range]
    lpv_system['B11'] = [system_remaining_elements['B11'][0] * alpha_1(h1)
                         + system_remaining_elements['B11'][1] * alpha_2(h1)
                         + system_remaining_elements['B11'][2] * alpha_3(h1)
                         + system_remaining_elements['B11'][3] * alpha_4(h1) for h1 in h1_range]
    lpv_system['B21'] = [system_remaining_elements['B21'][0] * alpha_1(h1)
                         + system_remaining_elements['B21'][1] * alpha_2(h1)
                         + system_remaining_elements['B21'][2] * alpha_3(h1)
                         + system_remaining_elements['B21'][3] * alpha_4(h1) for h1 in h1_range]

    if plot:
        plt.figure(1, figsize=(12, 9))
        plt.plot(h1_range, system_info['A21'], 'blue')
        plt.plot(h1_range, lpv_system['A21'], 'b--')
        plt.plot(h1_range, system_info['A22'], 'r')
        plt.plot(h1_range, lpv_system['A22'], 'r--')
        plt.grid()
        plt.legend(labels=(f'A21 op point', f'A21 lpv system', f'A22 op point', f'A22 lpv system'))
        plt.xlabel('h1 [cm]')
        plt.ylabel('A')

        plt.figure(2, figsize=(12, 9))
        plt.plot(h1_range, system_info['A12'], 'blue')
        plt.plot(h1_range, lpv_system['A12'], 'b--')
        plt.plot(h1_range, system_info['A11'], 'r')
        plt.plot(h1_range, lpv_system['A11'], 'r--')
        plt.grid()
        plt.legend(labels=(f'A12 op point', f'A12 lpv system', f'A11 op point', f'A11 lpv system'))
        plt.xlabel('h1 [cm]')
        plt.ylabel('A')

        plt.figure(3, figsize=(12, 9))
        plt.plot(h1_range, system_info['B11'], 'blue')
        plt.plot(h1_range, lpv_system['B11'], 'b--')
        plt.plot(h1_range, system_info['B21'], 'r')
        plt.plot(h1_range, lpv_system['B21'], 'r--')
        plt.grid()
        plt.legend(labels=(f'B11 op point', f'B11 lpv system', f'B21 op point', f'B21 lpv system'))
        plt.xlabel('h1 [cm]')
        plt.ylabel('A')

    lpv_system = {'A1': np.array([[system_remaining_elements['A11'][0], system_remaining_elements['A12'][0]],
                                  [system_info['A21'][-1] * 2, 0]]),
                  'A2': np.array([[system_remaining_elements['A11'][1], system_remaining_elements['A12'][1]],
                                  [0, system_info['A22'][-1] * 2]]),
                  'A3': np.array([[system_remaining_elements['A11'][2], system_remaining_elements['A12'][2]],
                                  [0, 0]]),
                  'A4': np.array([[system_remaining_elements['A11'][3], system_remaining_elements['A12'][3]],
                                  [0, 0]]),
                  'B1': np.array([[system_remaining_elements['B11'][0]], [system_remaining_elements['B21'][0]]]),
                  'B2': np.array([[system_remaining_elements['B11'][1]], [system_remaining_elements['B21'][1]]]),
                  'B3': np.array([[system_remaining_elements['B11'][2]], [system_remaining_elements['B21'][2]]]),
                  'B4': np.array([[system_remaining_elements['B11'][3]], [system_remaining_elements['B21'][3]]])}

    if verbose:
        print('LPV system:\n'
              f"A1 = [[{system_remaining_elements['A11'][0]}, {system_remaining_elements['A12'][0]}],"
              f" [{system_info['A21'][-1] * 4}, {0}]]\n\n"
              f"A2 = [[{system_remaining_elements['A11'][1]}, {system_remaining_elements['A12'][1]}],"
              f" [{0}, {system_info['A22'][0] * 2}]]\n\n"
              f"A3 = [[{system_remaining_elements['A11'][2]}, {system_remaining_elements['A12'][2]}],"
              f" [{0}, {system_info['A22'][-1] * 2}]]\n\n"
              f"A4 = [[{system_remaining_elements['A11'][3]}, {system_remaining_elements['A12'][3]}],"
              f" [{0}, {0}]]\n\n"
              f"B1 = [[{system_remaining_elements['B11'][0]}],"
              f"     [{0}]]\n\n"
              f"B2 = [[{system_remaining_elements['B11'][1]}],"
              f"     [{0}]]\n\n"
              f"B3 = [[{system_remaining_elements['B11'][2]}],"
              f"     [{0}]]\n\n"
              f"B3 = [[{system_remaining_elements['B11'][3]}],"
              f"     [{system_info['B21'][-1] * 4}]]\n\n")

        print('parameters equation:\n'
              f"alpha_1 = {parameters_equation[0]}\n"
              f"alpha_2 = {parameters_equation[1]}\n"
              f"alpha_3 = {parameters_equation[2]}\n"
              f"alpha_4 = {parameters_equation[3]}\n")

    plt.show()

    return lpv_system, parameters_equation


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


if __name__ == '__main__':

    h1_min = 7
    h1_max = 37
    n_points = 100
    period = 5.62
    system_info, parameters = non_linear_system_variation(h1_min, h1_max, n_points, period, True, False)
