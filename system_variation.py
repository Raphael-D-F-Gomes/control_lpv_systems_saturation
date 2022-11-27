import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from non_linear_system import operation_points, state_space_non_linear_system, solid_area
from linearization import system_linearization, get_response_discrete_system
from build_lpv_system import parameters_behavior_by_interpolation
from ortools.linear_solver import pywraplp


def get_lpv_model_from_op_points_range(h1, h2, u, h1_min, h1_max, n_points, period,
                                       plot: bool=False, verbose:bool =False):

    # operation points
    system_info = {'A11': [], 'A12': [], 'A21': [], 'A22': [], 'B11': [], 'B21': []}
    h1_range = np.linspace(h1_min, h1_max, n_points)

    for h1_point in h1_range:

        op_points = operation_points({'h1': h1_point})

        # state space system
        h1_p, h2_p = state_space_non_linear_system()


        _, sys = system_linearization(h1_p, h2_p, op_points['h1'], op_points['h2'], h1, h2, u, period)
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

        solver.Add(system_info['A21'][-1] * 4 * alpha1 == system_info['A21'][i])
        solver.Add(system_info['A22'][0] * 2 * alpha2 + system_info['A22'][-1] * 2 * alpha3 == system_info['A22'][i])
        solver.Add(system_info['B21'][-1] * 4 * alpha4 == system_info['B21'][i])
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
    for element in ['A11', 'A12', 'B11']:

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
                                                  a3.solution_value(),  a4.solution_value()]
        else:
            print('The problem does not have an optimal solution.')


    parameters_equation = parameters_behavior_by_interpolation(parameters['A'], h1_range, True)
    alpha_1 = np.poly1d(parameters_equation[0])
    alpha_2 = np.poly1d(parameters_equation[1])
    alpha_3 = np.poly1d(parameters_equation[2])
    alpha_4 = np.poly1d(parameters_equation[3])

    lpv_system = {}

    lpv_system['A21'] = [system_info['A21'][-1] * 4 * alpha_1(h1) for h1 in h1_range]
    lpv_system['B21'] = [system_info['B21'][-1] * 4 * alpha_4(h1) for h1 in h1_range]
    lpv_system['A22'] = [system_info['A22'][0] * 2 * alpha_2(h1) + system_info['A22'][-1] * 2 * alpha_3(h1)
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
                                   [system_info['A21'][-1] * 4, 0]]),
                   'A2': np.array([[system_remaining_elements['A11'][1], system_remaining_elements['A12'][1]],
                                   [0, system_info['A22'][0] * 2]]),
                   'A3': np.array([[system_remaining_elements['A11'][2], system_remaining_elements['A12'][2]],
                                   [0, system_info['A22'][-1] * 2]]),
                   'A4': np.array([[system_remaining_elements['A11'][3], system_remaining_elements['A12'][3]],
                                   [0, 0]]),
                   'B1': np.array([[system_remaining_elements['B11'][0]], [0]]),
                   'B2': np.array([[system_remaining_elements['B11'][1]], [0]]),
                   'B3': np.array([[system_remaining_elements['B11'][2]], [0]]),
                   'B4': np.array([[system_remaining_elements['B11'][3]], [system_info['B21'][-1] * 4]])}

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


if __name__ == '__main__':

    h1 = sym.var('h1')
    h2 = sym.var('h2')
    u = sym.var('u')

    h1_min = 7
    h1_max = 37
    period = 5.62
    plot = True
    verbose = True
    n_points = 100

    get_lpv_model_from_op_points_range(h1, h2, u, h1_min, h1_max, n_points, period,
                                       plot, verbose)