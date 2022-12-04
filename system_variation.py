import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from non_linear_system import operation_points, state_space_non_linear_system, get_non_linear_system_by_point
from linearization import system_linearization, get_response_discrete_system
from build_lpv_system import parameters_behavior_by_interpolation
from ortools.linear_solver import pywraplp


def get_system_linear_models(h1_range, period):

    h1 = sym.var('h1')
    h2 = sym.var('h2')
    u = sym.var('u')

    system_info = {'A11': [], 'A12': [], 'A21': [], 'A22': [], 'B11': [], 'B21': []}

    for h1_point in h1_range:
        op_points = operation_points({'h1': h1_point})

        # state space system
        h1_p, h2_p = state_space_non_linear_system()

        A, B, C, D = system_linearization(h1_p, h2_p, op_points['h1'], op_points['h2'], h1, h2, u, period)
        system_info['A11'].append(A[0][0])
        system_info['A12'].append(A[0][1])
        system_info['A21'].append(A[1][0])
        system_info['A22'].append(A[1][1])
        system_info['B11'].append(B[0][0])
        system_info['B21'].append(B[1][0])

    return system_info


def get_system_linear_models_full_behavior(h1_range, period):

    system_info = {'A11': [], 'A12': [], 'A21': [], 'A22': [], 'B11': [], 'B21': []}

    for h1_point in h1_range:

        A, B, C, D = get_non_linear_system_by_point(h1_point, period)
        system_info['A11'].append(A[0][0])
        system_info['A12'].append(A[0][1])
        system_info['A21'].append(A[1][0])
        system_info['A22'].append(A[1][1])
        system_info['B11'].append(B[0][0])
        system_info['B21'].append(B[1][0])

    return system_info


def get_parameters_points(system_info, gains, h1_range, nv, elements):
    parameters_values = {'A': np.zeros([nv, len(h1_range)])}

    for i, h1_point in enumerate(h1_range):

        solver = pywraplp.Solver.CreateSolver('GLOP')

        parameters = [solver.NumVar(0, 1, f'alpha{i+1}') for i in range(nv)]

        for j, element in enumerate(elements):
            solver.Add(max(system_info[element]) * gains[element] * parameters[j] == float(system_info[element][i]))

        solver.Add(sum(parameters) == 1)

        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            alpha = [parameter.solution_value() for parameter in parameters]
            for j in range(nv):
                parameters_values['A'][j][i] = alpha[j]
        else:
            print('The problem does not have an optimal solution.')

    return parameters_values


def get_remaining_elements_values(remaining_elements, system_info, nv, h1_range, parameters):

    system_remaining_elements = {}
    for element in remaining_elements:

        solver = pywraplp.Solver.CreateSolver('GLOP')

        a = [solver.NumVar(-float(max(system_info[element])) * 100, float(max(system_info[element])) * 100, f'a{j+1}')
             for j in range(nv)]

        # s = [solver.NumVar(0, solver.infinity(), f's{i}') for i in range(len(h1_range))]
        s = solver.NumVar(0, solver.infinity(), f's')

        for i, h1_point in enumerate(h1_range):

            equation = [parameters['A'][j][i] * a[j] for j in range(nv)]
            solver.Add(sum(equation) - float(system_info[element][i]) <= s)
            solver.Add(sum(equation) - float(system_info[element][i]) >= -s)

        solver.Minimize(s)

        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            system_remaining_elements[element] = [value.solution_value() for value in a]
        else:
            print('The problem does not have an optimal solution.')

    return system_remaining_elements


def get_lpv_model_from_op_points_range(h1, h2, u, period, h1_min, h1_max, n_points, linearization,
                                       plot: bool = False, verbose: bool = True):

    h1_range = np.linspace(h1_min, h1_max, n_points)
    system_info = linearization(h1_range, period)

    nv = 3
    gains = {'A21': 4, 'A22': 4}
    elements = list(gains.keys())
    parameters = get_parameters_points(system_info, gains, h1_range, nv, elements)

    remaining_elements = list(set(list(system_info.keys())) - set(elements))
    system_remaining_elements = get_remaining_elements_values(remaining_elements, system_info, nv, h1_range, parameters)

    parameters_equation = parameters_behavior_by_interpolation(parameters['A'], h1_range, plot)
    alpha = []
    for j in range(nv):
        alpha += [np.poly1d(parameters_equation[j])]

    lpv_system = {}

    for i, element in enumerate(elements):
        lpv_system[element] = [max(system_info[element]) * gains[element] * alpha[i](h1) for h1 in h1_range]

    for element in remaining_elements:
        lpv_system[element] = [sum([system_remaining_elements[element][j] * alpha[j](h1) for j in range(nv)])
                               for h1 in h1_range]

    if plot:
        plot_elements_comparison(system_info, lpv_system, h1_range)

    lpv_complete_system = get_lpv_complete_system(system_info, nv, elements, system_remaining_elements, gains, verbose)

    return lpv_complete_system, parameters_equation


def plot_elements_comparison(system_info, lpv_system, h1_range):
    colors = ['b', 'k', 'r', 'm', 'c', 'g', 'y', 'p']
    for i, element in enumerate(system_info):
        plt.figure(i, figsize=(12, 9))
        plt.plot(h1_range, system_info[element], colors[i])
        plt.plot(h1_range, lpv_system[element], colors[i] + '--')
        plt.legend(labels=(f'{element} op point', f'{element} lpv system'))
        plt.grid()
        plt.xlabel('h1 [cm]')
        plt.ylabel(element)

    plt.show()


def get_lpv_complete_system(system_info, nv, elements, system_remaining_elements, gains, verbose):
    lpv_complete_system = {}
    for i, vertice in enumerate([f'A{j + 1}' for j in range(nv)]):

        b_vertice = f'B{i+1}'

        lpv_complete_system[vertice] = np.zeros([2, 2])
        lpv_complete_system[b_vertice] = np.zeros([2, 1])

        vertice_system = {}
        for element in system_info:

            if element in elements:
                if elements.index(element) == i:
                    vertice_system[element] = max(system_info[element]) * gains[element]
                else:
                    vertice_system[element] = 0
            else:
                vertice_system[element] = system_remaining_elements[element][i]

            row = int(element[-2]) - 1
            column = int(element[-1]) - 1
            if element[0] == 'A':
                lpv_complete_system[vertice][row][column] = vertice_system[element]
            if element[0] == 'B':
                lpv_complete_system[b_vertice][row][column] = vertice_system[element]

        if verbose:
            print(f'\n\n{vertice}:\n', lpv_complete_system[vertice])
            print(f'\n\n{b_vertice}:\n', lpv_complete_system[b_vertice])

    return lpv_complete_system


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

    get_lpv_model_from_op_points_range(h1, h2, u, period, h1_min, h1_max, n_points, plot, verbose)
