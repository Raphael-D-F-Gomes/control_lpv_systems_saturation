import sympy as sym
import numpy as np
from non_linear_system import operation_points, state_space_non_linear_system, solid_area
from linearization import system_linearization
import matplotlib.pyplot as plt


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

        op_points = operation_points({'h1': h1_point})
        system_info['operation_points'].append(op_points)

        # state space system
        h1_p, h2_p = state_space_non_linear_system()
        system_info['non_linear_system'].append((h1_p, h2_p))

        T = 5.62
        _, sys = system_linearization(h1_p, h2_p, op_points['h1'], op_points['h2'], h1, h2, u, T)
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


def parameters_behavior_by_interpolation(alpha_points, h1, plot=False):

    alpha = [np.polyfit(h1, alpha_values, 5) for alpha_values in alpha_points]

    poly_alpha = [np.poly1d(equation) for equation in alpha]
    colors = ['b', 'k', 'r', 'm', 'c', 'g', 'y', 'p']

    if plot:
        plt.figure(0, figsize=(12, 9))
        plt.grid()
        legend = []
        for i in range(len(alpha_points)):
            plt.plot(h1, alpha_points[i], colors[i])
            plt.plot(h1, poly_alpha[i](h1), colors[i] + '--')
            legend += [f'alpha_{i+1} (points)', f'alpha_{i+1} (equation)']

        plt.legend(legend)
        plt.xlabel('h1 [cm]')
        plt.ylabel('alphas')

        plt.show()

    return alpha


def get_lpv_discrete_system_response(system_info, initial_conditions, parameters, input):

    n_vertices = len(parameters)
    h0 = np.array([[0], [0]])
    h = np.array(h0)
    h_real = np.array([[initial_conditions['h2']], [initial_conditions['h1']]])
    h1_real = initial_conditions['h1']
    poly_alpha = [np.poly1d(parameter) for parameter in parameters]

    for i in range(1, len(input)):

        '''if last_input != input[i-1]:
            h2_real = h[0][-1] + op_points['h2']
            op_points = operation_points({'u': input[i - 1]})
            h0 = np.array([[h2_real - op_points['h2']], [h1_real - op_points['h1']]])
            last_input = input[i - 1]'''

        op_points = operation_points({'h1': h1_real})
        uk = input[i - 1] - op_points['u']
        alpha = np.array([p(h1_real) for p in poly_alpha])
        A = system_info['A1'] * alpha[0]
        B = system_info['B1'] * alpha[0]

        for j in range(1, n_vertices):
            A += system_info[f'A{j+1}'] * alpha[j]
            B += system_info[f'B{j+1}'] * alpha[j]

        h_aux = np.dot(A, h0) + np.dot(B, uk)
        h = np.append(h, h_aux, axis=1)

        h0_real = np.array([[h[0][-1] + op_points['h2']], [h[1][-1] + op_points['h1']]])
        h_real = np.append(h_real, h0_real, axis=1)
        h1_real = h0_real[1][0]
        # h0 = np.array([[h[0][-1]], [h[1][-1]]])
        h0 = np.array([[0], [0]])

    return h_real


def get_lpv_discrete_system_response_full_behavior(system_info, op_points, parameters, input):

    n_vertices = len(parameters)
    h0 = np.array([[op_points['h2']], [op_points['h1']]])
    h = np.array(h0)
    poly_alpha = [np.poly1d(parameter) for parameter in parameters]

    for i in range(1, len(input)):

        h0 = np.array([[h[0][-1]], [h[1][-1]]])

        uk = input[i - 1]
        alpha = np.array([p(h0[1][0]) for p in poly_alpha])
        A = system_info['A1'] * alpha[0]
        B = system_info['B1'] * alpha[0]

        for j in range(1, n_vertices):
            A += system_info[f'A{j+1}'] * alpha[j]
            B += system_info[f'B{j+1}'] * alpha[j]

        h_aux = np.dot(A, h0) + np.dot(B, uk)  # + np.array([[0.1679304266487], [-0.0004155742875]])
        h = np.append(h, h_aux, axis=1)

    return h


if __name__ == '__main__':

    h1_min = 7
    h1_max = 37
    system_info = lpv_system(h1_min, h1_max)
    print(f"parameters behavior:\n"
          f" alpha_1 = {system_info['parameters'][0][0]}h1 + {system_info['parameters'][0][0]}\n"
          f" alpha_2 = {system_info['parameters'][1][0]}h1 + {system_info['parameters'][1][0]}\n")

    print(f"Discrete system:\n"
          f"A1 = {system_info['linear_system'][0].A},\n\n A2 = {system_info['linear_system'][1].A}\n\n"
          f"B1 = {system_info['linear_system'][0].A},\n\n B2 = {system_info['linear_system'][1].B}")
