import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from non_linear_system import operation_points, state_space_non_linear_system, solid_area
from linearization import system_linearization, get_response_discrete_system


def system_validation(h1, h2, u, op_point, period, time_range, input_variation, plot=False):

    op_points = operation_points(op_point)

    diff_h1, diff_h2 = state_space_non_linear_system()

    A, B, C, D = system_linearization(diff_h1, diff_h2, op_points['h1'], op_points['h2'], h1, h2, u, period)

    t0 = 0
    t1 = period * time_range
    nt = int(np.ceil(t1 / period))
    t = np.linspace(t0, t1, nt)
    F = np.array([input_variation if i < len(t) / 2 else -input_variation for i in range(len(t))])

    h1_0 = 0
    h2_0 = 0
    h0 = np.array([[h2_0], [h1_0]])

    h = get_response_discrete_system(A, B, t, F, h0)

    if plot:
        h2 = h[0] + op_points['h2']
        h1 = h[1] + op_points['h1']

        op_points_u0 = operation_points({'u': op_points['u'] + input_variation})
        op_points_u1 = operation_points({'u': op_points['u'] - input_variation})

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

    return h


if __name__ == '__main__':

    # system variables
    h1 = sym.var('h1')
    h2 = sym.var('h2')
    u = sym.var('u')
    op_point = {'h1': 7}
    period = 5.62
    input_variation = 0.5
    plot = True
    time_range = 2000
    sys, h = system_validation(h1, h2, u, op_point, period, time_range, input_variation, plot)
