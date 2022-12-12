import numpy as np
import matplotlib.pyplot as plt
from non_linear_system import operation_points, non_linear_system_variation
from build_lpv_system import get_lpv_discrete_system_response, get_lpv_discrete_system_response_full_behavior
from system_validation import system_validation
from system_variation import get_lpv_model_from_op_points_range, get_system_linear_models,\
    get_system_linear_models_full_behavior
import sympy as sym


def lpv_system_validation_with_simulation_data(initial_conditions, system_info, parameters, input, time, get_reponse):

    time_sim = np.load('time.npy')
    h1 = np.load('level3.npy')[0]
    h2 = np.load('level4.npy')[0]

    h = get_reponse(system_info, initial_conditions, parameters, input)

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(time_sim, h1, label='real', color='y')
    ax[0].plot(time, h[1], label='simulated', color='b')
    ax[0].set_title('h1 [cm]')
    ax[0].legend(['real', 'simulated'])
    ax[1].plot(time_sim, h2, label='real', color='y')
    ax[1].plot(time, h[0], label='simulated', color='b')
    ax[1].set_title('h2 [cm]')
    ax[1].legend(['real', 'simulated'])
    plt.savefig('simulation.png')
    plt.show()


def lpv_system_validation(initial_conditions, system_info, parameters, time_range, period, input_variation, plot=False):

    t0 = 0
    t1 = period * time_range
    nt = int(np.ceil(t1 / period))
    t = np.linspace(t0, t1, nt)
    input = np.array([input_variation if i < len(t) / 2 else -input_variation for i in range(len(t))])

    h = get_lpv_discrete_system_response(system_info, initial_conditions, parameters, input)

    if plot:
        h2 = h[0] + initial_conditions['h2']
        h1 = h[1] + initial_conditions['h1']

        op_points_u0 = operation_points({'u': initial_conditions['u'] + input_variation})
        op_points_u1 = operation_points({'u': initial_conditions['u'] - input_variation})

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


def execute_validation(op_point, n_points, h1_min, h1_max, period, get_response, linearization):
    initial_conditions = operation_points(op_point)

    h1 = sym.var('h1')
    h2 = sym.var('h2')
    u = sym.var('u')

    system_info, parameters = get_lpv_model_from_op_points_range(h1, h2, u, period, h1_min, h1_max, n_points,
                                                                 linearization, True, True)
    # print(system_info)
    print(parameters)

    simulation_time = 12500
    simulation_step = 2

    iterations = int(simulation_time / simulation_step)

    time = np.linspace(.0, simulation_time, iterations)

    input = []

    for t in time:
        if t // 2500 == 0:
            input += [55]
        elif t // 2500 == 1:
            input += [30]
        elif t // 2500 == 2:
            input += [40]
        elif t // 2500 == 3:
            input += [35]
        elif t // 2500 >= 4:
            input += [55]

    lpv_system_validation_with_simulation_data(initial_conditions, system_info, parameters, input, time, get_response)


def execute_simple_validation(op_point, n_points, h1_min, h1_max, period, input_variation, n_iterations, get_response,
                              linearization):
    initial_conditions = operation_points(op_point)

    h1 = sym.var('h1')
    h2 = sym.var('h2')
    u = sym.var('u')

    system_info, parameters = get_lpv_model_from_op_points_range(h1, h2, u, period, h1_min, h1_max, n_points,
                                                               linearization, True, True)
    print(system_info)
    simulation_time = 1000
    time = np.linspace(.0, simulation_time, n_iterations)
    input = [initial_conditions['u'] + input_variation if i <= n_iterations / 2
             else initial_conditions['u'] - input_variation for i in range(n_iterations)]

    h = get_response(system_info, initial_conditions, parameters, input)

    op_points0 = operation_points({'u': initial_conditions['u'] + input_variation})
    op_points1 = operation_points({'u': initial_conditions['u'] - input_variation})
    h1_ref = [op_points0['h1'] if i <= n_iterations / 2 else op_points1['h1'] for i in range(n_iterations)]
    h2_ref = [op_points0['h2'] if i <= n_iterations / 2 else op_points1['h2'] for i in range(n_iterations)]

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(time, h1_ref, label='real', color='y')
    ax[0].plot(time, h[1], label='simulated', color='b')
    ax[0].set_title('h1 [cm]')
    ax[0].legend(['real', 'simulated'])
    ax[1].plot(time, h2_ref, label='real', color='y')
    ax[1].plot(time, h[0], label='simulated', color='b')
    ax[1].set_title('h2 [cm]')
    ax[1].legend(['real', 'simulated'])
    plt.savefig('simulation.png')
    plt.show()


if __name__ == '__main__':

    get_response = get_lpv_discrete_system_response
    # get_response = get_lpv_discrete_system_response_full_behavior
    linearization = get_system_linear_models
    # linearization = get_system_linear_models_full_behavior
    period = 5.62
    h1_min = 15
    h1_max = 37
    n_points = 100
    op_point = {'h1': 15}
    n_iterations = 3000
    input_variation = 5
    execute_validation(op_point, n_points, h1_min, h1_max, period, get_response, linearization)
    #execute_simple_validation(op_point, n_points, h1_min, h1_max, period, input_variation, n_iterations, get_response,
                              # linearization)
