import numpy as np
import matplotlib.pyplot as plt
from non_linear_system import operation_points
from build_lpv_system import get_lpv_discrete_system_response
from system_validation import system_validation
import sympy as sym


def lpv_system_validation(initial_conditions, system_info, parameters, input, time):

    time_sim = np.load('time.npy')
    h1 = np.load('level3.npy')[0]
    h2 = np.load('level4.npy')[0]

    h = get_lpv_discrete_system_response(system_info, initial_conditions, parameters, input)

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(time_sim, h1, label='real', color='y')
    ax[0].plot(time, h[1] + initial_conditions['h1'], label='simulated', color='b')
    ax[0].set_title('h1 [cm]')
    ax[0].legend(['real', 'simulated'])
    ax[1].plot(time_sim, h2, label='real', color='y')
    ax[1].plot(time, h[0] + initial_conditions['h2'], label='simulated', color='b')
    ax[1].set_title('h2 [cm]')
    ax[1].legend(['real', 'simulated'])
    plt.savefig('simulation.png')
    plt.show()


if __name__ == '__main__':

    initial_conditions = operation_points({'u': 55})

    system_info = {'A1': np.array([[0.9584366195787899, 0.05203895336807539],
                                   [1.7080459164461486, 0]]),
                   'A2': np.array([[0.903468750323001, 0.05203895336807539],
                                   [0, 1.739494467646276]]),
                   'A3': np.array([[0.9562674154279642, 0.05203895336807539],
                                   [0, 0.7392788987777366]]),
                   'A4': np.array([[0.9584366195787899, -0.0034507773230491867],
                                   [0, 0]]),
                   'B1': np.array([[0.03113440092276224], [0]]),
                   'B2': np.array([[0.030511985868078958], [0]]),
                   'B3': np.array([[0.031103048736836673], [0]]),
                   'B4': np.array([[0.03113440092276224], [0.03207515126707978]])}

    alpha_1 = [-3.16271176e-08, 3.51740270e-06, -1.36508196e-04, 2.51129576e-03, -2.07718961e-02, 9.90654553e-02]
    alpha_2 = [4.69169416e-08, -5.04130447e-06, 1.97852950e-04, -3.81096560e-03, 3.59193154e-02, 5.84897603e-02]
    alpha_3 = [6.31841299e-09, -1.10411984e-06, 4.51642619e-05, -7.23074583e-04, 1.93516208e-03, 7.58235658e-01]
    alpha_4 = [-2.16082371e-08, 2.62802161e-06, -1.06509016e-04, 2.02274442e-03, -1.70825814e-02, 8.42091262e-02]
    parameters = [alpha_1, alpha_2, alpha_3, alpha_4]

    simulation_time = 12500
    simulation_step = 5.62

    iterations = int(simulation_time / simulation_step)

    time = np.linspace(.0, simulation_time, iterations)
    time_span = [time[0], time[-1]]

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

    lpv_system_validation(initial_conditions, system_info, parameters, input, time)

    '''
    op_point = {'h1': 19.9152504178994}
    period = 5.62
    input_variation = 2
    plot = True
    time_range = 2000
    sys, h = system_validation(sym.var('h1'), sym.var('h2'), sym.var('u'),
     op_point, period, time_range, input_variation, plot)'''
