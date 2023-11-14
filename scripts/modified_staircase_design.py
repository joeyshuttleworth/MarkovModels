import logging
import multiprocessing
import os
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import regex as re
import seaborn as sns
import pandas as pd
import numpy as np
import markovmodels
import scipy

from argparse import ArgumentParser
from markovmodels.model_generation import make_model_of_class
from markovmodels.ArtefactModel import ArtefactModel
from markovmodels.BeattieModel import BeattieModel
from markovmodels.fitting import infer_reversal_potential_with_artefact
from markovmodels.SensitivitiesMarkovModel import SensitivitiesMarkovModel
from markovmodels.voltage_protocols import detect_spikes, remove_spikes


def D_opt_utility(desc, params, s_model, hybrid=False, crhs=None, indices=None):
    """ Evaluate the D-optimality of design, d for a certain parameter vector"""
    s_model.protocol_description = desc
    s_model.voltage = markovmodels.voltage_protocols.make_voltage_function_from_description(desc)
    # output = model.make_hybrid_solver_current(njitted=False, hybrid=hybrid)

    sens = s_model.make_hybrid_solver_states(njitted=False, hybrid=False, crhs=crhs)()
    voltages = np.array([s_model.voltage(t) for t in s_model.times])
    I_Kr_sens = s_model.auxiliary_function(sens.T, params, voltages)[:, 0, :].T

    if indices is not None:
        I_Kr_sens = I_Kr_sens[indices]

    return np.log(np.linalg.det(I_Kr_sens.T @ I_Kr_sens))


def main():
    # get staircase protocol
    sc_func, sc_times, sc_desc = markovmodels.voltage_protocols.get_ramp_protocol_from_csv('staircaseramp1')

    model_name = 'model3'
    protocol = 'staircaseramp1'
    voltage_func, times, desc = markovmodels.voltage_protocols.get_ramp_protocol_from_csv(protocol)

    model = make_model_of_class(model_name, times, voltage_func,
                                protocol_description=desc)

    s_model = SensitivitiesMarkovModel(model)
    crhs = s_model.get_cfunc_rhs()

    params = model.get_default_parameters()

    voltages = np.array([voltage_func(t) for t in times])

    # Remove observations within 5ms of a spike
    removal_duration = 5
    spike_times, _ = detect_spikes(sc_times, voltages, window_size=0)
    _, _, indices = remove_spikes(sc_times, voltages, spike_times, removal_duration)
    print(D_opt_utility(sc_desc, params, s_model, indices=indices))

    global output_dir
    output_dir = markovmodels.utilities.setup_output_directory(None, 'modified_staircase')

    step_to_modify = -25

    # optimise one step (8th from last)
    def opt_func(x):
        v, t = x
        new_desc = [[t1, t2, v1, v2] for t1, t2, v1, v2 in sc_desc]
        new_tend = new_desc[step_to_modify][0] + t
        new_desc[step_to_modify][1] = new_tend

        if step_to_modify + 1 < len(desc):
            new_desc[step_to_modify + 1][0] = new_tend

        new_desc[step_to_modify][2] = v
        new_desc[step_to_modify][3] = v

        new_desc = tuple([tuple(entry) for entry in new_desc])

        util = D_opt_utility(new_desc, params, s_model, crhs=crhs, indices=indices)
        print(util)
        return -util

    t_bound = np.array([.5, 2]) * desc[step_to_modify][1] - desc[step_to_modify][0]

    res = scipy.optimize.minimize(opt_func, [0, 250], bounds=[(-120, 0), t_bound])

    # output optimised protocol
    new_desc = [[t1, t2, v1, v2] for t1, t2, v1, v2 in sc_desc]
    new_tend = new_desc[step_to_modify][0] + res.x[1]
    new_desc[step_to_modify][1] = new_tend

    if step_to_modify + 1 < len(desc):
        new_desc[step_to_modify + 1][0] = new_tend

    v = res.x[0]

    new_desc[step_to_modify][2] = v
    new_desc[step_to_modify][3] = v

    if step_to_modify + 1 < len(desc):
        new_desc[step_to_modify + 1][0] = new_tend

    v = res.x[0]

    new_desc[step_to_modify][2] = v
    new_desc[step_to_modify][3] = v
    new_desc = tuple([tuple(entry) for entry in new_desc])

    new_v_func = markovmodels.voltage_protocols.make_voltage_function_from_description(new_desc)

    fig = plt.figure()
    axs = fig.subplots(2)
    axs[1].plot(sc_times, [new_v_func(t) for t in times])

    model.protocol_description = new_desc
    model.voltage = markovmodels.voltage_protocols.make_voltage_function_from_description(new_desc)
    output = model.make_hybrid_solver_current(njitted=False, hybrid=False)()
    axs[0].plot(sc_times, output)

    fig.savefig(os.path.join(output_dir, 'optimised_protocol'))


if __name__ == '__main__':
    main()
