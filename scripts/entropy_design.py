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
import pints
import cma

from argparse import ArgumentParser
from markovmodels.model_generation import make_model_of_class
from markovmodels.ArtefactModel import ArtefactModel
from markovmodels.BeattieModel import BeattieModel
from markovmodels.fitting import infer_reversal_potential_with_artefact
from markovmodels.SensitivitiesMarkovModel import SensitivitiesMarkovModel
from markovmodels.voltage_protocols import detect_spikes, remove_spikes
from markovmodels.fitting import infer_reversal_potential_with_artefact
from markovmodels.utilities import get_data
from numba import njit, jit


def entropy_utility(desc, params, model, hybrid=False, removal_duration=5, include_vars=None):
    """ Evaluate the D-optimality of design, d for a certain parameter vector"""
    model.protocol_description = desc
    model.voltage = markovmodels.voltage_protocols.make_voltage_function_from_description(desc)
    # output = model.make_hybrid_solver_current(njitted=False, hybrid=hybrid)
    times = np.arange(0, desc[-1][0], .5)
    voltages = np.array([model.voltage(t) for t in times])

    # Remove observations within 5ms of a spike
    removal_duration = args.removal_duration
    spike_times, _ = detect_spikes(times, voltages, window_size=0)
    _, _, indices = remove_spikes(times, voltages, spike_times, removal_duration)

    states = model.make_hybrid_solver_states(njitted=False, hybrid=False)()
    n_boxes_per_variable = 10

    times_in_each_box = count_box_visitations(states[:, include_vars],
                                              n_boxes_per_variable, times, indices)

    log_prob = np.full(times_in_each_box.shape, 0)
    visited_box_indices = times_in_each_box != 0
    log_prob[visited_box_indices] = -np.log(times_in_each_box[visited_box_indices] / times[indices].shape[0]).flatten()
    return np.sum(log_prob * (times_in_each_box / times[indices].shape[0]))


@jit
def count_box_visitations(states, n_boxes_per_variable, times, indices):
    boxes = np.full((times.shape[0], states.shape[1]), 0, dtype=int)
    n_skip = 100

    for i, (t, x) in enumerate(zip(times[indices][::n_skip], states[indices, ][::n_skip, :])):
        # Get box
        for j in range(x.shape[0]):
            boxes[i, j] = np.floor(x.flatten()[j] * n_boxes_per_variable)

    no_states = states.shape[1]
    times_in_each_box = np.zeros([n_boxes_per_variable] * no_states).astype(int)

    for box in boxes:
        times_in_each_box[*box.astype(int)] += 1

    return times_in_each_box


def main():
    parser = ArgumentParser()
    parser.add_argument('data_directory')
    parser.add_argument('--reversal', type=float, default=-91.71)
    parser.add_argument('-o', '--output')
    parser.add_argument('--subtraction_df_file')
    parser.add_argument('--qc_df_file')
    parser.add_argument('--use_parameter_file')
    parser.add_argument('--artefact_default_kinetic_param_file')
    parser.add_argument('--wells', '-w', type=str, default=[], nargs='+')
    parser.add_argument('--sweeps', '-s', type=str, default=[], nargs='+')
    parser.add_argument('--protocols', type=str, default=[], nargs='+')
    parser.add_argument('--selection_file')
    parser.add_argument('--model_class', default='model3')
    parser.add_argument('--max_iterations', '-i', type=int, default=100000)
    parser.add_argument("--experiment_name", default='25112022_MW')
    parser.add_argument("--removal_duration", default=5.0, type=float)

    global args
    args = parser.parse_args()

    params_for_Erev = \
        np.loadtxt(args.artefact_default_kinetic_param_file).flatten().astype(np.float64)

    if args.use_parameter_file:
        default_parameters = \
            np.loadtxt(args.use_parameter_file).flatten().astype(np.float64)
    else:
        default_parameters = make_model_of_class(args.model_class).get_default_parameters()

    if args.selection_file:
        with open(args.selection_file) as fin:
            passed_wells = fin.read().splitlines()
    else:
        passed_wells = None

    # get staircase protocol
    sc_func, sc_times, sc_desc = markovmodels.voltage_protocols.get_ramp_protocol_from_csv('staircaseramp1')


    protocol = 'staircaseramp1'
    voltage_func, times, desc = markovmodels.voltage_protocols.get_ramp_protocol_from_csv(protocol)

    model = make_model_of_class(args.model_class, times, voltage_func,
                                protocol_description=desc,
                                default_parameters=default_parameters)

    params = model.get_default_parameters()
    voltages = np.array([voltage_func(t) for t in times])

    print(entropy_utility(sc_desc, params, model))

    global output_dir
    output_dir = markovmodels.utilities.setup_output_directory(None, 'modified_staircase')

    step_to_modify = -25

    if args.subtraction_df_file:
        subtraction_df = pd.read_csv(args.subtraction_df_file)

    if args.qc_df_file:
        qc_df = pd.read_csv(args.qc_df_file)
        qc_df = qc_df[qc_df.drug == 'before']
        # qc_df = qc_df.set_index(['well', 'protocol', 'sweep'])

    # optimise one step (8th from last)
    protocols = subtraction_df.protocol.unique()

    if args.protocols:
        subtraction_df = subtraction_df[(subtraction_df.protocol.isin(args.protocols))]
        qc_df = qc_df[(qc_df.protocol.isin(args.protocols))]

    if args.wells:
        subtraction_df = subtraction_df[(subtraction_df.well.isin(args.wells))]
        qc_df = qc_df[(qc_df.well.isin(args.wells))]
        passed_wells = [well for well in passed_wells if well in args.wells]

    if args.sweeps:
        subtraction_df = subtraction_df[(subtraction_df.sweep.astype(str).isin(args.sweeps))]
        qc_df = qc_df[qc_df.sweep.astype(str).isin(args.sweeps)]

    df_rows = []
    for well in passed_wells:
        for protocol in protocols:
            for sweep in subtraction_df[(subtraction_df.well == well) & \
                                        (subtraction_df.protocol == protocol)]['sweep'].unique():
                leak_row = subtraction_df[(subtraction_df.well == well) &\
                                          (subtraction_df.protocol == protocol) \
                                          & (subtraction_df.sweep == sweep)]

                gleak = leak_row['pre-drug leak conductance'].values[0]
                Eleak = leak_row['pre-drug leak reversal'].values[0]
                qc_row = qc_df[(qc_df.well == well) & (qc_df.protocol == protocol) \
                               & (qc_df.sweep == sweep)][['Rseries', 'Cm']]
                Rseries = qc_row['Rseries'].values[0] * 1e-9
                Cm = qc_row['Cm'].values[0] * 1e9
                data = get_data(well, protocol, args.data_directory,
                                args.experiment_name, label=None, sweep=sweep)

                times_df = pd.read_csv(os.path.join(args.data_directory,
                                                    f"{args.experiment_name}-{protocol}-times.csv"))
                times = times_df['time'].to_numpy().flatten()

                E_obs = leak_row['fitted_E_rev'].values[0]
                V_off = E_obs - args.reversal
                param_set = np.concatenate((model.get_default_parameters(),
                                           np.array([gleak, Eleak, 0, 0, V_off, Rseries, Cm])))

                prot_func, _, desc = markovmodels.voltage_protocols.get_ramp_protocol_from_csv(protocol)
                voltages = np.array([prot_func(t) for t in times])

                c_model = make_model_of_class(args.model_class, voltage=prot_func,
                                              times=times, E_rev=args.reversal,
                                              protocol_description=desc)

                solver = c_model.make_forward_solver_current(njitted=True)
                default_parameters = c_model.get_default_parameters()

                # Remove observations within 5ms of a spike
                removal_duration = args.removal_duration
                spike_times, _ = detect_spikes(times, voltages, window_size=0)
                _, _, indices = remove_spikes(times, voltages, spike_times, removal_duration)

                # @njit
                def min_func(g_kr):
                    p = default_parameters.copy()
                    p[-1] = g_kr
                    return np.sum((solver(p)[indices] - data[indices]) ** 2)

                # Minimise SSE to find best conductance
                res = scipy.optimize.minimize_scalar(min_func, method='bounded', bounds=[0, 1e5])
                gkr = res.x
                row = [well, protocol, sweep] + list(param_set)
                row[-8] = gkr
                df_rows.append(row)

    params = pd.DataFrame(df_rows,
                          columns=['well', 'protocol', 'sweep'] + \
                          ArtefactModel(model).get_parameter_labels(),
                          )

    a_model = ArtefactModel(model)

    @jit
    def opt_func(x):
        # Force positive durations
        # x[1::2] = np.abs(x[1::2])

        # constrain total length
        if x[1::2].sum() > 12_500:
            return np.inf

        # Constrain voltage
        # if np.any(x[::2] < -120) or np.any(x[::2] > 60):
        #     return np.inf

        desc = markovmodels.voltage_protocols.design_space_to_desc(x)
        params.set_index(['well', 'protocol', 'sweep'])
        kinetic_indices = [i for i in range(a_model.get_no_state_vars() - 1)]
        utils = np.array([entropy_utility(desc, params, a_model,
                                          include_vars=kinetic_indices) for
                          p_vec in params.values])
        return -utils.mean()

    # t_bound = np.array([.5, 2]) * desc[step_to_modify][1] - desc[step_to_modify][0]

    # res = scipy.optimize.minimize(opt_func, [-60, 500], bounds=[(-120, 60), [250, 750]])

    x0 = markovmodels.voltage_protocols.get_design_space_representation(sc_desc)
    stds = np.empty(x0.shape)
    stds[::2] = .25 * (60 + 120)
    stds[1::2] = .25 * 1000

    n_steps = 60 - 13
    bounds = [[-120, 60] if (i % 2) == 0 else [1, 3000] for i in range(n_steps)]
    options = {'maxfevals': args.max_iterations,
               'CMA_stds': stds,
               'bounds': bounds
               }

    xopt, es = cma.fmin2(opt_func, [x0], 1, options=options)
    s_model = SensitivitiesMarkovModel(model,
                                       parameters_to_use=[i for i in range(len(model.get_parameter_labels()))])

    # Check D_optimality of design vs staircase
    params = model.get_default_parameters()
    u_D_staircase = markovmodels.optimal_design.D_opt_utility(sc_desc,
                                                              params,
                                                              s_model,
                                                              indices=indices)
    print(f"u_D of staircase = {u_D_staircase}")

    found_desc = markovmodels.voltage_protocols.design_space_to_desc(xopt)

    u_D_found = markovmodels.optimal_design.D_opt_utility(found_desc,
                                                          params,
                                                          s_model,
                                                          indices=indices)

    print(f"u_D of found design = {u_D_found}")

    # # print(res.x)
    # # output optimised protocol
    # new_desc = [[t1, t2, v1, v2] for t1, t2, v1, v2 in sc_desc]
    # # new_tend = new_desc[step_to_modify][0] + res.x[1]
    # new_desc[step_to_modify][1] = new_tend

    # if step_to_modify + 1 < len(desc):
    #     new_desc[step_to_modify + 1][0] = new_tend

    # v = res.x[0]

    new_desc = markovmodels.voltage_protocols.design_space_to_desc(xopt)
    # new_desc = tuple([tuple(entry) for entry in new_desc])
    fig = plt.figure()
    axs = fig.subplots(2)

    model.protocol_description = new_desc
    model.voltage = markovmodels.voltage_protocols.make_voltage_function_from_description(new_desc)
    model.times = np.arange(0, new_desc[-1][0], .5)
    output = model.make_hybrid_solver_current(njitted=False, hybrid=False)()
    axs[0].plot(model.times, output)
    axs[1].plot(model.times, [model.voltage(t) for t in model.times])

    fig.savefig(os.path.join(output_dir, 'optimised_protocol'))


if __name__ == '__main__':
    main()
