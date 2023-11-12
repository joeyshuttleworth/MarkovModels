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
import loky

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
from markovmodels.optimal_design import entropy_utility, D_opt_utility, prediction_spread_utility


def main():
    parser = ArgumentParser()
    parser.add_argument('data_directory')
    parser.add_argument('fitting_df')
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
    parser.add_argument("-c", "--no_cpus", type=int, default=1)

    global args
    args = parser.parse_args()

    fitting_df = pd.read_csv(args.fitting_df)

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
    global sc_func
    sc_func, sc_times, sc_desc = markovmodels.voltage_protocols.get_ramp_protocol_from_csv('staircaseramp1')


    protocol = 'staircaseramp1'
    voltage_func, sc_times, desc = markovmodels.voltage_protocols.get_ramp_protocol_from_csv(protocol)

    model = make_model_of_class(args.model_class, sc_times, voltage_func,
                                protocol_description=desc,
                                default_parameters=default_parameters)

    params = model.get_default_parameters()
    sc_voltages = np.array([voltage_func(t) for t in sc_times])

    global output_dir
    output_dir = markovmodels.utilities.setup_output_directory(None, 'max_sop')

    if args.subtraction_df_file:
        subtraction_df = pd.read_csv(args.subtraction_df_file)

    if args.qc_df_file:
        qc_df = pd.read_csv(args.qc_df_file)
        qc_df = qc_df[qc_df.drug == 'before']
        # qc_df = qc_df.set_index(['well', 'protocol', 'sweep'])

    # optimise one step (8th from last)
    protocols = fitting_df.protocol.unique()

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

    # fitting_df = pd.read_csv(args.fitting_df).sort_values('score', axis=0)
    df_rows = []
    for well in passed_wells:
        for protocol in protocols:
            for sweep in fitting_df[(fitting_df.well == well) & \
                                    (fitting_df.protocol == protocol)]['sweep'].unique():
                # leak_row = subtraction_df[(subtraction_df.well == well) &\
                #                           (subtraction_df.protocol == protocol) \
                #                           & (subtraction_df.sweep == sweep)]

                # gleak = leak_row['pre-drug leak conductance'].values[0]
                # Eleak = leak_row['pre-drug leak reversal'].values[0]
                # qc_row = qc_df[(qc_df.well == well) & (qc_df.protocol == protocol) \
                #                & (qc_df.sweep == sweep)][['Rseries', 'Cm']]
                # Rseries = qc_row['Rseries'].values[0] * 1e-9
                # Cm = qc_row['Cm'].values[0] * 1e9
                # data = get_data(well, protocol, args.data_directory,
                #                 args.experiment_name, label=None, sweep=sweep)

                # times_df = pd.read_csv(os.path.join(args.data_directory,
                #                                     f"{args.experiment_name}-{protocol}-times.csv"))
                # times = times_df['time'].to_numpy().flatten()

                # E_obs = leak_row['fitted_E_rev'].values[0]
                # V_off = E_obs - args.reversal

                fitted_params = fitting_df[(fitting_df.well == well)
                                           & (fitting_df.protocol == protocol)
                                           & (fitting_df.sweep == sweep)].head(1)[model.get_parameter_labels()].values.flatten().astype(np.float64)

                # param_set = np.concatenate((fitted_params,
                #                            np.array([gleak, Eleak, 0, 0, V_off, Rseries, Cm])))
                row = [well, protocol, sweep] + list(fitted_params)
                df_rows.append(row)

    params = pd.DataFrame(df_rows,
                          columns=['well', 'protocol', 'sweep'] + \
                          model.get_parameter_labels())

    cfunc = model.get_cfunc_rhs()

    x0 = markovmodels.voltage_protocols.get_design_space_representation(sc_desc).flatten()

    x0[1::2] = x0[1::2] / 2

    n_additional_steps = int(64 - 13 - (x0.shape[0] / 2))
    x0 = np.append(x0, [10] * 2 * n_additional_steps)

    print('initial score: ', opt_func([x0, model, params, cfunc]))
    stds = np.empty(x0.shape)
    stds[::2] = .25 * (60 + 120)
    stds[1::2] = .25 * 1000

    n_steps = 64 - 13
    l_bounds = [-120 if (i % 2) == 0 else 1 for i in range(n_steps * 2)]
    u_bounds = [60 if (i % 2) == 0 else 2000 for i in range(n_steps * 2)]

    bounds = [l_bounds, u_bounds]
    seed = np.random.randint(2**32 - 1)
    options = {'maxfevals': args.max_iterations,
               'CMA_stds': stds,
               'bounds': bounds,
               'tolx': 2,
               'tolfun': 1,
               'popsize': 15,
               'seed': seed
               }

    es = cma.CMAEvolutionStrategy(x0, 1, options)
    with open(os.path.join('pycma_seed.txt'), 'w') as fout:
        fout.write(str( seed))
        fout.write('\n')

    best_scores = []
    while not es.stop():
        d_list = es.ask()
        x = [(d, model, params, cfunc) for d in d_list]
        # Check bounds

        res = np.array([opt_func(pars) for pars in x])

        best_scores.append(res.min())
        es.tell(d_list, res)
        es.result_pretty()

    np.savetxt(os.path.join('best_scores_from_generations'), np.array(best_scores))

    xopt = es.result.xbest
    s_model = SensitivitiesMarkovModel(model,
                                       parameters_to_use=model.get_parameter_labels())

    default_params = model.get_default_parameters()
    # Check D_optimality of design vs staircase
    u_D_staircase = markovmodels.optimal_design.D_opt_utility(sc_desc,
                                                              default_params,
                                                              s_model,
                                                              removal_duration=args.removal_duration)
    print(f"u_D of staircase = {u_D_staircase}")

    found_desc = markovmodels.voltage_protocols.design_space_to_desc(xopt)
    u_D_found = markovmodels.optimal_design.D_opt_utility(found_desc,
                                                          default_params,
                                                          s_model,
                                                          removal_duration=args.removal_duration)

    print(f"u_D of found design = {u_D_found}")

    found_voltage_func = markovmodels.voltage_protocols.make_voltage_function_from_description(found_desc)
    model.protocol_description = found_desc
    model.voltage = found_voltage_func
    model.times = np.arange(0, found_desc[-1][0], .5)

    output = model.make_hybrid_solver_current(njitted=False, hybrid=False)()

    fig = plt.figure()
    axs = fig.subplots(2)
    axs[0].plot(model.times, output)
    axs[1].plot(model.times, [model.voltage(t) for t in model.times])

    fig.savefig(os.path.join(output_dir, 'optimised_protocol'))

    # Output protocol
    with open(os.path.join(output_dir, 'found_design.txt'), 'w') as fout:
        for line in markovmodels.voltage_protocols.desc_to_table(desc):
            fout.write(line)
            fout.write('\n')
    with open(os.path.join(output_dir, 'found_design_desc.txt'), 'w') as fout:
        for line in found_desc:
            fout.write(", ".join([str(entry) for entry in line]))
            fout.write('\n')

    axs[0].cla()
    axs[1].cla()

    # Plot phase diagram for the new design (first two states)
    model.voltage = found_voltage_func
    model.protocol_description = found_desc
    states = model.make_hybrid_solver_states(njitted=False, hybrid=False)()
    cols = [plt.cm.jet(i / states.shape[0]) for i in range(states.shape[0])]
    axs[0].scatter(states[:, 0], states[:, 1], alpha=.25, color=cols, marker='o')

    # Plot phase diagram (first two states)
    model.voltage = sc_func
    model.protocol_description = sc_desc
    states = model.make_hybrid_solver_states(njitted=False, hybrid=False)()
    np.savetxt(os.path.join('found_design_trajectory.csv'), states)

    model.voltage = sc_func
    model.protocol_description = sc_desc
    cols = [plt.cm.jet(i / states.shape[0]) for i in range(states.shape[0])]
    axs[1].scatter(states[:, 0], states[:, 1], alpha=.25, color=cols, marker='o')

    fig.savefig(os.path.join(output_dir, "phase_diagrams.png"))

    # output_score
    with open(os.path.join(output_dir, 'best_score.txt'), 'w') as fout:
        fout.write(str(es.result[1]))
        fout.write('\n')

    with open(os.path.join(output_dir, 'u_d.txt'), 'w') as fout:
        fout.write(str(u_D_found))
        fout.write('\n')


def opt_func(x):
    d, model, params, cfunc = x

    # Force positive durations
    d = d.copy()
    d[1::2] = np.abs(d[1::2])

    model.times = np.arange(0, d[1::2].sum(), .5)

    # constrain total length
    protocol_length = d[1::2].sum()
    if protocol_length > 15_000:
        return np.inf

    # Constrain voltage
    # if np.any(x[::2] < -120) or np.any(x[::2] > 60):
    #     return np.inf

    desc = markovmodels.voltage_protocols.design_space_to_desc(d)

    # ignore Vm state
    kinetic_indices = [i for i in range(model.get_no_state_vars())]

    params = params.loc[np.all(np.isfinite(params[model.get_parameter_labels()]), axis=1), :]

    utils = []
    for well in params.well.unique():
        sub_df = params[params.well == well]
        util = prediction_spread_utility(desc,
                                         sub_df[model.get_parameter_labels()].values,
                                         model,
                                         removal_duration=args.removal_duration,
                                         cfunc=cfunc)
        utils.append(util)
    utils = np.array(utils)
    print('utils are', utils)
    return -np.min(utils)


if __name__ == '__main__':
    main()
