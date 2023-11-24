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
from markovmodels.voltage_protocols import detect_spikes, remove_spikes, get_design_space_representation
from markovmodels.fitting import infer_reversal_potential_with_artefact
from markovmodels.utilities import get_data, put_copy
from numba import njit, jit
from markovmodels.optimal_design import entropy_utility, D_opt_utility, prediction_spread_utility


seed = np.random.randint(2**32 - 1)
max_time = 15_000
leak_ramp_length = 3_400


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
    parser.add_argument('--ignore_protocols', type=str, default=['longap'], nargs='+')
    parser.add_argument('--selection_file')
    parser.add_argument('--model_class', default='model3')
    parser.add_argument('--max_iterations', '-i', type=int, default=100000)
    parser.add_argument("--experiment_name", default='25112022_MW')
    parser.add_argument("--removal_duration", default=5.0, type=float)
    parser.add_argument("--steps_at_a_time", type=int)
    parser.add_argument("--hybrid", action='store_true')
    parser.add_argument("--mode", default='spread')
    parser.add_argument("--n_sample_starting_points", type=int)
    parser.add_argument("-c", "--no_cpus", type=int, default=1)

    global args
    args = parser.parse_args()

    for protocol in args.protocols:
        if protocol in args.ignore_protocols:
            raise ValueError(f"{protocol} is specified as a protocol and with --ignore_protocls")

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

    solver = model.make_hybrid_solver_current(hybrid=args.hybrid, njitted=True)

    params = model.get_default_parameters()
    sc_voltages = np.array([voltage_func(t) for t in sc_times])

    global output_dir
    output_dir = markovmodels.utilities.setup_output_directory(None, 'max_sop')

    # optimise one step (8th from last)
    protocols = fitting_df.protocol.unique()

    if args.wells:
        passed_wells = [well for well in passed_wells if well in args.wells]

    fitting_df['score'] = fitting_df.score.astype(np.float64)
    fitting_df = pd.read_csv(args.fitting_df).sort_values('score', axis=0)
    df_rows = []
    for well in passed_wells:
        for protocol in protocols:
            if protocol in args.ignore_protocols:
                continue
            for sweep in fitting_df[(fitting_df.well == well) & \
                                    (fitting_df.protocol == protocol)]['sweep'].unique():
                fitted_params = fitting_df[(fitting_df.well == well)
                                           & (fitting_df.protocol == protocol)
                                           & (fitting_df.sweep == sweep)].head(1)[model.get_parameter_labels()].values.flatten().astype(np.float64)

                row = [well, protocol, sweep] + list(fitted_params)
                df_rows.append(row)

    params = pd.DataFrame(df_rows,
                          columns=['well', 'protocol', 'sweep'] + \
                          model.get_parameter_labels())

    n_steps = 64 - 13
    x0 = np.zeros(n_steps*2).astype(np.float64)
    x0[1::2] = 50.0
    x0[::2] = 60.0

    if args.n_sample_starting_points:
        starting_guesses = np.random.uniform(size=(x0.shape[0], args.n_sample_starting_points))
        starting_guesses[:, ::2] = (starting_guesses[:, ::2]*160) - 120
        starting_guesses[:, 1::2] = (starting_guesses[:, 1::2]*500) + 1

        scores = [opt_func([d, model, params, solver, None]) for d in starting_guesses]
        print(scores)

        best_guess_index = np.argmin(scores)
        x0 = starting_guesses[best_guess_index, :].flatten()
        print('x0', x0)

    steps_fitted = 0
    if args.steps_at_a_time is None:
        args.steps_at_a_time = int(x0.shape[0]/2)

    step_group = 0

    fig = plt.figure()
    axs = fig.subplots(2)
    sc_x = get_design_space_representation(sc_desc)

    initial_score = opt_func([x0, model, params, solver, None], ax=axs[0])
    sc_score = opt_func([sc_x, model, params, solver, None], ax=axs[1])
    print('initial score: ', initial_score)
    print('staircase score: ', sc_score)
    fig.savefig(os.path.join(output_dir, 'initial_design_sc_compare'))

    for ax in axs:
        ax.cla()

    fig.clf()
    axs = fig.subplots(2)

    previous_d = x0.astype(np.float64)
    while steps_fitted < n_steps:
        stds = np.empty(args.steps_at_a_time * 2)
        stds[::2] = .25 * (60 + 120)
        stds[1::2] = .25 * 1000

        if steps_fitted + args.steps_at_a_time > n_steps:
            steps_to_fit = n_steps - steps_fitted
            if steps_to_fit == 0:
                break
        else:
            steps_to_fit = args.steps_at_a_time

        l_bounds = [-120 if (i % 2) == 0 else 1 for i in range(steps_to_fit * 2)]
        u_bounds = [60 if (i % 2) == 0 else 2000 for i in range(steps_to_fit * 2)]

        bounds = [l_bounds, u_bounds]
        options = {'maxfevals': args.max_iterations,
                   'CMA_stds': stds,
                   'bounds': bounds,
                   'tolx': 2,
                   'tolfun': 1,
                   'popsize': 15,
                   'seed': seed
                   }

        es = cma.CMAEvolutionStrategy(previous_d[steps_fitted * 2: steps_fitted * 2 + steps_to_fit * 2], 1, options)
        with open(os.path.join(output_dir, 'pycma_seed.txt'), 'w') as fout:
            fout.write(str(seed))
            fout.write('\n')

        best_scores = []

        iteration = 0
        # Stop optimising if we've used up the majority of the time
        if args.steps_at_a_time != int(x0.shape[0]/2)\
           and previous_d[:steps_fitted*2:2].sum() >= 0.95 * (max_time - leak_ramp_length):
            break

        def get_t_range(d):
            if args.steps_at_a_time == int(x0.shape[0] / 2):
                return (0, 0)
            t_start = leak_ramp_length + d[1::2][:steps_fitted].sum()
            t_end = t_start + d[1::2][steps_fitted: steps_fitted +
                                      steps_to_fit].sum()
            t_range = (t_start, t_end)
            return t_range

        while not es.stop():
            d_list = es.ask()
            if args.steps_at_a_time != x0.shape[0] / 2:
                ind = list(range(steps_fitted * 2,
                                 (steps_fitted + steps_to_fit) * 2))
                modified_d_list = [put_copy(previous_d, ind, d) for d in d_list]

            else:
                modified_d_list = d_list

            x = [(d, model, params, solver, get_t_range(d)) for d in modified_d_list]

            # Check bounds
            res = np.array([opt_func(pars) for pars in x])

            best_scores.append(res.min())
            es.tell(d_list, res)
            if iteration % 10 == 0:
                es.result_pretty()
                d = modified_d_list[0]
                desc = markovmodels.voltage_protocols.design_space_to_desc(d)
                times = np.arange(0, desc[-1, 0], 0.5)
                initial_score = opt_func([d, model, params, solver, None], ax=axs[0])
                axs[1].plot(times, [sc_func(t, protocol_description=desc) for t in times])
                axs[0].axvspan(*get_t_range(d), alpha=.25, color='grey')
                axs[1].axvspan(*get_t_range(d), alpha=.25, color='grey')
                fig.savefig(os.path.join(output_dir,
                                         f"{step_group}_{iteration}_example.png"))
                for ax in axs:
                    ax.cla()

            if iteration % 100 == 0:
                markovmodels.optimal_design.save_es(es, output_dir,
                                                    f"optimisation_iteration_{iteration}_{step_group}")
            iteration += 1

        ind = list(range(steps_fitted * 2,
                         (steps_fitted + steps_to_fit) * 2))

        steps_fitted += steps_to_fit
        step_group += 1

        np.put(previous_d, ind, es.result.xbest)
        print(f"fitted {steps_fitted} steps")

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

    found_desc = markovmodels.voltage_protocols.design_space_to_desc(xopt.copy())
    u_D_found = markovmodels.optimal_design.D_opt_utility(found_desc,
                                                          default_params,
                                                          s_model,
                                                          removal_duration=args.removal_duration)

    print(f"u_D of found design = {u_D_found}")

    model.protocol_description = found_desc
    model.times = np.arange(0, found_desc[-1][0], .5)

    output = model.make_hybrid_solver_current(njitted=False, hybrid=False)()

    axs[0].plot(model.times, output)
    axs[1].plot(model.times, [model.voltage(t, protocol_descripton=found_desc) for t in model.times])

    fig.savefig(os.path.join(output_dir, 'optimised_protocol'))

    # Output protocol
    with open(os.path.join(output_dir, 'found_design.txt'), 'w') as fout:
        for line in markovmodels.voltage_protocols.desc_to_table(found_desc):
            fout.write(line)
            fout.write('\n')
    with open(os.path.join(output_dir, 'found_design_desc.txt'), 'w') as fout:
        for line in found_desc:
            fout.write(", ".join([str(entry) for entry in line]))
            fout.write('\n')

    axs[0].cla()
    axs[1].cla()

    # Plot phase diagram for the new design (first two states)
    model.protocol_description = found_desc
    model.times = np.arange(0, found_desc[-1][0], .5)
    states = model.make_hybrid_solver_states(njitted=False, hybrid=False,
                                             protocol_description=found_desc)()
    cols = [plt.cm.jet(i / states.shape[0]) for i in range(states.shape[0])]
    axs[0].scatter(states[:, 0], states[:, 1], alpha=.25, color=cols, marker='o')

    # Plot phase diagram (first two states)
    model.voltage = sc_func
    model.protocol_description = sc_desc
    states = model.make_hybrid_solver_states(njitted=False, hybrid=False,
                                             protocol_description=sc_desc)()
    np.savetxt(os.path.join(output_dir, 'found_design_trajectory.csv'), states)

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

    # Pickle and save optimisation results
    filename = 'es-status-pickled'
    with open(filename, 'wb') as fout:
        fout.write(es.pickle_dumps())

    markovmodels.optimal_design.save_es(es, output_dir,
                                        "es_halted")

    fig.clf()
    axs = fig.subplots(4)

    found_score = opt_func([xopt, model, params, solver, None], ax=axs[0])
    found_times = np.arange(0, found_desc[-1][0], .5)
    found_voltages = np.array([model.voltage(t, protocol_description=found_desc)\
                               for t in found_times])

    axs[1].plot(found_times, found_voltages)
    sc_score = opt_func([sc_x, model, params, solver, None], ax=axs[2])
    axs[3].plot(sc_times, sc_voltages)

    print('found score: ', found_score)
    print('staircase score: ', sc_score)

    fig.savefig(os.path.join(output_dir, 'found_design_vs_staircase'))


def opt_func(x, ax=None):
    d, model, params, solver, t_range = x

    # Force positive durations
    d = d.copy()
    d[1::2] = np.abs(d[1::2])

    # constrain total length
    protocol_length = d[1::2].sum()

    penalty = 0
    if protocol_length > max_time:
        penalty = (protocol_length - max_time) ** 2 * 1e3

    desc = markovmodels.voltage_protocols.design_space_to_desc(d)
    model.protocol_description = desc
    times = np.arange(0, desc[-1][0], .5)
    model.times = times

    voltages = np.array([model.voltage(t, protocol_description=desc)
                         for t in model.times])

    spike_times, _ = detect_spikes(times, voltages, window_size=0)
    _, _, indices = remove_spikes(times, voltages, spike_times,
                                  args.removal_duration)

    params = params.loc[np.all(np.isfinite(params[model.get_parameter_labels()]), axis=1), :]

    utils = []
    for well in params.well.unique():
        sub_df = params[params.well == well]
        util = prediction_spread_utility(desc,
                                         sub_df[model.get_parameter_labels()].values,
                                         model, indices=indices,
                                         removal_duration=args.removal_duration,
                                         ax=ax, mode=args.mode, solver=solver,
                                         t_range=t_range)
        utils.append(util)
    utils = np.array(utils) - penalty

    # Return the worst score across wells
    return -np.min(utils)


if __name__ == '__main__':
    main()
