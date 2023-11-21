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
from markovmodels.utilities import get_data, put_copy
from numba import njit, jit
from markovmodels.optimal_design import D_opt_utility, discriminate_spread_of_predictions_utility


def main():
    parser = ArgumentParser()
    parser.add_argument('data_directory')
    parser.add_argument('fitting_dfs', nargs=2)
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
    parser.add_argument('--model_classes', default=('model3', 'Wang'), nargs=2)
    parser.add_argument('--max_iterations', '-i', type=int, default=100000)
    parser.add_argument("--experiment_name", default='25112022_MW')
    parser.add_argument("--removal_duration", default=5.0, type=float)
    parser.add_argument("--steps_at_a_time", type=int)
    parser.add_argument("--n_sample_starting_points", type=int)
    parser.add_argument("-c", "--no_cpus", type=int, default=1)

    global args
    args = parser.parse_args()

    fitting_dfs = [pd.read_csv(fname) for fname in args.fitting_dfs]

    for i in range(len(fitting_dfs)):
        fitting_dfs[i].score = fitting_dfs[i].score.astype(np.float64)
        fitting_dfs[i] = fitting_dfs[i].sort_values('score', ascending=True)

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

    models = [0, 0]
    models[0] = make_model_of_class(args.model_classes[0], sc_times,
                                    voltage_func, protocol_description=desc)

    models[1] = make_model_of_class(args.model_classes[1], sc_times,
                                    voltage_func, protocol_description=desc)

    sc_voltages = np.array([voltage_func(t) for t in sc_times])

    global output_dir
    output_dir = markovmodels.utilities.setup_output_directory(None, 'sop_discriminate')

    # optimise one step (8th from last)
    protocols = fitting_dfs[0].protocol.unique()

    if args.wells:
        passed_wells = [well for well in passed_wells if well in args.wells]

    df_rows = []

    params = []
    for model, fitting_df in zip(models, fitting_dfs):

        for well in passed_wells:
            for protocol in protocols:
                if protocol in args.ignore_protocols:
                    continue
                for sweep in fitting_df[(fitting_df.well == well) & \
                                        (fitting_df.protocol == protocol)]['sweep'].unique():
                    fitted_params = fitting_df[(fitting_df.well == well) &
                                               (fitting_df.protocol == protocol) &
                                               (fitting_df.sweep == sweep)].head(1)[model.get_parameter_labels()].values.flatten().astype(np.float64)

                    row = [well, protocol, sweep] + list(fitted_params)
                    df_rows.append(row)

        params.append(pd.DataFrame(df_rows,
                                   columns=['well', 'protocol', 'sweep'] + \
                                   model.get_parameter_labels()))

    params[0]['noise'] = [get_noise(row) for _, row in params[0].iterrows()]
    params[1]['noise'] = [get_noise(row) for _, row in params[1].iterrows()]

    n_steps = 64 - 13
    x0 = np.zeros(n_steps).astype(np.float64)
    x0[1::2] = 100.0
    x0[::2] = -80.0

    if args.n_sample_starting_points:
        starting_guesses = \
            np.random.uniform(size=(args.n_sample_starting_points, x0.shape[0]))

        starting_guesses[:, ::2] = (starting_guesses[:, ::2]*160) - 120
        starting_guesses[:, 1::2] = (starting_guesses[:, 1::2]*500) + 1

        scores = [opt_func([d, models, params]) for d in starting_guesses]
        print(scores)

        best_guess_index = np.argmin(scores)
        x0 = starting_guesses[best_guess_index, :].flatten()
        print('x0', x0)

    steps_fitted = 0
    if args.steps_at_a_time is None:
        args.steps_at_a_time = int(x0.shape[0]/2)

    previous_d = x0.astype(np.float64)

    step_group = 0
    while steps_fitted != int(x0.shape[0] / 2):
        print('initial score: ', opt_func([x0, models, params]))
        stds = np.empty(args.steps_at_a_time * 2)
        stds[::2] = .25 * (40 + 120)
        stds[1::2] = .25 * 1000

        l_bounds = [-120 if (i % 2) == 0 else 1 for i in range(args.steps_at_a_time * 2)]
        u_bounds = [40 if (i % 2) == 0 else 2000 for i in range(args.steps_at_a_time * 2)]

        bounds = [l_bounds, u_bounds]
        seed = np.random.randint(2**32 - 1)
        options = {'maxfevals': args.max_iterations,
                   'CMA_stds': stds,
                   'bounds': bounds,
                   'tolx': 2,
                   'tolfun': 1e-5,
                   'popsize': 15,
                   'seed': seed
                   }

        es = cma.CMAEvolutionStrategy(x0[steps_fitted * 2: steps_fitted * 2 + args.steps_at_a_time * 2], 1, options)
        with open(os.path.join('pycma_seed.txt'), 'w') as fout:
            fout.write(str(seed))
            fout.write('\n')

        best_scores = []

        iteration = 0
        while not es.stop():
            d_list = es.ask()
            if args.steps_at_a_time != x0.shape[0] / 2:
                ind = list(range(steps_fitted * 2,
                                 (steps_fitted + args.steps_at_a_time) * 2))
                [put_copy(previous_d, ind, d) for d in d_list]

            x = [(d, models, params) for d in d_list]
            # Check bounds

            res = np.array([opt_func(pars) for pars in x])

            best_scores.append(res.min())
            es.tell(d_list, res)
            es.result_pretty()
            if iteration % 10 == 0:
                es.result_pretty()
            if iteration % 100 == 0:
                markovmodels.optimal_design.save_es(es, output_dir,
                                                    f"optimisation_iteration_{iteration}_{step_group}")
            iteration += 1
        steps_fitted += args.steps_at_a_time
        step_group += 1
        x0 = es.result.xbest
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
    model.voltage = found_voltage_func
    model.protocol_description = found_desc
    states = model.make_hybrid_solver_states(njitted=False, hybrid=False)()
    cols = [plt.cm.jet(i / states.shape[0]) for i in range(states.shape[0])]
    axs[0].scatter(states[:, 0], states[:, 1], alpha=.25, color=cols, marker='o')

    # Plot phase diagram (first two states)
    model.voltage = sc_func
    model.protocol_description = sc_desc
    states = model.make_hybrid_solver_states(njitted=False, hybrid=False)()
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


def opt_func(x, ax=None, hybrid=False):
    d, models, (params1, params2) = x

    model1, model2 = models

    # Force positive durations
    d = d.copy()
    d[1::2] = np.abs(d[1::2])

    model1.times = np.arange(0, d[1::2].sum(), .5)
    model2.times = np.arange(0, d[1::2].sum(), .5)

    # constrain total length
    protocol_length = d[1::2].sum()
    if protocol_length > 15_000:
        return np.inf

    # Constrain voltage
    # if np.any(x[::2] < -120) or np.any(x[::2] > 60):
    #     return np.inf

    desc = markovmodels.voltage_protocols.design_space_to_desc(d)

    params1 = params1.loc[np.all(np.isfinite(params1[model1.get_parameter_labels()]), axis=1), :]

    params2 = params2.loc[np.all(np.isfinite(params2[model2.get_parameter_labels()]), axis=1), :]

    wells = params1.well.unique().flatten()
    wells = [w for w in wells if w in params2.well.unique()]

    solver1 = model1.make_hybrid_solver_current(hybrid=hybrid)
    solver2 = model2.make_hybrid_solver_current(hybrid=hybrid)

    utils = []
    for well in wells:
        sub_df1 = params1[params1.well == well]
        sub_df2 = params2[params2.well == well]

        noise = sub_df1.noise.values.flatten()[0]

        util = discriminate_spread_of_predictions_utility(desc,
                                                          sub_df1[model1.get_parameter_labels()].values,
                                                          sub_df2[model2.get_parameter_labels()].values,
                                                          model1, model2,
                                                          removal_duration=args.removal_duration,
                                                          sigma2=noise**2,
                                                          ax=ax, solver1=solver1,
                                                          solver2=solver2)

        utils.append(util)
    utils = np.array(utils)

    print('utils are', utils)
    return -np.min(utils)


def get_noise(row):
    well, protocol, sweep = row[['well', 'protocol', 'sweep']]

    data = markovmodels.utilities.get_data(well, protocol, args.data_directory,
                                           experiment_name=args.experiment_name,
                                           sweep=sweep)

    return data[:200].std()


if __name__ == '__main__':
    main()
