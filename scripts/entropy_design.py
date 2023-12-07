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
from markovmodels.optimal_design import entropy_utility, D_opt_utility


seed = np.random.randint(2**32 - 1)
max_time = 15_000

def main():
    parser = ArgumentParser()
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
    parser.add_argument("--n_voxels_per_variable", type=int, default=10)
    parser.add_argument("--n_skip", type=int, default=10)
    parser.add_argument("--n_sample_starting_points", type=int)
    parser.add_argument("-c", "--no_cpus", type=int, default=1)

    global args
    args = parser.parse_args()

    if args.artefact_default_kinetic_param_file:
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
    voltage_func, times, desc = markovmodels.voltage_protocols.get_ramp_protocol_from_csv(protocol)

    model = make_model_of_class(args.model_class, times, voltage_func,
                                protocol_description=desc,
                                default_parameters=default_parameters)

    solver = model.make_hybrid_solver_states(hybrid=False)

    params = model.get_default_parameters()
    voltages = np.array([voltage_func(t) for t in times])

    global output_dir
    output_dir = markovmodels.utilities.setup_output_directory(None, 'max_entropy')

    if args.subtraction_df_file:
        subtraction_df = pd.read_csv(args.subtraction_df_file)

    if args.qc_df_file:
        qc_df = pd.read_csv(args.qc_df_file)
        qc_df = qc_df[qc_df.drug == 'before']
        # qc_df = qc_df.set_index(['well', 'protocol', 'sweep'])

    if args.protocols:
        qc_df = qc_df[(qc_df.protocol.isin(args.protocols))]

    if args.wells:
        qc_df = qc_df[(qc_df.well.isin(args.wells))]
        passed_wells = [well for well in passed_wells if well in args.wells]

    if args.sweeps:
        qc_df = qc_df[qc_df.sweep.astype(str).isin(args.sweeps)]

    x0 = markovmodels.voltage_protocols.get_design_space_representation(sc_desc).flatten()
    x0[1::2] = x0[1::2] / 2

    n_additional_steps = int(64 - 13 - (x0.shape[0] / 2))
    x0 = np.append(x0, [10.0] * 2 * n_additional_steps)

    stds = np.empty(x0.shape)
    stds[::2] = .25 * (60 + 120)
    stds[1::2] = .25 * 1000

    n_steps = 64 - 13
    l_bounds = [-120 if (i % 2) == 0 else 1 for i in range(n_steps * 2)]
    u_bounds = [60 if (i % 2) == 0 else 2000 for i in range(n_steps * 2)]

    bounds = l_bounds, u_bounds
    options = {'maxfevals': args.max_iterations,
               'CMA_stds': stds,
               'bounds': bounds,
               'tolx': 2,
               'tolfun': 1e-3,
               'popsize': max(args.no_cpus, 15),
               'seed': seed
               }

    s_model = SensitivitiesMarkovModel(model,
                                       parameters_to_use=model.get_parameter_labels())

    if args.n_sample_starting_points:
        starting_guesses = np.random.uniform(size=(args.n_sample_starting_points, x0.shape[0]))
        starting_guesses[:, ::2] = (starting_guesses[:, ::2]*180) - 120
        starting_guesses[:, 1::2] = (starting_guesses[:, 1::2]*500) + 1

        scores = [opt_func([d, s_model, params, solver]) for d in starting_guesses]
        print(scores)

        scores = [s for s in scores if np.isfinite(s)]

        best_guess_index = np.argmin(scores)
        x0 = starting_guesses[best_guess_index, :].flatten()

    es = cma.CMAEvolutionStrategy(x0, 1, options)

    with open(os.path.join('pycma_seed.txt'), 'w') as fout:
        fout.write(str(seed))
        fout.write('\n')

    print('x0', x0)

    iteration = 0
    params = model.get_default_parameters()
    while not es.stop():
        d_list = es.ask()
        x = [(d, model, params, solver) for d in d_list]
        res = list(map(opt_func, x))
        es.tell(d_list, res)
        if iteration % 10 == 0:
            es.result_pretty()
        if iteration % 100 == 0:
            markovmodels.optimal_design.save_es(es, output_dir,
                                                f"optimisation_iteration_{iteration}")
        iteration += 1

    xopt = es.result[0]

    # Check D_optimality of design vs staircase
    default_params = model.get_default_parameters()

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

    fig = plt.figure()
    axs = fig.subplots(2)
    found_voltage_func = markovmodels.voltage_protocols.make_voltage_function_from_description(found_desc)

    model.protocol_description = found_desc
    model.voltage = found_voltage_func
    model.times = np.arange(0, found_desc[-1][0], .5)
    output = model.make_hybrid_solver_states(njitted=False, hybrid=False)()


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

    states = model.make_hybrid_solver_states(njitted=False, hybrid=False)()[::args.n_skip]
    cols = [plt.cm.jet(i / states.shape[0]) for i in range(states.shape[0])]
    axs[0].scatter(states[:, 0], states[:, 1], alpha=.25, color=cols, marker='o')

    # Plot phase diagram (first two states)
    model.voltage = sc_func
    model.protocol_description = sc_desc
    states = model.make_hybrid_solver_states(njitted=False, hybrid=False)()
    np.savetxt(os.path.join('found_design_trajectory.csv'), states)
    states = states[::args.n_skip]

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

    markovmodels.optimal_design.save_es(es, output_dir,
                                        "es_halted")


def opt_func(x):
    d, model, params, solver = x

    # constrain total length
    protocol_length = d[1::2].sum()

    penalty = 0
    if protocol_length > max_time:
        penalty = (protocol_length - max_time)**2 * 1e3

    desc = markovmodels.voltage_protocols.design_space_to_desc(d)

    util = entropy_utility(desc, params, model,
                           removal_duration=args.removal_duration,
                           solver=solver)
    return -util + penalty


if __name__ == '__main__':
    main()
