#!/usr/bin/env python3

import multiprocessing
import regex as re
import logging
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import pints
import argparse
from markovmodels import common
from markovmodels.BeattieModel import BeattieModel
from markovmodels.ClosedOpenModel import ClosedOpenModel
from markovmodels.KempModel import KempModel

from threadpoolctl import threadpool_limits

import os
import pandas as pd
import numpy as np
from numba import njit

import matplotlib
# matplotlib.use('Agg')


T = 298
K_in = 120
K_out = 5

pool_kws = {'maxtasksperchild': 1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_directory')
    parser.add_argument('--output')
    parser.add_argument('--optimise', action='store_true')
    parser.add_argument('--max_iterations', '-i', type=int, default=100000)
    parser.add_argument('--repeats', type=int, default=16)
    parser.add_argument('--removal_duration', '-r', default=5, type=float)
    parser.add_argument('--cores', '-c', default=1, type=int)
    parser.add_argument('--model', '-m', default='Beattie', type=str)
    parser.add_argument('--experiment_name', default='synthetic', type=str)
    parser.add_argument('--figsize', '-f', help='mcmc chains to run', type=int)
    parser.add_argument('--use_parameter_file')
    parser.add_argument('--protocols', default=['staircase'], nargs='+')
    parser.add_argument('--wells', default=[0, 1], nargs='+')
    parser.add_argument('--noise', default=0.05, type=float)
    parser.add_argument('--no_repeats', default=100, type=int)
    parser.add_argument('--no_parameter_steps', default=25, type=int)
    parser.add_argument('--fix_params', default=[], type=int, nargs='+')
    parser.add_argument('--sampling_frequency', default=0.1, type=float)
    parser.add_argument('--parameter_index', type=int, default=-1)

    global args
    args = parser.parse_args()

    global model_class
    model_class = common.get_model_class(args.model)

    fig = plt.figure()
    ax = fig.subplots()

    global output_dir
    output_dir = common.setup_output_directory(args.output,
                                               'test_parameter_profile')

    for protocol in args.protocols:
        print(f'protocol: {protocol}')
        voltage_func, times, protocol_desc = common.get_ramp_protocol_from_csv(protocol)

        if args.data_directory:
            times = pd.read_csv(os.path.join(args.data_directory,
                                             f"{args.experiment_name}-{protocol}-times.csv"),
                                float_precision='round_trip')['time'].values.astype(np.float64).flatten()

        voltages = np.array([voltage_func(t) for t in times])
        spike_times, _ = common.detect_spikes(times, voltages, window_size=0)

        _, _, indices = common.remove_spikes(times, voltages, spike_times,
                                                args.removal_duration)

        model = model_class(voltage=voltage_func, times=times,
                            protocol_description=protocol_desc)

        parameter_labels = model_class().get_parameter_labels()

        parameters = model.get_default_parameters().copy()
        for well in args.wells:
            if args.data_directory:
                data = common.get_data(well, protocol, args.data_directory,
                                       experiment_name=args.experiment_name)
            else:
                data, times = generate_data(protocol, 1, Erev=None)
                data = data[0, :].flatten()

            model = model_class(voltage=voltage_func, times=times,
                                protocol_description=protocol_desc)
            solver = model.make_forward_solver_current()
            do_parameter_profiles(solver, parameters, args.parameter_index,
                                  data, indices, times, ax=ax)

        ax.axvline(parameters[args.parameter_index], ls='--')
        fig.savefig(os.path.join(output_dir,
                                 f"{args.model}_{protocol}_{parameter_labels[args.parameter_index]}"
                                 "_plot"))
        ax.cla()


def generate_data(protocol, no_repeats, Erev, output=False):

    prot_func, _times, desc = common.get_ramp_protocol_from_csv(protocol)
    print('generating data')

    no_samples = int((_times[-1] - _times[0]) / args.sampling_frequency) + 1

    times = np.linspace(_times[0], (no_samples - 1) * args.sampling_frequency,
                        no_samples)

    times_df = pd.DataFrame(times.T, columns=('time',))
    times_df.to_csv(os.path.join(output_dir, f"{args.experiment_name}-{protocol}-times.csv"))
    model = model_class(voltage=prot_func, times=times, Erev=Erev,
                        protocol_description=desc)

    mean = model.make_forward_solver_current()()

    all_data = []
    if not np.all(np.isfinite(mean)):
        print('inf times', times[np.argwhere(~np.isfinite(mean))])
        raise Exception()

    for repeat in range(no_repeats):
        data = np.random.normal(mean, args.noise, times.shape)

        # Output data
        if output:
            out_fname = os.path.join(output_dir, f"{args.experiment_name}-{protocol}-{repeat}.csv")
            pd.DataFrame(data.T, columns=('current',)).to_csv(out_fname)

        all_data.append(data)

    return np.vstack(all_data), times


@njit
def utility_func(solver, data, parameters, subset_indices, noise):
    # if np.any(~np.isfinite(solver(parameters))):
    #     print(np.argwhere(~np.isfinite(solver(parameters))))

    SSE = np.sum((solver(parameters)[subset_indices] - data[subset_indices])**2)
    n = data.shape[0]
    ll = -n * 0.5 * np.log(2 * np.pi * noise**2) - SSE / (2 * noise**2)
    return ll


def do_parameter_profiles(solver, true_parameters, parameter_index, data,
                          indices, times, ax):
    true_val = true_parameters[parameter_index]
    p_range = np.linspace(true_val*.99, true_val*1.01, 51)

    utility_vec = np.empty(shape=p_range.shape)

    noise = args.noise

    def optim_func(p):
        sim_parameters = true_parameters.copy()
        sim_parameters[parameter_index] = p
        return utility_func(solver, data, sim_parameters, indices, noise)

    for i, p in enumerate(p_range):
        utility_vec[i] = optim_func(p)

    # find optimum
    if args.optimise:
        res = scipy.optimize.minimize_scalar(optim_func, bounds=[[p_range[0], p_range[1]]])
        print(res)

    ax.plot(p_range, utility_vec)
    # ax.axvline(true_val, ls='--', label='true_value')
    if args.optimise:
        ax.axvline(res.x, ls='--', label='optimised_value')


if __name__ == "__main__":
    multiprocessing.freeze_support()
    with threadpool_limits(limits=1, user_api='blas'):
        main()
