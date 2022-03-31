#!/usr/bin/env python3

from MarkovModels import common
import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import pints
import pints.plot
import argparse
from pathos.multiprocessing import ProcessPool as Pool

from MarkovModels.BeattieModel import BeattieModel
from MarkovModels.LinearModel import LinearModel

from numba import njit

import matplotlib.pyplot as plt
import matplotlib as mpl

# Don't use scientific notation offsets on plots (it's confusing)
mpl.rcParams["axes.formatter.useoffset"] = False

sigma2 = 0.01**2


def main():
    plt.style.use('classic')
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--removal_durations_file", type=str)
    parser.add_argument("-R", "--removal_durations", nargs='+', type=float)
    parser.add_argument('-o', '--output')
    parser.add_argument('-n', '--no_experiments', default=10)
    parser.add_argument('-s', '--short', action='store_true')
    parser.add_argument('-c', '--cpus', default=1, type=int)
    parser.add_argument('-i', '--max_iterations', type=int)
    parser.add_argument('-r', '--repeats', default=1)
    args = parser.parse_args()

    assert not (args.removal_durations_file and args.removal_durations)

    if args.removal_durations:
        removal_durations = args.removal_durations
    elif args.removal_durations_file:
        removal_durations = pd.read_csv(args.removal_durations_file)['removal_duration'].values.flatten().astype(np.float64)
    else:
        assert False

    output_dir = common.setup_output_directory(args.output, 'mle_errors')

    if args.short:
        removal_durations = removal_durations[[0, -1]]

    params = np.array([2.07E-3, 7.17E-2, 3.44E-5, 6.18E-2, 4.18E-1, 2.58E-2,
                       4.75E-2, 2.51E-2, 3.33E-2])

    protocol_func, tstart, tend, tstep, protocol_desc = common.get_ramp_protocol_from_csv('staircase')

    full_times = np.linspace(tstart, tend, 150000)

    voltages = np.array([protocol_func(t) for t in full_times])

    spike_times, spike_indices = common.detect_spikes(full_times, voltages,
                                                      window_size=0)

    Erev = common.calculate_reversal_potential(310.15)

    model = BeattieModel(times=full_times, voltage=protocol_func, Erev=Erev, parameters=params)
    model.protocol_description = protocol_desc
    mean_trajectory = model.SimulateForwardModel()

    simulated_data = [mean_trajectory + np.random.normal(0, np.sqrt(sigma2), len(full_times))
                      for i in range(args.no_experiments)]

    def get_mle_error(time_to_remove, data):
        indices = common.remove_indices(list(range(len(full_times))),
                                        [(spike,
                                          int(spike + time_to_remove / tstep))
                                         for spike in spike_indices])
        model = BeattieModel(times=full_times, voltage=protocol_func, Erev=Erev, parameters=params)
        model.protocol_description = protocol_desc
        solver = model.make_hybrid_solver_current()

        mle, _ = common.fit_model(model, data, params, subset_indices=indices,
                                  solver=solver,
                                  max_iterations=args.max_iterations,
                                  repeats=3)

        score = np.sqrt(np.sum((solver(mle) - mean_trajectory)**2))

        return score, mle

    args_list = [(r, data) for r in removal_durations for data in simulated_data]

    print("number of fitting tasks", len(args_list))

    pool = Pool(min(args.cpus, len(removal_durations) * args.repeats * args.no_experiments))

    mle_errors, mles = list(zip(*pool.map(get_mle_error, *zip(*args_list))))

    mle_errors = np.array(mle_errors).reshape((len(removal_durations),
                                               args.no_experiments), order='C')

    fig = plt.figure(figsize=(9, 9))
    ax = fig.subplots()

    ax.plot(removal_durations, np.mean(mle_errors, axis=1), ls='--',
            marker='x', label='mean RMSE in MLE prediction')

    xs = [removal_durations[i] for i in range(mle_errors.shape[0]) for j in range(mle_errors.shape[1])]
    ax.scatter(xs, mle_errors, label='RMSE in MLE prediction')
    ax.set_xlabel('time remove after each spike / ms')
    ax.set_ylabel('RMSE from MLE predictions')
    ax.legend()

    fig.savefig(os.path.join(output_dir, 'mle_errors'))

    np.save(os.path.join(output_dir, 'mle_errors.npy'), mle_errors)
    np.save(os.path.join(output_dir, 'mles.npy'), mles)

    fits_dir = os.path.join(output_dir, 'fits')

    if not os.path.exists(fits_dir):
        os.makedirs(fits_dir)

    # plot fits
    fits_fig, fits_ax = plt.subplots()
    for i in range(len(removal_durations)):
        fits_ax.plot(full_times, model.SimulateForwardModel(mles[i]), label='fitted model')
        fits_ax.plot(full_times, mean_trajectory, label='data', color='grey', alpha=.5)
        fits_ax.set_xlabel('time / ms')
        fits_ax.set_ylabel('current / nA')
        fits_fig.savefig(os.path.join(fits_dir, f"{removal_durations[i]}_removed.png"))

        ax.legend()
        fits_ax.cla()

    plt.close(fits_fig)


if __name__ == "__main__":
    main()
