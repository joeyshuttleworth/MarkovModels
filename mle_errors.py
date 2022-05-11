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
import uuid

from MarkovModels.BeattieModel import BeattieModel
from MarkovModels.LinearModel import LinearModel
from MarkovModels.ArtefactModel import ArtefactModel

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
    parser.add_argument('-n', '--no_experiments', default=10, type=int)
    parser.add_argument('-s', '--short', action='store_true')
    parser.add_argument('-c', '--cpus', default=1, type=int)
    parser.add_argument('-i', '--max_iterations', type=int)
    parser.add_argument('-r', '--repeats', default=1, type=int)
    parser.add_argument("-m", "--method", default='CMAES')
    parser.add_argument('-A', '--use_artefact_model', action='store_true')

    global args
    args = parser.parse_args()

    if args.method == 'CMAES':
        optimiser = pints.CMAES
    elif args.method == 'NelderMead':
        optimiser = pints.NelderMead
    else:
        assert False

    assert not (args.removal_durations_file and args.removal_durations)

    if args.removal_durations:
        removal_durations = args.removal_durations
    elif args.removal_durations_file:
        removal_durations = pd.read_csv(args.removal_durations_file)['removal_duration'].values.flatten().astype(np.float64)
    elif args.short:
        removal_durations = np.array([0, 10])
    else:
        removal_durations = np.linspace(0, 50, 11)

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

    channel_model = BeattieModel(times=full_times, voltage=protocol_func,
                                 Erev=Erev, protocol_description=protocol_desc,
                                 parameters=params)
    global solver
    solver = channel_model.make_forward_solver_current()

    if args.use_artefact_model:
        model = ArtefactModel(channel_model)
        data_generator = model.make_solver()

    else:
        data_generator = solver

    mean_trajectory = data_generator(p=params)

    # Simulate data
    simulated_data = [mean_trajectory + np.random.normal(0, np.sqrt(sigma2), len(full_times))
                      for i in range(args.no_experiments)]

    longap_func, longap_tstart, longap_tend, longap_tstep, longap_desc = common.get_ramp_protocol_from_csv('longap')

    longap_times = np.linspace(longap_tstart, longap_tend, int((longap_tend - 0)/longap_tstep))

    validation_dir = os.path.join(output_dir, 'prediction_errors')

    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)

    def get_mle_error(time_to_remove, data):
        indices = common.remove_indices(list(range(len(full_times))),
                                        [(spike,
                                          int(spike + time_to_remove / tstep))
                                         for spike in spike_indices])

        # Use Beattie parameters (default)
        beattie_parameters = np.array((2.26E-4, 6.99E-2, 3.44E-5, 5.460E-2, 0.0873,
                                            8.91E-3, 5.15E-3, 0.003158, 0.1524))

        model = BeattieModel(times=full_times, voltage=protocol_func,
                             Erev=Erev, parameters=beattie_parameters,
                             protocol_description=protocol_desc)

        mle, _ = common.fit_model(model, data, params, subset_indices=indices,
                                  solver=solver,
                                  max_iterations=args.max_iterations,
                                  repeats=args.repeats,
                                  method=optimiser)

        validation_model = BeattieModel(times=longap_times,
                                        voltage=longap_func, Erev=Erev,
                                        parameters=params,
                                        protocol_description=longap_desc)

        validation_solver = validation_model.make_hybrid_solver_current()

        validation_trajectory = validation_solver()
        prediction = validation_solver(p=mle)

        # Plot prediction
        fig = plt.figure(figsize=(12, 9))
        axs = fig.subplots(2)

        axs[0].plot(longap_times, prediction, label='prediction')
        axs[0].plot(longap_times, validation_trajectory, label='true value')
        axs[1].set_xlabel('times')
        axs[0].set_title('longap prediction')
        axs[1].plot(longap_times, (prediction - validation_trajectory),
                    label='prediction_errors')

        axs[0].legend()
        axs[1].legend()
        fig.savefig(os.path.join(validation_dir, f"{time_to_remove:.2f}_removed_prediction_{uuid.uuid4()}.png"))

        plt.close(fig)

        # Compute error when predicting `longap``
        score = np.sqrt(np.sum((prediction - validation_trajectory)**2))

        return score, mle

    args_list = np.array([[r, data] for r in removal_durations for data in
                          simulated_data], dtype=object)

    print("number of fitting tasks", len(args_list))

    pool = Pool(min(args.cpus, len(args_list)))

    mle_errors, mles = list(zip(*pool.map(get_mle_error, *zip(*args_list))))

    param_labels = channel_model.get_parameter_labels()
    results_df = pd.DataFrame(np.column_stack((args_list[:,0], mles,
                                               mle_errors)), columns=['removal_duration'] + param_labels + ['mle_error'])

    results_df.to_csv(os.path.join(output_dir, 'mle_errors_results.csv'))

    mle_errors = np.array(mle_errors).reshape((len(removal_durations),
                                               args.no_experiments), order='C')

    mles = np.array(mles).reshape((len(removal_durations), args.no_experiments,
                                   -1), order='C')

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
    fits_fig = plt.figure(figsize=(12, 10))
    fits_axs = fits_fig.subplots(3)

    for i in range(mles.shape[0]):
        for j in range(mles.shape[1]):
            fits_axs[0].plot(full_times, model.SimulateForwardModel(mles[i, j]), label='fitted model')
            fits_axs[0].plot(full_times, simulated_data[j], label='data', color='grey', alpha=.5)
            fits_axs[0].set_xlabel('time / ms')
            fits_axs[0].set_ylabel('current / nA')

            fits_axs[1].plot(full_times, model.SimulateForwardModel(mles[i, j]) -
                             mean_trajectory, label='fitted model error')
            fits_axs[1].set_xlabel('time / ms')
            fits_axs[1].set_ylabel('current / nA')

            fits_axs[2].plot(full_times, voltages, label='V_in')
            fits_axs[2].plot(full_times, model.SimulateForwardModel(mles[i, j],
                                                                    return_current=False)[:,
                                                                                          -1],
                             label='V_m')

            for ax in fits_axs:
                ax.legend()
            fits_fig.savefig(os.path.join(fits_dir, f"{removal_durations[i]}_removed_run_{j}.png"))
            for ax in fits_axs:
                ax.cla()

    # plot errors
    fits_fig, fits_ax = plt.subplots()
    for i in range(mles.shape[0]):
        for j in range(mles.shape[1]):

            ax.legend()
            fits_ax.cla()

    plt.close(fits_fig)


if __name__ == "__main__":
    main()
