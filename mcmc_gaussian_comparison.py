#full_datafull_data!/usr/bin/env python3

from MarkovModels import common
import logging
import pathos
import pints
import os
import numpy as np
import pandas as pd
import seaborn as sns
from MarkovModels.MarkovModel import MarkovModel
from MarkovModels.BeattieModel import BeattieModel
from quality_control.leak_fit import fit_leak_lr
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import argparse
import regex as re
import itertools
import uuid


params = np.array([2.07E-3, 7.17E-2, 3.44E-5, 6.18E-2, 4.18E-1, 2.58E-2,
                   4.75E-2, 2.51E-2, 3.33E-2])

sigma2 = 0.01**2
Erev = common.calculate_reversal_potential(310.15)


def main():

    description = ''

    parser = argparse.ArgumentParser(description)

    parser.add_argument('input_dir', type=str, help="path to the directory containing the plot_criteria results")
    parser.add_argument('--output_dir', '-o', type=str, help="output path")
    parser.add_argument('--max_iterations', '-i', type=int)
    parser.add_argument('--repeats', '-r', type=int, default=1)
    parser.add_argument('--short', '-s', action='store_true')
    parser.add_argument('--cpus', '-c', type=int)
    parser.add_argument('--removal_durations', '-R', nargs='+')
    parser.add_argument("-m", "--method", default='CMAES')

    args = parser.parse_args()

    if args.method == 'CMAES':
        optimiser = pints.CMAES
    elif args.method == 'NelderMead':
        optimiser = pints.NelderMead
    else:
        assert False

    output_dir = common.setup_output_directory(args.output_dir, 'mcmc_gaussian_comparison')

    # Load in synthetic data
    data_df = pd.read_csv(os.path.join(args.input_dir, 'synthetic_data.csv'))

    data = data_df['current'].values.flatten()
    times = data_df['time'].values.flatten()

    # Load in mcmc samples
    mcmc_samples = np.load(os.path.join(args.input_dir, 'mcmc_samples_all.npy'))

    # Find MLE
    protocol_func, tstart, tend, tstep, protocol_desc = common.get_ramp_protocol_from_csv('staircase')

    voltages = np.array([protocol_func(t) for t in times])

    spike_times, spike_indices = common.detect_spikes(times, voltages,
                                                      window_size=0)

    full_removal_durations = pd.read_csv(os.path.join(args.input_dir,
                                                      'removal_durations.csv')).values[:, 1].flatten()

    if args.removal_durations:

        def get_closest(lst, val):
            differences = np.abs(np.array(lst) - val)
            return lst[np.argmin(differences)]

        args.removal_durations = [int(r) for r in args.removal_durations]
        print(full_removal_durations)

        removal_durations = [(get_closest(full_removal_durations, r), i) for i, r in
                             enumerate(args.removal_durations)]

        removal_durations, indices_included = list(zip(*removal_durations))

        mcmc_samples = mcmc_samples[indices_included, :, :, :]

    if args.short:
        removal_durations = removal_durations[[0, -1]]

    # Use median of MCMC samples as initial parameters (will speed up optimisation)
    starting_parameters = [np.quantile(mcmc_samples[0, :, :, i].flatten(), 0.5)
                         for i in range(len(params))]

    def get_mle_cov(removal_duration):
        indices = common.remove_indices(list(range(len(times))), [(spike,
                                                                   int(spike + removal_duration /
                                                                       tstep)) for spike in
                                                                  spike_indices])

        model = BeattieModel(times=times, voltage=protocol_func, Erev=Erev, parameters=params)

        model.protocol_description = protocol_desc
        model.window_locs = [t for t, _, _, _ in protocol_desc]
        solver = model.make_hybrid_solver_current()
        mle, _ = common.fit_model(model, data, subset_indices=indices,
                                  solver=solver,
                                  max_iterations=args.max_iterations,
                                  repeats=args.repeats,
                                  starting_parameters=starting_parameters,
                                  method=optimiser)

        _, S1 = model.SimulateForwardModelSensitivities(params)
        S1 = S1[indices]
        H = S1.T @ S1
        H_inv = np.linalg.inv(H)
        cov = sigma2 * H_inv

        return mle, cov

    pool = pathos.multiprocessing.ProcessingPool(min(args.cpus, len(removal_durations) * args.repeats))
    mles, covs = list(zip(*pool.map(get_mle_cov, removal_durations)))

    fig = plt.figure(figsize=(14, 10))
    ax = fig.subplots()

    print('plotting')
    model = BeattieModel(times=times, voltage=protocol_func, Erev=Erev,
                         parameters=params, protocol_description=protocol_desc)

    fits_dir = os.path.join(output_dir, 'fits')
    if not os.path.exists(fits_dir):
        os.makedirs(fits_dir)

    # plot fits
    fits_fig, fits_ax = plt.subplots()
    for i in range(len(removal_durations)):
        fits_ax.plot(times, model.SimulateForwardModel(mles[i]), label='fitted model')
        fits_ax.plot(times, data, label='data')
        fits_ax.set_xlabel('time / ms')
        fits_ax.set_ylabel('current / nA')
        fits_fig.savefig(os.path.join(fits_dir, f"{removal_durations[i]}_removed.png"))

    fits_fig.close()

    # Loop over parameters, making a plot for each
    for i in range(model.get_no_parameters()):
        dfs = []
        param_label = model.parameter_labels[i]
        for j, removal_duration in enumerate(removal_durations):
            ax.cla()
            mean = mles[j][i]
            std = np.sqrt(covs[j][i, i])

            samples = mcmc_samples[j, :, :, i].flatten()

            # # Truncate to show only middle 80%
            # smin = np.quantile(samples, .1)
            # smax = np.quantile(samples, .9)

            # print(smin)
            # print(samples.shape)

            # samples = samples[np.argwhere((samples > smin) & (samples < smax))]
            print(samples.shape)

            no_gaussian_samples = 100000
            gaussian_samples = np.random.normal(mean, std, no_gaussian_samples)

            g_min, g_max = min(samples), max(samples)

            filtered_gaussian_samples = gaussian_samples[np.argwhere((gaussian_samples >
                                                             g_min) &
                                                            (gaussian_samples
                                                             < g_max))]

            if len(filtered_gaussian_samples) != 0:
                gaussian_samples = filtered_gaussian_samples

            hue = ['MCMC'] * samples.shape[0] + ['Gaussian'] * gaussian_samples.shape[0]
            all_samples = np.append(samples, gaussian_samples)
            print(all_samples.shape)

            df = pd.DataFrame(all_samples,
                              columns=('y',))

            df['hue'] = hue
            df['removal_duration'] = removal_duration
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)

        if 'Gaussian' not in df['hue'].values:
            df = pd.concat((df, pd.DataFrame(([np.nan], ['Gaussian'],
                                              [removal_durations[0]]))), ignore_index=True)

        print(df)

        try:
            sns.violinplot(data=df, ax=ax, x='removal_duration', y='y', hue='hue', split=True)
        except ValueError as e:
            print(str(e))

        ax.axhline(params[i], ls='--')
        fig.savefig(os.path.join(output_dir, "mcmc_comparison_%s.png" % param_label))


if __name__ == '__main__':
    main()
