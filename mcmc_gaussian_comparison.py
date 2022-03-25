#!/usr/bin/env python3

from MarkovModels import common
import logging
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

    parser.add_argument('input_directory', type=str, help="path to the directory containing the plot_criteria results")
    parser.add_argument('--output_dir', '-o', type=str, help="output path")

    args = parser.parse_args()

    output_dir = common.setup_output_directory(args.output_dir, 'mcmc_gaussian_comparison')

    # Load in synthetic data
    data_df = pd.read_csv(os.path.join(args.input_dir, 'synthetic_data.csv'))

    data = data_df['current'].values.flatten()
    times = data_df['current'].values.flatten()

    # Load in mcmc samples
    mcmc_samples = np.load(os.path.join(args.input_dir, 'mcmc_samples_all.npy'))

    # Find MLE
    protocol_func, tstart, tend, tstep, protocol_desc = common.get_ramp_protocol_from_csv('staircase')

    voltages = np.array([protocol_func(t) for t in times])

    spike_times, spike_indices = common.detect_spikes(times, voltages,
                                                      window_size=0)

    model = BeattieModel(times=times, voltage=protocol_func, Erev=Erev, parameters=params)

    model.protocol_description = protocol_desc
    model.window_locs = [t for t, _, _, _ in protocol_desc]

    solver = model.make_hybrid_solver_current()

    covs, mles = [], []

    # Compute parameter covariariance matrices
    removal_durations = pd.read_csv(os.path.join(args.input_dir, 'removal_durations.csv')).values.flatten()

    for removal_duration in removal_durations:
        _, _, indices = common.remove_spikes(times, voltages, spike_times, removal_duration)
        mle, _ = common.fit_model(model, data,
                                  subset_indices=indices,
                                  solver=solver,
                                  max_iterations=args.max_iterations)

        _, S1 = model.SimulateForwardModelSensitivities(params)
        H = S1.T @ S1
        H_inv = np.linalg.inv(H)
        cov = sigma2 * H_inv

        covs.append(cov)
        mles.append(mle)

        fig = plt.figure(figsize=(14, 10))
        ax = fig.subplots()

    # Loop over parameters, making a plot for each
    for i in range(model.get_no_parameters()):
        dfs = []
        param_label = model.parameter_labels[i]
        for removal_duration in removal_durations:
            ax.cla()
            means = [mle[i] for mle in mles]
            stds = [cov[i, i] for cov in covs]

            samples = [chains.flatten() for chains in mcmc_samples]
            no_gaussian_samples = samples[0].shape[0]

            gaussian_samples = [np.random.normal(mean, std, no_gaussian_samples)
                                for mean, std in zip(means, stds)]

            hue = ['MCMC'] * no_gaussian_samples + ['Gaussian'] * no_gaussian_samples

            df = pd.DataFrame(np.concatenate((samples, gaussian_samples)),
                              columns=(param_label,))

            df['hue'] = hue
            df['removal_duration'] = removal_duration
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)
        sns.violinplot(df, ax=ax, x='removal_duration', y=param_label, hue='hue')
        fig.savefig(os.path.join(output_dir, "mcmc_comparison_%s.png" % param_label))

if __name__ == '__main__':
    main()
