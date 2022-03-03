#!/usr/bin/env python3

from MarkovModels import common
import os
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
import regex as re

params = np.array([2.07E-3, 7.17E-2, 3.44E-5, 6.18E-2, 4.18E-1, 2.58E-2,
                   4.75E-2, 2.51E-2, 3.33E-2])

def compute_tau_inf(samples, voltage):
    k1 = samples[:, 0] * np.exp(samples[:,  1] * voltage)
    k2 = samples[:,  2] * np.exp(-samples[:,  3] * voltage)
    k3 = samples[:,  4] * np.exp(samples[:,  5] * voltage)
    k4 = samples[:,  6] * np.exp(-samples[:,  7] * voltage)

    a_inf = k1 / (k1 + k2)
    tau_a = 1 / (k1 + k2)

    r_inf = k4 / (k3 + k4)
    tau_r = 1 / (k3 + k4)

    return np.array((a_inf, tau_a, r_inf, tau_r))


def read_chains(chain_dir, flatten_chains=True):
    # Get all chain files
    regexpr = re.compile(r'^mcmc_samples_\[([0-9]+)\]_chain_([0-9]+)\.csv$')
    files = list(filter(regexpr.match, os.listdir(chain_dir)))

    attr = []
    for f in files:
        spike_index = re.search(regexpr, f).groups()[0]
        chain = re.search(regexpr, f).groups()[1]

        attr.append((spike_index, chain, f))

    df = pd.DataFrame(attr, columns=('spike_index', 'chain', 'fname'))

    # chains_indices = df['chain'].unique()
    # no_chains = len(chains_indices)
    # no_spike_removals = len(df['spike_index'].unique())

    all_chains = []
    for spike_index in sorted(df['spike_index'].unique()):
        sub_df = df[df.spike_index == spike_index].sort_values('chain')

        fnames = sub_df['fname']
        chains = np.array([pd.read_csv(os.path.join(chain_dir, fname)).values for fname in fnames])
        if flatten_chains:
            chains = np.concatenate(chains)
        all_chains.append(chains)

    param_labels = pd.read_csv(os.path.join(chain_dir, fnames.iloc[0])).columns.tolist()

    return np.array(all_chains), param_labels


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("chain_dir", help="Directory containing MCMC chains")
    parser.add_argument("-o", "--output", help="Directory to output to", default=None)

    args = parser.parse_args()

    output_dir = common.setup_output_directory(args.output,
                                               "plot_criteria_standard_errors")

    # Get chains
    all_chains, param_names = read_chains(args.chain_dir)
    all_chains = all_chains[:, :, 1:]
    param_names = param_names[1:]
    # all_chains[spike_indicies, sample, parameter]

    # Plot standard errors of parameters
    standard_errors = np.log10(all_chains.std(axis=1)/params)

    removal_durations = pd.read_csv(os.path.join(args.chain_dir),
                                    'removal_durations.csv')['removal_durations'].values.flatten()

    fig = plt.figure(figsize=(16, 12))
    ax = fig.subplots()
    ax.plot(removal_durations, standard_errors, label=param_names)
    ax.legend()
    fig.savefig(os.path.join(output_dir, "param_standard_errors.png"))

    fig.clf()

    axs = fig.subplots(4)

    voltages = np.linspace(-120, 40, 25)
    for j, voltage in enumerate(voltages):
        col = plt.cm.jet(j/voltages.shape[0])

        true_vals = compute_tau_inf(params[None, :], voltage)

        vois = np.empty((all_chains.shape[0], 4))

        for i in range(all_chains.shape[0]):
            voi = np.array(compute_tau_inf(all_chains[i, :, :], voltage))
            voi = voi.std(axis=1)/true_vals.flatten()
            vois[i, :] = voi

        voi_labels = ('a_inf', 'tau_a', 'r_inf', 'tau_r')

        for i in range(4):
            axs[i].plot(removal_durations, np.log10(vois[:, i]), color=col)
            axs[i].set_ylabel(removal_durations, voi_labels[i])

    fig.savefig(os.path.join(output_dir, "voi_plot.png"))


if __name__ == "__main__":
    main()
