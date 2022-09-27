#!/usr/bin/env python3

from MarkovModels import common
from MarkovModels.BeattieModel import BeattieModel
import os
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
import regex as re

figsize = (12, 8)

params = np.array([2.07E-3, 7.17E-2, 3.44E-5, 6.18E-2, 4.18E-1, 2.58E-2,
                   4.75E-2, 2.51E-2, 3.33E-2])

Erev = common.calculate_reversal_potential(310.15)

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
    print(df)

    df['spike_index'] = df['spike_index'].astype(np.float64)

    # chains_indices = df['chain'].unique()
    # no_chains = len(chains_indices)
    # no_spike_removals = len(df['spike_index'].unique())

    all_chains = []
    for spike_index in sorted(df['spike_index'].unique()):
        sub_df = df[df.spike_index == spike_index].sort_values('chain')

        fnames = sub_df['fname']
        print(fnames)
        chains = [pd.read_csv(os.path.join(chain_dir, fname)).values for fname in fnames]
        chains = np.stack(chains)
        if flatten_chains:
            chains = np.concatenate(chains)
        all_chains.append(chains)

    param_labels = pd.read_csv(os.path.join(chain_dir, fnames.iloc[0])).columns.tolist()

    return np.stack(all_chains), param_labels


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("chain_dir", help="Directory containing MCMC chains")
    parser.add_argument("-o", "--output", help="Directory to output to", default=None)
    parser.add_argument("-N", "--normalise", help="Divide through by true values",
                        action='store_true', default=False)

    args = parser.parse_args()

    output_dir = common.setup_output_directory(args.output,
                                               "plot_criteria_summary_plots")

    # Get chains
    all_chains, param_names = read_chains(args.chain_dir)
    all_chains = all_chains[:, :, 1:]
    param_names = param_names[1:]
    # all_chains[spike_indicies, sample, parameter]

    # Plot standard deviation of parameter estimates
    # standard_devs = np.log10(all_chains.std(axis=1)/params)
    log_RMSEs = np.log10(np.sqrt(np.mean((all_chains - params[None, :])**2, axis=1) / params[None, :]))

    removal_durations = pd.read_csv(os.path.join(args.chain_dir,
                                    'removal_durations.csv'))['removal_duration'].values.flatten()

    fig = plt.figure(figsize=figsize)
    ax = fig.subplots()
    ax.plot(removal_durations, log_RMSEs, label=param_names)
    ax.legend()

    ax.set_ylabel('RMSE / true value')
    ax.set_xlabel('time removed from each spike / ms')
    fig.savefig(os.path.join(output_dir, "param_summary.png"))

    # Zoomed in version
    ax.set_xlim([0,25])
    fig.savefig(os.path.join(output_dir, "param_summary_zoomed.png"))

    fig.clf()

    axs = fig.subplots(4)

    voltages = np.linspace(-120, 40, 25)
    all_vois = []
    for j, voltage in enumerate(voltages):
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=voltages[0], vmax=voltages[-1]))

        true_vals = compute_tau_inf(params[None, :], voltage).flatten()

        vois = np.empty((all_chains.shape[0], 4))

        for i in range(all_chains.shape[0]):
            voi = np.array(compute_tau_inf(all_chains[i, :, :], voltage))
            voi = np.sqrt(np.mean((voi - true_vals[:, None])**2, axis=1))
            vois[i, :] = voi

        vois = vois / true_vals[None, :] if args.normalise else vois

        all_vois.append(vois)

        voi_labels = ('a_inf', 'tau_a', 'r_inf', 'tau_r')
        units = ("", '/ ms', "", '/ ms')

        labels = []
        for i in range(4):
            axs[i].plot(removal_durations, np.log10(vois[:, i]), color=sm.to_rgba(voltage))
            labels.append(
                f"RMSE of {voi_labels[i]} estimate {units[i]}" if not args.normalise\
                else "normalised RMSE of %s estimate" % voi_labels[i]
            )

            axs[i].set_ylabel(labels[-1])

    axs[-1].set_xlabel('time removed from each spike / ms')
    fig.colorbar(sm, label='voltage / mV')
    fig.savefig(os.path.join(output_dir, "voi_plot.png"))

    for ax in axs:
        ax.set_xlim((0, 25))

    fig.savefig(os.path.join(output_dir, "voi_plot_zoomed.png"))

    # Now plot other way round
    all_vois = np.stack(all_vois, axis=-1)

    i_end = np.argmax(removal_durations > 25)
    for k, durations in enumerate([removal_durations, removal_durations[:i_end]]):
        fig.clf()
        axs = fig.subplots(4)
        # Cmap based on time removed

        if k == 0:
            cm = plt.cm.plasma
        elif k == 1:
            cm = plt.cm.cividis

        sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=0, vmax=durations[-1]))
        for i in range(4):
            axs[i].cla()
            axs[i].set_ylabel(labels[i])

        # Iterate over time removed
        for i in range(len(durations)):
            for j in range(4):
                time_removed = removal_durations[i]
                axs[j].plot(voltages, all_vois[i, j, :], color=sm.to_rgba(time_removed))

        axs[-1].set_xlabel('voltage / mV')

        fig.colorbar(sm, label='time removed / ms')
        if k == 0:
            fig.savefig(os.path.join(output_dir, "voltage_voi_plot.png"))
        elif k == 1:
            fig.savefig(os.path.join(output_dir, "voltage_voi_plot_zoomed.png"))

    model = BeattieModel(parameters=params)

    # Plot true IV curve
    analytic_solver = model.get_analytic_solver()
    rhs0_func = model.rhs_inf

    IV_voltages = np.linspace(-120, -40, 50)
    voi_voltages = np.linspace(-120, 40, 100)

    true_peak_Is = get_peak_currents(params, IV_voltages, analytic_solver, rhs0_func, model)

    for i in range(all_chains.shape[0]):
        fig.clf()
        ax = fig.subplots()

        peak_Is_list = []

        n_samples = 1000
        sample_indices = np.random.randint(0, all_chains.shape[-2], size=n_samples)

        for j in range(n_samples):
            sample_index = sample_indices[j]
            param_sample = all_chains[i, sample_index, :]
            peak_Is = get_peak_currents(param_sample, IV_voltages, analytic_solver, rhs0_func, model)

            peak_Is_list.append(peak_Is)

            label = "samples" if j == 0 else None
            ax.plot(IV_voltages, peak_Is - true_peak_Is, color='grey', alpha=0.1, label=label)

        ax.set_xlabel('voltage / mV')
        ax.set_ylabel('error in peak current / nA')
        ax.set_title(f"Sampled IV errors with {removal_durations[i]:.2f}ms removed")
        ax.legend()
        fig.savefig(os.path.join(output_dir, f"I_V_plot_errors_plot_{i}.png"))

        fig.clf()
        ax = fig.subplots()

        for peak_Is in peak_Is_list:
            label = "samples" if j == 0 else None
            ax.plot(IV_voltages, peak_Is, color='grey', alpha=0.1, label=label)
            ax.set_xlabel('voltage / mV')
            ax.set_ylabel('peak current / nA')
            ax.set_title(f"Sampled I-V curve with {removal_durations[i]:.2f}ms removed")

        ax.plot(IV_voltages, true_peak_Is, label='true peak current')

        ax.legend()
        fig.savefig(os.path.join(output_dir, f"I_V_plot_{i}.png"))

        fig.clf()
        axs = fig.subplots(4)

        voi_labels = ['a_inf', 'tau_a', 'r_inf', 'tau_r']

        voi_units = ['', ' / ms', '', ' / ms']

        for l in range(4):
            for m in range(n_samples):
                p = all_chains[i, sample_indices[m], :]
                vois = get_voi_curves(p, voi_voltages)
                label = "samples" if m == 0 else None
                axs[l].plot(voi_voltages, vois[l], color='grey', alpha=0.1, label=label)

            axs[l].set_xlabel('voltage / mV')
            axs[l].set_ylabel(voi_labels[l] + voi_units[l])

            true_voi = get_voi_curves(params, voi_voltages)

            axs[l].plot(voi_voltages, true_voi[l], label='true %s' % voi_labels[l])
            axs[l].legend()

        axs[0].set_title(f"timescale and steady state curve samples with {removal_durations[i]:.2f}ms removed")
        fig.savefig(os.path.join(output_dir, f"VOI_plot_{i}.png"))



def get_peak_currents(p, voltages, analytic_solver, rhs0_func, model):
    peaks = np.full(voltages.shape, np.nan)
    for i, voltage in enumerate(voltages):
        # Get equilibrium at 40mV
        rhs0 = rhs0_func(params, 40)

        # solve forwards for 250ms (should be plenty)
        times = np.linspace(0, 250, 1000)

        sol = analytic_solver(times, voltage, p, rhs0.flatten())
        current = sol[:, model.open_state_index] * p[-1] * (voltage - Erev)

        peak = current[np.argmax(np.abs(current))]

        peaks[i] = peak

    return peaks


def get_voi_curves(p, voltages):
    a_inf, tau_a, r_inf, tau_r = np.column_stack([compute_tau_inf(p[None, :], v) for v in voltages])

    return a_inf, tau_a, r_inf, tau_r


if __name__ == "__main__":
    main()
