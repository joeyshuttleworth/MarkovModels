#!/usr/bin/env python3

import multiprocessing
import regex as re
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from MarkovModels import common
from MarkovModels.BeattieModel import BeattieModel
from MarkovModels.ClosedOpenModel import ClosedOpenModel
from MarkovModels.KempModel import KempModel
import argparse
import seaborn as sns
import os
import string
import re
import scipy
import math

import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes
from matplotlib.gridspec import GridSpec

import matplotlib.lines as mlines

from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 8})
rc('text', usetex=True)
rc('figure', dpi=300)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('results_dirs', nargs='+')
    parser.add_argument('--repeats', type=int, default=16)
    parser.add_argument('--wells', '-w', type=str, default=[], nargs='+')
    parser.add_argument('--removal_duration', '-r', default=0, type=float)
    parser.add_argument('--experiment_name', default='newtonrun4', type=str)
    parser.add_argument('--no_chains', '-N', default=0, help='mcmc chains to run', type=int)
    parser.add_argument('--chain_length', '-l', default=500, help='mcmc chains to run', type=int)
    parser.add_argument('--figsize', '-f', type=int, nargs=2, default=[4.6, 4.6])
    parser.add_argument('--use_parameter_file')
    parser.add_argument('-i', '--ignore_protocols', nargs='+',
                        default=['longap'])

    parser.add_argument('-o', '--output_dir')
    parser.add_argument("-F", "--file_format", default='pdf')
    parser.add_argument("-m", "--model_class", default='Beattie')
    parser.add_argument('--prediction_protocol', default='longap')

    parser.add_argument("--vlim", nargs=2, type=float)

    global linestyles
    linestyles = [(0, ()),
      (0, (1, 2)),
      (0, (1, 1)),
      (0, (5, 5)),
      (0, (3, 5, 1, 5)),
      (0, (3, 5, 1, 5, 1, 5))]

    global args
    args = parser.parse_args()

    model_class = common.get_model_class(args.model_class)

    parameter_labels = model_class().get_parameter_labels()

    global true_parameters
    true_parameters = model_class().get_default_parameters()

    output_dir = common.setup_output_directory(args.output_dir, "error_compare")

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    axes, scatter_axes = create_axes(fig)

    global protocols
    protocols = pd.read_csv(os.path.join(args.results_dirs[0], 'fitting.csv')).protocol.unique()

    global relabel_dict
    relabel_dict = {p: f"$d_{i+1}$" for i, p in enumerate(protocols)}
    relabel_dict['longap'] = '$d^*$'

    results_dfs = []
    for results_dir in args.results_dirs:
        results_df = pd.read_csv(os.path.join(results_dir, 'fitting.csv'))
        results_df['protocol'] = results_df.protocol
        results_df = results_df[~results_df.protocol.isin(args.ignore_protocols)]
        results_df.replace({'protocol': relabel_dict}, inplace=True)
        results_df = results_df.sort_values('protocol')

        results_df = get_best_params(results_df)

        if 'g_Kr' in results_df:
            results_df['Gkr'] = results_df['g_Kr']

        results_dfs.append(results_df)

    global palette
    palette = sns.color_palette('husl', len(results_df.protocol.unique()))

    current = pd.read_csv(os.path.join(args.data_dir,
                                   f"synthetic-{args.prediction_protocol}-0.csv"))['current'].values.flatten().astype(np.float64)
    times = pd.read_csv(os.path.join(args.data_dir,
                                     f"synthetic-{args.prediction_protocol}-times.csv"))['time'].values.flatten().astype(np.float64)

    # do_barplot(axes, results_dfs[0], args.prediction_protocol)

    do_interval_vs_error_plot(axes, scatter_axes, results_dfs[0],
                              args.prediction_protocol, current, times)

    fig.savefig(os.path.join(output_dir, f"figure.{args.file_format}"))


def do_interval_vs_error_plot(axes, scatter_ax, results_df,
                              prediction_protocol, current, times):
    voltage_func, _, protocol_desc = common.get_ramp_protocol_from_csv(prediction_protocol)
    print(protocol_desc)

    voltages = np.array([voltage_func(t) for t in times])
    spike_times, _ = common.detect_spikes(times, voltages, window_size=0)
    print(spike_times)

    if args.removal_duration:
        _, _, indices = common.remove_spikes(times, voltages, spike_times,
                                             time_to_remove=args.removal_duration)
    else:
        indices = np.array(range(times.shape[0]))

    colno = 1
    prediction_axes = [axes[i] for i in range(len(axes)) if (i % 3) == colno
                       and i > 2]

    training_protocols = results_df.protocol.unique()

    # filter out ignored protocols
    training_protocols = [p for p in training_protocols if p not in args.ignore_protocols]

    model_class = common.get_model_class(args.model_class)

    model_names = ['Beattie', 'Wang']
    ymin, ymax = [np.inf, -np.inf]

    model_class = common.get_model_class('Beattie')
    parameter_labels = model_class().get_parameter_labels()

    model = model_class(voltage_func, times=times,
                        protocol_description=protocol_desc)
    solver = model.make_forward_solver_current()

    predictions = []
    for training_protocol in sorted(training_protocols):

        results_df['score'] = results_df['score'].astype(np.float64)
        row = results_df[(results_df.protocol == training_protocol)
                            & (results_df.well.astype(int) == 1)].sort_values('score')
        parameters = row[parameter_labels].head(1).values.flatten().astype(np.float64)
        print('parameters are', parameters)

        prediction = solver(parameters)
        predictions.append(prediction)
        # ax.plot(prediction, times, linewidth=0.1)

        ymin = min(ymin, prediction.min())
        ymax = max(ymax, prediction.max())

    predictions = np.vstack(predictions)

    max_pred = predictions.max(axis=0)
    min_pred = predictions.min(axis=0)
    mid_pred = (max_pred + min_pred)/2

    interval_width = max_pred[indices] - min_pred[indices]

    true_model = common.get_model_class('Wang')(voltage_func, times=times,
                                                protocol_description=protocol_desc)

    truth = true_model.make_forward_solver_current()()
    errors = np.abs(predictions - truth[None, :])[:, indices]

    midpoint_errors = (mid_pred - truth)[indices]

    interval_error = np.min(errors, axis=0)
    bound_correct_args = np.argwhere((truth[indices] < np.max(errors, axis=0)) & (truth[indices] > np.min(errors, axis=0)))

    interval_error[bound_correct_args] = 0

    print(truth.shape, times.shape, max_pred.shape)

    axes[2].fill_between(times[indices], min_pred[indices] - truth[indices],
                         max_pred[indices] - truth[indices], lw=.5,
                         color='orange', alpha=.5)
    axes[2].axhline(0, lw=.1, ls='--', color='grey')
    # axes[1].set_ylim([axes[0].get_ylim()[0], 2])

    axes[1].plot(times[indices], truth[indices], lw=.5)
    axes[1].fill_between(times[indices], min_pred[indices],
                         max_pred[indices], lw=.5,
                         color='orange', alpha=.5)

    axes[0].set_ylabel('V / mV')
    axes[2].set_ylabel(r'$I_\textrm{Kr}$ / nA')
    axes[1].set_ylabel(r'$I_\textrm{Kr}$ / nA')
    axes[2].set_xlabel(r't / ms')

    palette = sns.color_palette('cubehelix', as_cmap=True)
    scatter_ax[0].scatter(midpoint_errors, interval_width, cmap=palette,
                          c=range(interval_width.shape[0]), s=.2, marker='.')

    xlims = scatter_ax[0].get_xlim()
    ylims = scatter_ax[0].get_ylim()

    x_range = np.linspace(*xlims, 1000)
    x_range = np.linspace(*xlims, 1000)

    x_pos = x_range[np.argwhere(x_range > 0)]
    scatter_ax[0].plot(-x_pos, 2*np.abs(x_pos), color='red', lw=1, linestyle='--')
    scatter_ax[0].plot(x_pos, 2*np.abs(x_pos), color='red', lw=1, linestyle='--')

    # scatter_ax[0].set_xscale('log')
    # scatter_ax[0].set_yscale('log')

    scatter_ax[1].plot(times, truth, '--', lw=.2, color='black', alpha=.5)
    scatter_ax[1].scatter(times[indices], truth[indices], c=range(interval_width.shape[0]),
                          cmap=palette, marker='.', s=.01, zorder=3)

    scatter_ax[0].set_ylabel(r'$\mathcal{B}_\textrm{upper} - \mathcal{B}_\textrm{lower}$')
    scatter_ax[0].set_xlabel(r'$\mathbf{y}(\mathbf \theta^*; d^*) - \mathcal{B}_\textrm{mid}$')

    scatter_ax[1].set_ylabel(r'$I_{\textrm{Kr}}$ / mV')
    scatter_ax[1].set_xlabel('t / ms')

    axes[0].scatter(times, voltages, lw=.2, cmap='cubehelix',
                    c=range(times.shape[0]), s=1, marker='.')
    axes[0].plot(times, voltages, '--', lw=.2, color='black', alpha=.5)


def get_protocol_name(label):
    for k, v in relabel_dict.items():
        if v == label:
            return k


def create_axes(fig):
    gs = GridSpec(3, 2, figure=fig,
                  height_ratios=[.6, 1, 1])

    axes = [fig.add_subplot(gs[i, 1]) for i in range(3)]

    for ax in axes[:-1]:
        ax.set_xticks([])

    axes[0].set_title(r'\textbf c', loc='left', )
    axes[1].set_title(r'\textbf d', loc='left', )
    axes[2].set_title(r'\textbf e', loc='left', )

    scatter_axes = [fig.add_subplot(gs[1:, 0]),
                    fig.add_subplot(gs[0, 0])]

    scatter_axes[1].set_title(r'\textbf a', loc='left', )
    scatter_axes[0].set_title(r'\textbf b', loc='left', )

    for ax in list(axes) + list(scatter_axes):
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)

    for ax in axes[:-1]:
        ax.spines['bottom'].set_visible(False)

    # box = axes[0].get_position()
    # # box.y0 -= 0.025
    # axes[0].set_position(box)

    return axes, scatter_axes


def get_best_params(fitting_df, protocol_label='protocol'):
    best_params = []

    print(fitting_df)
    fitting_df['score'] = fitting_df['score'].astype(np.float64)
    fitting_df = fitting_df[np.isfinite(fitting_df['score'])].copy()

    for protocol in fitting_df[protocol_label].unique():
        for well in fitting_df['well'].unique():
            sub_df = fitting_df[(fitting_df['well'] == well)
                                & (fitting_df[protocol_label] == protocol)].copy()

            # Get index of min score
            if len(sub_df.index) == 0:
                continue
            best_params.append(sub_df[sub_df.score == sub_df.score.min()].head(1).copy())

    return pd.concat(best_params, ignore_index=True)


if __name__ == "__main__":
    main()
