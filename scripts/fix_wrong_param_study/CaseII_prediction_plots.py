#!/usr/bin/env python3

import multiprocessing
import regex as re
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from markovmodels import common
from markovmodels.BeattieModel import BeattieModel
from markovmodels.ClosedOpenModel import ClosedOpenModel
from markovmodels.KempModel import KempModel
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

from matplotlib.patches import ConnectionPatch, Rectangle

import matplotlib.lines as mlines

from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 8})
rc('text', usetex=True)
rc('figure', dpi=500, facecolor=[0]*4)
rc('axes', facecolor=[0]*4)
rc('savefig', facecolor=[0]*4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('results_dirs', nargs='+')
    parser.add_argument('--repeats', type=int, default=16)
    parser.add_argument('--wells', '-w', type=str, default=[], nargs='+')
    parser.add_argument('--experiment_name', default='newtonrun4', type=str)
    parser.add_argument('--no_chains', '-N', default=0, help='mcmc chains to run', type=int)
    parser.add_argument('--chain_length', '-l', default=500, help='mcmc chains to run', type=int)
    parser.add_argument('--figsize', '-f', nargs=2, default=[4.685, 6.5])
    parser.add_argument('--use_parameter_file')
    parser.add_argument('-i', '--ignore_protocols', nargs='+',
                        default=['longap'])

    parser.add_argument('-o', '--output_dir')
    parser.add_argument("-F", "--file_format", default='pdf')
    parser.add_argument("-m", "--model_class", nargs=2, default=['Beattie',
                                                                 'Wang'])
    parser.add_argument('--true_param_file')
    parser.add_argument('--fixed_param', default='Gkr')
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

    output_dir = common.setup_output_directory(args.output_dir, "CaseII_predictions")

    global fig
    fig = plt.figure(figsize=args.figsize, constrained_layout=True)

    global protocols
    protocols = sorted(pd.read_csv(os.path.join(args.results_dirs[0], 'fitting.csv')).protocol.unique())

    relabel_dict = {p: f"$d_{i+1}$" for i, p in enumerate([p for p in protocols if p not in args.ignore_protocols and p != 'longap'])}
    relabel_dict['longap'] = '$d_0$'

    global results_dfs
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

    global markers
    markers = [None] * 100
    markers = [markers[i] for i in range(len(results_df.protocol.unique()))]

    global colours
    colours = [palette[i] for i in range(len(results_df.protocol.unique()))]

    protocols = sorted(results_df.protocol.unique())
    protocols = results_dfs[0].protocol.unique()

    axes = create_axes(fig)

    current = pd.read_csv(os.path.join(args.data_dir,
                                               f"synthetic-{args.prediction_protocol}-1.csv"))['current'].values.flatten().astype(np.float64)
    times = pd.read_csv(os.path.join(args.data_dir,
                                            f"synthetic-{args.prediction_protocol}-times.csv"))['time'].values.flatten().astype(np.float64)


    do_prediction_plots(axes, results_dfs, args.prediction_protocol, current, times)

    fig.savefig(os.path.join(output_dir, f"Fig6.{args.file_format}"))


def do_prediction_plots(axes, results_dfs, prediction_protocol, current, times):
    voltage_func, times, protocol_desc = common.get_ramp_protocol_from_csv(prediction_protocol)

    voltages = np.array([voltage_func(t) for t in times])
    # _, spike_indices = common.detect_spikes(times, voltages, window_size=0)

    colno = 1
    prediction_axes = axes[2:]

    training_protocols = sorted(results_dfs[0].protocol.unique())

    # filter out ignored protocols
    training_protocols = [p for p in training_protocols if p not in args.ignore_protocols]

    colours = [palette[i] for i in range(len(protocols))]

    print(linestyles)

    ymin, ymax = [0, 0]

    for i in range(len(results_dfs)):
        predictions = []
        ax = prediction_axes[i]

        ax.plot(times, current, color='grey', alpha=.5, lw=0.3)

        for training_protocol in sorted(training_protocols):
            model_class = common.get_model_class(args.model_class[i])
            parameter_labels = model_class().get_parameter_labels()
            model = model_class(voltage_func, times, protocol_description=protocol_desc)
            solver = model.make_forward_solver_current()

            results_df = results_dfs[i]
            row = results_df[(results_df.protocol == training_protocol)]
            parameters = row[parameter_labels].head(1).values.flatten()
            prediction = solver(parameters)
            predictions.append(prediction)
            # ax.plot(prediction, times, lw=0.1)
            ymin = min(ymin, prediction.min())
            ymax = max(ymax, prediction.max())

        predictions = np.array(predictions)

        max_pred = predictions.max(axis=0)
        min_pred = predictions.min(axis=0)
        ax.plot(times, max_pred, '--', color='red',
                lw=.3, )
        ax.plot(times, min_pred, '--', color='red',
                lw=.3, )

        ax.fill_between(times, min_pred, max_pred, color='orange', alpha=0.25,
                        lw=0, )

        axins = inset_axes(ax, width='50%', height='50%', loc='lower center')
        axins2 = inset_axes(ax, width='15%', height='40%', loc='lower left')

        axins.plot(times, current, color='grey', alpha=.3, lw=0.3)
        axins2.plot(times, current, color='grey', alpha=.3, lw=0.3)

        # axins.axis('off')
        axins.set_xticks([])
        axins.set_yticks([])

        axins2.set_xticks([])
        axins2.set_yticks([])

        axins.fill_between(times, min_pred, max_pred, color='orange', alpha=.2,
                           lw=0)

        axins2.fill_between(times, min_pred, max_pred, color='orange', alpha=.2,
                            lw=0)

        for j in range(predictions.shape[0]):
            linestyle = linestyles[j]
            prediction = predictions[j, :]
            axins.plot(times, prediction, ls=linestyle,
                       lw=0.5, color=colours[j])

            axins2.plot(times, prediction, ls=linestyle,
                        lw=0.5, color=colours[j])

        axins.set_xlim([3250, 6000])
        axins.set_ylim(-0.5, 2.25)

        axins2.set_xlim([750, 1250])
        axins2.set_ylim(-0.65, 1)

        mark_inset(ax, axins, edgecolor="black", fc="none", loc1=1, loc2=2,
                   lw=.3, alpha=.8)

        mark_inset(ax, axins2, edgecolor="black", fc="none", loc1=1, loc2=2,
                   lw=.3, alpha=.8)

    axes[1].set_xlim([0, 9000])
    for i, ax in enumerate(prediction_axes):
        ax.set_xlim(ymin, max(ymax, np.quantile(current, 0.9)))
        # ax.yaxis.tick_right()

        ax.set_xlim([0, 9000])
        ax.set_xticks([0, 8000])
        ax.set_xticklabels(['0', '8'], rotation='horizontal')

        ylims = [-7.5,  3]
        ax.set_yticks([-7.5, 0, 3])
        ax.set_ylim(ylims)

        ax.set_yticks(ylims)

        yticklabs = ax.get_yticklabels()
        ax.set_yticklabels(yticklabs, rotation='horizontal')
        ax.set_ylim()

        # remove spines
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)

        ax.set_rasterization_zorder(2)
        ax.set_ylabel(r'$I_\textrm{Kr}$ (nA)')

    # Plot voltage
    axes[1].plot(times[::50], [voltage_func(t) for t in times][::50],
                     color='black', lw=.5)

    prediction_axes[1].set_xlabel(r'$t$ (s)')

    # axes[colno].yaxis.tick_right()
    labels = ['0', '7.5']
    axes[colno].spines.right.set_visible(False)
    axes[colno].spines.top.set_visible(False)

    axes[colno].set_xticks([])

    for ax in prediction_axes[:-1]:
        ax.set_xticks([])

    prediction_axes[1].set_xticks([0, 7500])
    prediction_axes[1].set_xticklabels(labels)

    prediction_axes[-1].set_visible(False)

    axes[colno].set_yticks([-100, 40])
    axes[colno].set_ylabel(r'$V$ (mV)')
    # axes[colno].set_yticklabels(['-100', '40'])


def create_axes(fig):
    global gs
    nrows = 5
    ncols = 1

    gs = GridSpec(nrows, ncols, height_ratios=[0.15, 0.3, 1, 1, 1],
                  figure=fig)

    axes = [fig.add_subplot(cell) for cell in gs]

    axes[1].set_title(r'\textbf{a}', loc='left')
    axes[2].set_title(r'\textbf{b}', loc='left')
    axes[3].set_title(r'\textbf{c}', loc='left')
    axes[4].set_title(r'\textbf{d}', loc='left')

    # Put legend on the top left axis
    ax = axes[0]
    legend_kws = {'loc': 10,
                  'frameon': False,
                  'bbox_to_anchor': [0, 0, 1, 1],
                  'ncol': 2,
                  'fontsize': 8
                  }

    handles = [mlines.Line2D(xdata=[1], ydata=[1], color=color, marker=marker,
                             linestyle=linestyles[i], markersize=5,
                             label=label, linewidth=1) for i, (label, marker,
                                                                color) in enumerate(zip(protocols, markers,
                                                                                        colours))]

    handles, labels = list(handles), list(results_dfs[0]['protocol'].unique())
    ax.legend(labels=labels, handles=handles, **legend_kws)
    ax.axis('off')

    return axes


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
