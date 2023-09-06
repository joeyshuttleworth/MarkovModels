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
    parser.add_argument('results_dir')
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
    parser.add_argument("-m", "--model_class", default='Beattie')
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

    model_class = common.get_model_class(args.model_class)

    if args.true_param_file:
        assert(False)
    else:
        parameter_labels = model_class().get_parameter_labels()

    global true_parameters
    true_parameters = model_class().get_default_parameters()

    output_dir = common.setup_output_directory(args.output_dir, "CaseI_predictions")

    global fig
    fig = plt.figure(figsize=args.figsize, constrained_layout=True)

    global results_df
    results_df = pd.read_csv(os.path.join(args.results_dir, 'results_df.csv'))

    results_df = results_df[~results_df.protocol.isin(args.ignore_protocols)]

    global palette
    palette = sns.color_palette('husl', len(results_df.protocol.unique()))

    global markers
    markers = [None] * 100
    markers = [markers[i] for i in range(len(results_df.protocol.unique()))]

    global colours
    colours = [palette[i] for i in range(len(results_df.protocol.unique()))]

    global protocols
    protocols = sorted(results_df.protocol.unique())
    relabel_dict = {p: f"$d_{i+1}$" for i, p in enumerate([p for p in protocols if p not in args.ignore_protocols and p != 'longap'])}
    relabel_dict['longap'] = '$d_0$'
    results_df.replace({'protocol': relabel_dict}, inplace=True)
    results_df = results_df.sort_values('protocol')

    print(relabel_dict)

    protocols = results_df.protocol.unique()

    axes = create_axes(fig)

    for param_label in parameter_labels:
        results_df[param_label] = results_df[param_label].astype(np.float64)

    # Plot heatmaps
    prediction_df = pd.read_csv(os.path.join(args.results_dir, 'predictions.csv'))
    prediction_df.replace({'fitting_protocol': relabel_dict,
                           'validation_protocol': relabel_dict},
                          inplace=True)

    for lab in parameter_labels:
        prediction_df[lab] = prediction_df[lab].astype(np.float64)

    keep_rows = ~prediction_df.fitting_protocol.isin(args.ignore_protocols) &\
        prediction_df.validation_protocol.isin(relabel_dict.values())

    prediction_df = prediction_df[keep_rows]

    data = pd.read_csv(os.path.join(args.results_dir,
                                    f"synthetic_data_{args.prediction_protocol}_0.csv"))

    do_prediction_plots(axes, results_df, args.prediction_protocol, data)
    axes[1].set_title(r'\textbf{a}', loc='left')
    axes[2].set_title(r'\textbf{b}', loc='left')
    axes[3].set_title(r'\textbf{c}', loc='left')
    axes[4].set_title(r'\textbf{d}', loc='left')


    fig.savefig(os.path.join(output_dir, f"Fig4.{args.file_format}"))


def do_prediction_plots(axes, results_df, prediction_protocol, data):
    times = data['time / ms'].astype(np.float64).values
    current = data['current / nA'].astype(np.float64).values

    print(times, current)

    vals = sorted(results_df[args.fixed_param].unique())[::2]

    voltage_func, times, protocol_desc = common.get_ramp_protocol_from_csv(prediction_protocol)

    voltages = np.array([voltage_func(t) for t in times])
    # _, spike_indices = common.detect_spikes(times, voltages, window_size=0)

    colno = 1
    prediction_axes = axes[2:]

    training_protocols = sorted(results_df.protocol.unique())

    # filter out ignored protocols
    training_protocols = [p for p in training_protocols if p not in args.ignore_protocols]

    model_class = common.get_model_class(args.model_class)
    parameter_labels = model_class().get_parameter_labels()

    model = model_class(voltage_func, times, protocol_description=protocol_desc)
    solver = model.make_forward_solver_current()

    colours = [palette[i] for i in range(len(protocols))]

    print(linestyles)

    ymin, ymax = [0, 0]
    for i in range(len(prediction_axes)):
        # plot data
        ax = prediction_axes[i]

        ax.plot(times, current, color='grey', alpha=.5, lw=0.3)
        val = vals[i]

        predictions = []
        for training_protocol in sorted(training_protocols):

            row = results_df[(results_df.protocol == training_protocol) &
                             (results_df[args.fixed_param] == val)]
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

        # axins.axis('off')
        axins.set_xticks([])
        axins.set_yticks([])

        axins2.set_xticks([])
        axins2.set_yticks([])

        axins.fill_between(times, min_pred, max_pred, color='orange', alpha=.2,
                           lw=0)
        axins2.fill_between(times, min_pred, max_pred, color='orange', alpha=.2,
                            lw=0)

        axins.plot(times, current, color='grey', alpha=.3, lw=0.3)
        axins2.plot(times, current, color='grey', alpha=.3, lw=0.3)

        for j in range(predictions.shape[0]):
            linestyle = linestyles[j]
            prediction = predictions[j, :]

            axins.plot(times, prediction, ls=linestyle,
                       lw=0.5, color=colours[j])

            axins2.plot(times, prediction, ls=linestyle,
                        lw=0.5, color=colours[j])

        axins.set_xlim([3250, 6000])
        axins.set_ylim(-0.15, 0.65)

        axins2.set_xlim([750, 2050])
        axins2.set_ylim(-0.1, .45)

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

        yticks = [0, -2]
        ax.set_yticks(yticks)

        yticklabs = ax.get_yticklabels()
        ax.set_yticklabels(yticklabs, rotation='horizontal')
        ax.set_ylim([-2, 1])

        # remove spines
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)

        ax.set_rasterization_zorder(2)

        ax.set_ylabel(r'$I_\textrm{Kr}$ (nA)')

    # Plot voltage
    axes[1].plot(times[::50], [voltage_func(t) for t in times][::50],
                 color='black', lw=.5)

    axes[1].set_xlim([0, 9000])

    prediction_axes[-1].set_xlabel(r'$t$ (s)')

    # axes[colno].yaxis.tick_right()
    labels = ['0', '7.5']
    axes[colno].spines.right.set_visible(False)
    axes[colno].spines.top.set_visible(False)

    axes[colno].set_xticks([])

    for ax in prediction_axes[:-1]:
        ax.set_xticks([])

    prediction_axes[-1].set_xticks([0, 7500])
    prediction_axes[-1].set_xticklabels(labels)

    axes[colno].set_yticks([-100, 40])
    axes[colno].set_ylabel(r'$V$ (mV)')
    # axes[colno].set_yticklabels(['-100mV', '+40mV'])


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

    handles, labels = list(handles), list(results_df['protocol'].unique())
    ax.legend(labels=labels, handles=handles, **legend_kws)
    ax.axis('off')

    return axes

if __name__ == "__main__":
    main()
