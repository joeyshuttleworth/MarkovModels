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
    parser.add_argument('results_dir')
    parser.add_argument('--repeats', type=int, default=16)
    parser.add_argument('--wells', '-w', type=str, default=[], nargs='+')
    parser.add_argument('--removal_duration', '-r', default=5, type=float)
    parser.add_argument('--experiment_name', default='newtonrun4', type=str)
    parser.add_argument('--no_chains', '-N', default=0, help='mcmc chains to run', type=int)
    parser.add_argument('--chain_length', '-l', default=500, help='mcmc chains to run', type=int)
    parser.add_argument('--figsize', '-f', type=int, nargs=2, default=[4.5, 7.5])
    parser.add_argument('--use_parameter_file')
    parser.add_argument('-i', '--ignore_protocols', nargs='+',
                        default=['longap'])

    parser.add_argument('-o', '--output_dir')
    parser.add_argument("-F", "--file_format", default='pdf')
    parser.add_argument("-m", "--model_class", default='Beattie')
    parser.add_argument('--true_param_file')
    parser.add_argument('--fixed_param', default='Gkr')
    parser.add_argument('--prediction_protocol', default='longap')

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

    output_dir = common.setup_output_directory(args.output_dir, "multi_plot")

    fig = plt.figure(figsize=args.figsize)
    axes = create_axes(fig)

    results_df = pd.read_csv(os.path.join(args.results_dir, 'results_df.csv'))

    results_df = results_df[~results_df.protocol.isin(args.ignore_protocols)]

    global palette
    palette = sns.color_palette('husl', len(results_df.protocol.unique()))

    global protocols
    protocols = sorted(results_df.protocol.unique())
    relabel_dict = {p: f"$d_{i+1}$" for i, p in enumerate(protocols)}

    relabel_dict['longap'] = '$d^*$'

    results_df.replace({'protocol': relabel_dict}, inplace=True)

    results_df = results_df.sort_values('protocol')

    protocols = results_df.protocol.unique()

    print(results_df)
    for param_label in parameter_labels:
        results_df[param_label] = results_df[param_label].astype(np.float64)

    # plot scatter_plots
    scatter_plots(axes, results_df)
    # fig.tight_layout()

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
    plot_heatmaps(axes, prediction_df)

    data = pd.read_csv(os.path.join(args.results_dir, f"synthetic_data_{args.prediction_protocol}_0.csv"))

    do_prediction_plots(axes, results_df, args.prediction_protocol, data)

    fig.savefig(os.path.join(output_dir, f"figure.{args.file_format}"))


def do_prediction_plots(axes, results_df, prediction_protocol, data):
    times = data['time / ms'].astype(np.float64).values
    current = data['current / nA'].astype(np.float64).values

    print(times, current)

    vals = sorted(results_df[args.fixed_param].unique())

    assert(len(vals) == 5)

    voltage_func, t_start, t_end, t_step, protocol_desc = common.get_ramp_protocol_from_csv(prediction_protocol)

    voltages = np.array([voltage_func(t) for t in times])
    _, spike_indices = common.detect_spikes(times, voltages, window_size=0)

    indices = common.remove_indices(list(range(len(times))),
                                    [(spike, int(spike + args.removal_duration / t_step)) for spike in
                                     spike_indices])

    colno = 1
    prediction_axes = [axes[i] for i in range(len(axes)) if (i % 3) == colno
                       and i > 2]

    training_protocols = sorted(results_df.protocol.unique())

    # filter out ignored protocols
    training_protocols = [p for p in training_protocols if p not in args.ignore_protocols]

    val = sorted(results_df[args.fixed_param].unique())

    model_class = common.get_model_class(args.model_class)
    parameter_labels = model_class().get_parameter_labels()

    model = model_class(voltage_func, times, protocol_description=protocol_desc)
    solver = model.make_forward_solver_current()

    colours = [palette[i] for i in range(len(protocols))]

    print(linestyles)

    ymin, ymax = [0, 0]
    for i in range(5):
        # plot data
        ax = prediction_axes[i]

        ax.plot(times, current, color='grey', alpha=.2, lw=0.1)

        val = vals[i]

        predictions = []
        for training_protocol in sorted(training_protocols):

            row = results_df[(results_df.protocol == training_protocol) &
                             (results_df[args.fixed_param] == val)]
            parameters = row[parameter_labels].head(1).values.flatten()

            prediction = solver(parameters)
            predictions.append(prediction)
            # ax.plot(prediction, times, linewidth=0.1)

            ymin = min(ymin, prediction.min())
            ymax = max(ymax, prediction.max())

        predictions = np.array(predictions)

        max_pred = predictions.max(axis=0)
        min_pred = predictions.min(axis=0)
        ax.plot(times, max_pred, color='red',
                linewidth=.1)
        ax.plot(times, min_pred, color='red',
                linewidth=.1)

        ax.fill_between(times, min_pred, max_pred, color='orange',
                         alpha=.3, linewidth=0)
        axins = inset_axes(ax, width='50%', height='50%', loc='lower center')

        # axins.axis('off')
        axins.set_xticks([])
        axins.set_yticks([])

        axins.fill_between(times, min_pred, max_pred,
                            color='orange', alpha=.2, linewidth=0)

        for j in range(predictions.shape[0]):
            linestyle = linestyles[j]
            prediction = predictions[j, :]
            axins.plot(times, prediction, ls=linestyle,
                       linewidth=0.5, color=colours[j])

        # axins.plot(times, current, color='grey', alpha=.2,
        #            lw=0.1)

        axins.set_xlim([4250, 6000])
        axins.set_ylim(-0.15, 0.75)

        mark_inset(ax, axins, edgecolor="black", fc="none", loc1=1, loc2=2,
                   linewidth=.3, alpha=.8)

    for i, ax in enumerate(prediction_axes):
        ax.set_xlim(ymin, max(ymax, np.quantile(current, 0.9)))
        # ax.yaxis.tick_right()

        ax.set_xlim([0, 9000])
        ax.set_xticks([0, 8000])
        ax.set_xticklabels(['0s', '8s'], rotation='horizontal')

        yticks = [0, -2]
        ylabs = [str(l) + 'nA' for l in yticks]

        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabs, rotation='horizontal')
        ax.set_ylim([-2, 1])

        # remove spines
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)

        box = prediction_axes[i].get_position()
        # box.x0 += 0.05
        box.x1 += 0.05
        ax.set_position(box)

        if i != 4:
            ax.get_shared_x_axes().join(ax, prediction_axes[-1])
            ax.set_xticks([])

    # Plot voltage
    axes[colno].plot(times[::50], [voltage_func(t) for t in times][::50], color='black',
                     linewidth=.5)

    # axes[colno].yaxis.tick_right()
    labels = ['0s', '7.5s']
    axes[colno].spines.right.set_visible(False)
    axes[colno].spines.top.set_visible(False)

    axes[colno].get_shared_x_axes().join(ax, prediction_axes[-1])
    axes[colno].set_xticks([])

    axes[colno].set_yticks([-100, 40])
    axes[colno].set_yticklabels(['-100mV', '+40mV'])

    ax = axes[colno]
    box = ax.get_position()
    # box.x0 += 0.05
    box.x1 += 0.05
    ax.set_position(box)



def plot_heatmaps(axes, prediction_df):

    colno = 2
    # Drop parameter sets fitted to 'longap', for example

    averaged_df = prediction_df.groupby([args.fixed_param, 'fitting_protocol', 'validation_protocol']).mean().reset_index()

    vals = sorted(prediction_df[args.fixed_param].unique())
    print(args.fixed_param, vals)

    vmin, vmax = averaged_df['RMSE'].min(), averaged_df['RMSE'].max()

    assert(len(vals) == 5)

    # Get central column
    heatmap_axes = [axes[i] for i in range(len(axes)) if i > 2 and (i % 3) == colno]

    cmap = sns.cm.mako_r
    norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)

    for i in range(5):
        ax = heatmap_axes[i]
        sub_df = averaged_df[averaged_df[args.fixed_param] == vals[i]].copy()
        sub_df = sub_df[~sub_df.fitting_protocol.isin(['V', '$d^*$'])]

        pivot_df = sub_df.pivot(index='validation_protocol', columns='fitting_protocol',
                                values='RMSE')

        hm = sns.heatmap(pivot_df, ax=ax, square=True, cbar=False, norm=norm,
                         cmap=cmap)

        hm.set_yticklabels(hm.get_yticklabels(), rotation=0)

        if i != 0:
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_yticks([])
            ax.set_xticks([])
        else:
            ax.set_ylabel('validation')
            ax.set_xlabel('fitting')
            ax.xaxis.tick_top()
            ax.yaxis.tick_right()

    cbar_kws = {'orientation': 'horizontal',
                'fraction': 1,
                'aspect': 10}

    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

    axes[colno].axis('off')
    cbar = plt.colorbar(ax=axes[colno], cmap=cmap, mappable=mappable,
                        norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax), label='',
                        **cbar_kws)

    cbar_ticks = [vmin]

    cbar.set_ticks(cbar_ticks)

    axes[colno].set_title('RMSE')


def prediction_plots(axes, results_df):
    pass


def create_axes(fig):
    gs = GridSpec(6, 3, height_ratios=[0.4, 1, 1, 1, 1, 1])

    # set up padding and margins
    gs.update(left=0.15, right=0.85, top=.975, bottom=0.075, wspace=0.75,
              hspace=0.5)

    axes = [fig.add_subplot(cell) for cell in gs]


    box = axes[0].get_position()
    box.x0 -= 0.05
    box.x1 -= 0.05
    axes[0].set_position(box)

    return axes


def scatter_plots(axes, results_df, params=['p1', 'p2'], col=0):
    # Use p1, p2
    xlim = results_df[params[0]].min()*.9, results_df[params[0]].max()*1.1
    ylim = results_df[params[1]].min()*.9, results_df[params[1]].max()*1.1

    xlim = np.array(xlim) * 1000

    scatter_axes = [ax for i, ax in enumerate(axes) if (i % 3) == col and i > 2]

    assert(len(scatter_axes) == 5)

    results_df = results_df.copy()

    vals = sorted(results_df[args.fixed_param].unique())
    assert(len(vals) == len(scatter_axes))

    val1 = true_parameters[0]
    val2 = true_parameters[1]

    markers = ['1', '2', '3', '4', '+', 'x']
    markers = [markers[i] for i in range(len(results_df.protocol.unique()))]

    colours = [palette[i] for i in range(len(protocols))]

    for i, val in enumerate(vals):
        ax = scatter_axes[i]
        sub_df = results_df[results_df[args.fixed_param] == val]
        # make scatter plot

        legend = False

        sub_df[params[0]] *= 1000

        g = sns.scatterplot(data=sub_df, x=params[0], y=params[1],
                             hue='protocol', style='protocol', legend=legend,
                             markers=markers, linewidth=0.2, ax=ax, size=1,
                             alpha=0.9)

        ax.axvline(val1*1000, linestyle='dotted', color='grey',
                   linewidth=.3)

        ax.axhline(val2, linestyle='dotted',
                   linewidth=.3, color='grey')

        g.set_xlabel(r'$p_1\times \textrm{1E3}$ / $\textrm{s}^{-1}$')
        g.set_ylabel(r'$p_2$ / $\textrm{V}^{-1}$')

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)

        # xlabels = ['{:.1e}'.format(x) for x in g.get_xticks()]
        # g.set_xticklabels(xlabels)

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
                             label=label, linewidth=.5) for i, (label, marker,
                                                                 color) in enumerate(zip(protocols, markers,
                                                                                         colours))]

    handles, labels = list(handles), list(protocols)
    ax.legend(labels=labels, handles=handles, **legend_kws)
    ax.axis('off')



if __name__ == "__main__":
    main()
