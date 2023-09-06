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
rc('figure', dpi=1000)
rc('axes', facecolor=[0]*4)
rc('savefig', facecolor=[0]*4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('results_dirs', nargs='+')
    parser.add_argument('--repeats', type=int, default=16)
    parser.add_argument('--wells', '-w', type=str, default=[], nargs='+')
    parser.add_argument('--removal_duration', '-r', default=5, type=float)
    parser.add_argument('--experiment_name', default='newtonrun4', type=str)
    parser.add_argument('--no_chains', '-N', default=0, help='mcmc chains to run', type=int)
    parser.add_argument('--chain_length', '-l', default=500, help='mcmc chains to run', type=int)
    parser.add_argument('--figsize', '-f', type=int, nargs=2, default=[4.685, 6.5])
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

    output_dir = common.setup_output_directory(args.output_dir, "CaseII_main")

    global fig
    fig = plt.figure(figsize=args.figsize)
    axes = create_axes(fig)

    global protocols
    protocols = sorted(pd.read_csv(os.path.join(args.results_dirs[0], 'fitting.csv')).protocol.unique())

    global relabel_dict
    relabel_dict = {p: f"$d_{i+1}$" for i, p in enumerate([p for p in protocols if p not in args.ignore_protocols and p not in args.prediction_protocol])}
    relabel_dict['longap'] = '$d^*$'

    print("protocols protocols:", relabel_dict)

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

    # plot scatter_plots
    scatter_plots(axes, results_dfs)
    # fig.tight_layout()

    # Plot heatmaps
    models = ['Beattie', 'Wang']
    prediction_dfs = []
    for model, results_dir in zip(models, args.results_dirs):
        prediction_df = pd.read_csv(os.path.join(results_dir,
                                                 'predictions_df.csv'))
        prediction_df.replace({'fitting_protocol': relabel_dict,
                               'validation_protocol': relabel_dict}, inplace=True)

        keep_rows = ~prediction_df.validation_protocol.isin(args.ignore_protocols) &\
            prediction_df.fitting_protocol.isin(relabel_dict.values())

        prediction_df = prediction_df[keep_rows]
        prediction_df['model'] = model
        prediction_dfs.append(prediction_df)

    plot_heatmaps(axes, prediction_dfs)

    current = pd.read_csv(os.path.join(args.data_dir,
                                               f"synthetic-{args.prediction_protocol}-1.csv"))['current'].values.flatten().astype(np.float64)
    times = pd.read_csv(os.path.join(args.data_dir,
                                            f"synthetic-{args.prediction_protocol}-times.csv"))['time'].values.flatten().astype(np.float64)

    do_prediction_plots(axes, results_dfs, args.prediction_protocol, current, times)

    fig.set_canvas(plt.gcf().canvas)
    fig.savefig(os.path.join(output_dir, f"Fig7.{args.file_format}"))


def do_prediction_plots(axes, results_dfs, prediction_protocol, current, times):

    voltage_func, times, protocol_desc = common.get_ramp_protocol_from_csv(prediction_protocol)

    voltages = np.array([voltage_func(t) for t in times])
    spike_times, _ = common.detect_spikes(times, voltages, window_size=0)
    indices = None

    colno = 1
    prediction_axes = [axes[i] for i in range(len(axes)) if (i % 3) == colno
                       and i > 2]

    for ax in prediction_axes[2:]:
        ax.set_visible(False)

    training_protocols = sorted(results_dfs[0].protocol.unique())

    unmap_dict = {v: k for k, v in relabel_dict.items()}

    # filter out ignored protocols
    training_protocols = [p for p in training_protocols if unmap_dict[p] not in
                          args.ignore_protocols]

    model_class = common.get_model_class(args.model_class)

    model_names = ['Beattie', 'Wang']
    ymin, ymax = [np.inf, -np.inf]

    print(training_protocols)

    for i, results_df in enumerate(results_dfs):
        # plot data
        ax = prediction_axes[i]
        ax.plot(times, current, color='grey', alpha=.5, lw=0.3)

        model_class = common.get_model_class(model_names[i])
        parameter_labels = model_class().get_parameter_labels()

        model = model_class(voltage_func, times,
                            protocol_description=protocol_desc)
        solver = model.make_forward_solver_current()

        predictions = []
        for training_protocol in sorted(training_protocols):
            print(training_protocol)

            results_df['score'] = results_df['score'].astype(np.float64)
            print(results_df[results_df.protocol == training_protocol])

            row = results_df[(results_df.protocol == training_protocol)
                             & (results_df.well.astype(int) == 1)].sort_values('score')
            parameters = row[parameter_labels].head(1).values.flatten().astype(np.float64)
            print('parameters are', parameters)

            prediction = solver(parameters)
            predictions.append(prediction)
            # ax.plot(prediction, times, linewidth=0.1)

            ymin = min(ymin, prediction.min())
            ymax = max(ymax, prediction.max())

        predictions = np.array(predictions)

        max_pred = predictions.max(axis=0)
        min_pred = predictions.min(axis=0)
        ax.plot(times, max_pred, color='red',
                linewidth=.15, )
        ax.plot(times, min_pred, color='red',
                linewidth=.15, )

        ax.fill_between(times, min_pred, max_pred, color='orange', alpha=0,
                        linewidth=0, rasterized=False)
        axins = inset_axes(ax, width='50%', height='45%', loc='lower center')

        # axins.axis('off')
        axins.set_xticks([])
        axins.set_yticks([])

        axins.fill_between(times, min_pred, max_pred, color='orange', alpha=.2,
                           linewidth=0, rasterized=False)

        for j in range(predictions.shape[0]):
            linestyle = linestyles[j]
            prediction = predictions[j, :]
            axins.plot(times, prediction, ls=linestyle,
                       linewidth=0.5, color=palette[j],
                       )

        # axins.plot(times, current, color='grey', alpha=.2,
                   # linewidth=0)

        axins.set_xlim([5000, 6000])
        axins.set_ylim(-0.5, 2)

        mark_inset(ax, axins, edgecolor="black", fc="none", loc1=1, loc2=2,
                   linewidth=.3, alpha=.8)

    for i, ax in enumerate(prediction_axes):
        ax.set_xticks([0, 8000])
        ax.set_xticklabels(['0s', '8s'], rotation='horizontal')

        ax.set_yticks([-20, 0, 10])
        yticks = ax.get_yticks()

        ylabs = [str(l) + '' for l in yticks]

        ax.set_yticklabels(ylabs, rotation='horizontal')
        ax.set_ylim([-20, 14])

        # remove spines
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)

        box = prediction_axes[i].get_position()
        # box.x0 += 0.05
        box.x1 += 0.05
        ax.set_position(box)

        ax.set_rasterization_zorder(10)

        ax.set_ylabel(r'$I_\textrm{Kr}$ (nA)')


    # Plot voltage
    axes[colno].plot(times[::50], [voltage_func(t) for t in times][::50], color='black',
                     linewidth=.5)

    axes[colno].set_ylabel(r'$V$ (mV)')

    # axes[colno].yaxis.tick_right()
    axes[colno].spines.right.set_visible(False)
    axes[colno].spines.top.set_visible(False)

    prediction_axes[-1].set_xlabel(r'$t$ (s)')

    prediction_axes[-1].sharex(axes[colno])

    axes[colno].set_yticks([-100, 40])
    # axes[colno].set_yticklabels(['-100mV', '+40mV'])

    ax = axes[colno]
    box = ax.get_position()
    # box.x0 += 0.05
    box.x1 += 0.05
    ax.set_position(box)

    axes[colno].set_xticks([])
    axes[colno + 3].set_xticks([])
    labels = ['0', '7.5']
    axes[colno + 6].set_xticks([0, 7500])
    axes[colno + 6].set_xticklabels(labels)

    for ax in prediction_axes:
        ax.set_xlim([0, 9000])

    axes[1].set_xlim([0, 9000])


def plot_heatmaps(axes, prediction_dfs):

    colno = 2
    # Drop parameter sets fitted to 'longap', for example
    # Get central column
    heatmap_axes = [axes[i] for i in range(len(axes)) if i > 2 and (i % 3) == colno]
    prediction_axes = [axes[i] for i in range(len(axes)) if i > 2 and (i % 3) == colno - 1]

    for ax in heatmap_axes[2:]:
        ax.set_visible(False)

    cmap = sns.cm.mako_r

    joint_df = pd.concat(prediction_dfs, ignore_index=True)
    averaged_df = joint_df.groupby(['fitting_protocol', 'validation_protocol',
                                    'model'])['RMSE'].mean().reset_index()

    if args.vlim is None:
        vmin, vmax = averaged_df['RMSE'].min(), averaged_df['RMSE'].max()
    else:
        vmin, vmax = args.vlim

    norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)

    for i, model in enumerate(sorted(averaged_df.model.unique())):
        ax = heatmap_axes[i]
        sub_df = averaged_df[averaged_df.model == model].copy()

        # Ignore training to validation protocols, labels 'V' and '$d^*$', have
        # been used for validation protocols
        sub_df = sub_df[~sub_df.fitting_protocol.isin(['V', '$d^*$'])]

        pivot_df = sub_df.pivot(columns='fitting_protocol',
                                index='validation_protocol', values='RMSE')

        hm = sns.heatmap(pivot_df, ax=ax, square=True, cbar=False, norm=norm,
                         cmap=cmap)

        hm.set_yticklabels(hm.get_yticklabels(), rotation=0)

        # Add arrow from heatmap to prediction plot
        ax2 = prediction_axes[i]
        xyA = [7750, 1]
        xyB = [-.1, 0.5]
        con = ConnectionPatch(
            xyA=xyB, coordsA=ax.transData,
            xyB=xyA, coordsB=ax2.transData,
            arrowstyle="->", shrinkB=5)

        # Add yellow highlight to first row
        autoAxis = ax.axis()
        rec = Rectangle(
            (autoAxis[0] - 0.05, autoAxis[3] - 0.05),
            (autoAxis[1] - autoAxis[0] + 0.1),
            1.1,
            fill=False,
            color='yellow',
            lw=.75
            )

        if i == 0:
            fig.add_artist(con)
            rec = ax.add_patch(rec)
            rec.set_clip_on(False)



        if i != 0:
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_yticks([])
            ax.set_xticks([])
        else:
            ax.set_xlabel('training', labelpad=0)
            ax.set_ylabel('validation')
            ax.xaxis.tick_top()
            ax.yaxis.tick_right()

    cbar_kws = {'orientation': 'horizontal',
                'fraction': 1,
                'aspect': 10,
                }

    norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = axes[colno]
    # cax.axis('off')
    cax.set_xticks([])
    cax.set_yticks([])

    for side in ['left', 'top', 'right', 'bottom']:
        cax.spines[side].set_visible(False)

    im = cax.imshow([[vmin, vmax]], cmap=cmap, norm=norm)
    im.set_visible(False)
    cax.plot([0], [0])

    cbar = plt.colorbar(ax=cax,
                        norm=norm, cmap=cmap, label='', **cbar_kws,
                        mappable=im, ticks=matplotlib.ticker.LogLocator(base=10))

    cax = cbar.ax
    # Move cbar up
    box = cax.get_position()
    box.y1 += 0.06
    box.y0 += 0.06
    cax.set_position(box)

    cax.xaxis.set_label_position('top')
    cax.set_xlabel(r'$\log_{10}$ RMSE')


def create_axes(fig):

    ncols = 4
    nrows = 6

    global gs

    gs = GridSpec(nrows, ncols, height_ratios=[0.3, 1, 1, 1, 1, 1],
                  width_ratios=[.05, 1, 1, .8], wspace=.55,
                  right=.95,
                  left=.11,
                  # hspace=.5,
                  bottom=0.1,
                  figure=fig)

    bottom_axes = [fig.add_subplot(gs[2, i]) for i in range(ncols) if i % 4 != 0]

    axes = []
    for i in range(2):
        cells = [gs[i, j + 1] for j in range(ncols - 1)]

        for j, cell in enumerate(cells):
            # if j != 2:
            #     sharex = bottom_axes[j]
            # else:
            # sharex = None
            axes.append(fig.add_subplot(cell))

    axes = axes + list(bottom_axes)

    for ax in axes:
        ax.set_rasterization_zorder(2)

    axes[3].set_title(r'\textbf{a}', loc='left', y=1.2)
    axes[1].set_title(r'\textbf{b}', loc='left')
    axes[5].set_title(r'\textbf{d}', loc='left')
    axes[4].set_title(r'\textbf{c}', loc='left', y=1.2)

    # move entire first row up
    for i, ax in enumerate(axes[:3]):
        box = ax.get_position()
        box.y0 += .075
        box.y1 += .075
        ax.set_position(box)

    box = axes[0].get_position()
    box.x0 -= 0.1
    box.x1 -= 0.1
    axes[0].set_position(box)

    number_line_axes = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[2, 0]),
    ]

    for ax in number_line_axes:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        pos1 = ax.get_position()
        # pos1.y1 -= .05
        # pos1.y0 -= .05
        pos1.x0 -= 0.1
        pos1.x1 -= 0.1
        ax.set_position(pos1)

        for side in ['right', 'top', 'bottom', 'left']:
            ax.spines[side].set_visible(False)

    for ax, model in zip(number_line_axes, ['Beattie model', 'Wang model']):
        ax.text(0.5, 0, model, rotation=90)

    # pos1 = number_line_axes.get_position()
    # pos1.y1 -= .05
    # pos1.y0 += .05
    # pos1.x0 -= 0.05
    # pos1.x1 -= 0.05
    # number_line_axes.set_position(pos1)

    return axes


def scatter_plots(axes, results_dfs, params=['p1', 'p2'], col=0):
    scatter_axes = [ax for i, ax in enumerate(axes) if (i % 3) == col and i > 2]

    for ax in scatter_axes[2:]:
        ax.set_visible(False)

    # assert(len(scatter_axes) == 5)
    gkrs = pd.concat(results_dfs)['Gkr'].values.flatten()
    ylims = [0.95 * min(gkrs), 1.05 * max(gkrs)]
    models = ['Beattie', 'Wang']
    for i, _ in enumerate(results_dfs):
        print(results_dfs[i].columns)
        results_dfs[i]['model'] = models[i]

    markers = ['1', '2', '3', '4', '+', 'x']
    markers = [markers[i] for i in range(len(results_dfs[0].protocol.unique()))]
    colours = [palette[i] for i in range(len(results_dfs[0].protocol.unique()))]

    for i, results_df in enumerate(results_dfs):
        ax = scatter_axes[i]

        ax.axhline(BeattieModel().get_default_parameters()[-1], ls='--',
                   color='grey', alpha=.9, lw=.5)
        sns.scatterplot(ax=ax, data=results_df, y=r'Gkr', x='protocol',
                        palette=palette, hue='protocol', style='protocol',
                        legend=False, size=2, linewidth=0)
        ax.set_ylim(ylims)
        ax.set_ylabel(r'$g$', rotation=0)
        ax.set_xlabel(r'protocol', rotation=0)

    for i, ax in enumerate(scatter_axes[:2]):
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.set_xlabel('')

    # Put legend on the top left axis
    ax = axes[0]
    legend_kws = {'loc': 10,
                  'frameon': False,
                  'bbox_to_anchor': [0, 0, 1, 1],
                  'ncol': 2,
                  'fontsize': 8
                  }

    ticks = scatter_axes[0].get_xticks()
    tick_labels = scatter_axes[0].get_xticklabels()
    scatter_axes[0].set_xticks([])
    scatter_axes[1].set_xticks(ticks)
    scatter_axes[1].set_xticklabels(tick_labels)
    scatter_axes[1].set_xlabel('protocol')

    handles = [mlines.Line2D(xdata=[1], ydata=[1], color=color, marker=marker,
                             linestyle=linestyles[i], markersize=5,
                             label=label, linewidth=.3) for i, (label, marker,
                                                                 color) in enumerate(zip(protocols, markers,
                                                                                         colours))]

    handles, labels = list(handles), list(results_dfs[0]['protocol'].unique())
    ax.legend(labels=labels, handles=handles, **legend_kws)
    ax.axis('off')

    for ax in scatter_axes:
        pos = ax.get_position()
        pos.x0 -= .05
        pos.x1 -= .05
        ax.set_position(pos)


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
