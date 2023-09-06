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

    output_dir = common.setup_output_directory(args.output_dir, "multi_plot")

    global fig
    fig = plt.figure(figsize=args.figsize)
    axes = create_axes(fig)

    results_df = pd.read_csv(os.path.join(args.results_dir, 'results_df.csv'))

    results_df = results_df[~results_df.protocol.isin(args.ignore_protocols)]

    global palette
    palette = sns.color_palette('husl', len(results_df.protocol.unique()))

    global protocols
    protocols = sorted(results_df.protocol.unique())
    relabel_dict = {p: f"$d_{i+1}$" for i, p in enumerate([p for p in protocols if p not in args.ignore_protocols and p != 'longap'])}

    relabel_dict['longap'] = '$d_0$'

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

    # gs.tight_layout(fig)
    fig.savefig(os.path.join(output_dir, f"fig4.{args.file_format}"))


def do_prediction_plots(axes, results_df, prediction_protocol, data):
    times = data['time / ms'].astype(np.float64).values
    current = data['current / nA'].astype(np.float64).values

    print(times, current)

    vals = sorted(results_df[args.fixed_param].unique())

    assert(len(vals) == 5)

    voltage_func, times, protocol_desc = common.get_ramp_protocol_from_csv(prediction_protocol)

    voltages = np.array([voltage_func(t) for t in times])
    # _, spike_indices = common.detect_spikes(times, voltages, window_size=0)

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

        ax.plot(times, current, color='grey', alpha=.2, lw=0.3,
                )
        # ax.plot(times, solver(), color='grey', lw=.3)

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

        # axins.axis('off')
        axins.set_xticks([])
        axins.set_yticks([])

        axins.fill_between(times, min_pred, max_pred, color='orange', alpha=.2,
                           lw=0)

        for j in range(predictions.shape[0]):
            linestyle = linestyles[j]
            prediction = predictions[j, :]
            axins.plot(times, prediction, ls=linestyle,
                       lw=0.5, color=colours[j])

        # axins.plot(times, current, color='grey', alpha=.2,
        #            lw=0.1)

        axins.set_xlim([4250, 6000])
        axins.set_ylim(-0.15, 0.75)

        mark_inset(ax, axins, edgecolor="black", fc="none", loc1=1, loc2=2,
                   lw=.3, alpha=.8)

    for i, ax in enumerate(prediction_axes):
        ax.set_xlim(ymin, max(ymax, np.quantile(current, 0.9)))
        # ax.yaxis.tick_right()

        ax.set_xlim([0, 9000])
        ax.set_xticks([0, 8000])
        ax.set_xticklabels(['0', '8'], rotation='horizontal')

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

        ax.set_rasterization_zorder(2)

    # Plot voltage
    axes[colno].plot(times[::50], [voltage_func(t) for t in times][::50], color='black',
                     lw=.5)

    prediction_axes[-1].set_xlabel(r'$t$ (s)')

    # axes[colno].yaxis.tick_right()
    labels = ['0s', '7.5s']
    axes[colno].spines.right.set_visible(False)
    axes[colno].spines.top.set_visible(False)

    axes[colno].set_xticks([])

    for ax in prediction_axes[:-1]:
        ax.set_xticks([])

    prediction_axes[-1].set_xticks([0, 7500])
    prediction_axes[-1].set_xticklabels(labels)

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

    if args.vlim is None:
        vmin, vmax = averaged_df['RMSE'].min(), averaged_df['RMSE'].max()
    else:
        vmin, vmax = args.vlim
    # vmin = 10**-1.5
    # vmax = 10**0

    assert(len(vals) == 5)

    # Get central column
    heatmap_axes = [axes[i] for i in range(len(axes)) if i > 2 and (i % 3) == colno]
    prediction_axes = [axes[i] for i in range(len(axes)) if i > 2 and (i % 3) == colno - 1]

    cmap = sns.cm.mako_r
    norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)

    for i in range(5):
        ax = heatmap_axes[i]
        sub_df = averaged_df[averaged_df[args.fixed_param] == vals[i]].copy()
        sub_df = sub_df[~sub_df.fitting_protocol.isin(['V', '$d_0$'])]

        pivot_df = sub_df.pivot(columns='fitting_protocol',
                                index='validation_protocol', values='RMSE')

        hm = sns.heatmap(pivot_df, ax=ax, square=True, cbar=False, norm=norm,
                         cmap=cmap)

        # Add arrow from heatmap to prediction plot
        ax2 = prediction_axes[i]
        xyA = [7750, .2]
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

        hm.set_yticklabels(hm.get_yticklabels(), rotation=0)

        if i != 0:
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_yticks([])
            ax.set_xticks([])
        else:
            ax.set_xlabel('training')
            ax.set_ylabel('validation')
            ax.xaxis.tick_top()
            ax.yaxis.tick_right()

    cbar_kws = {'orientation': 'horizontal',
                'fraction': 1,
                'aspect': 10,
                }

    norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    cax = axes[colno]
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
    # Move cbar up and make thinner
    box = cax.get_position()
    box.y1 += 0.035
    box.y0 += 0.035
    cax.set_position(box)

    cax.xaxis.set_label_position('top')
    cax.set_xlabel(r'$\log_{10}$ RMSE')


def create_axes(fig):
    global gs
    ncols = 4
    nrows = 6

    gs = GridSpec(nrows, ncols, height_ratios=[0.3, 1, 1, 1, 1, 1],
                  width_ratios=[.05, 1, 1, .8], wspace=.55,
                  right=.95,
                  hspace=.5,
                  left=.11,
                  bottom=0.1,
                  figure=fig)

    bottom_axes = [fig.add_subplot(gs[-1, i]) for i in range(ncols) if i % 4 != 0]
    axes = []
    for i in range(nrows - 1):
        cells = [gs[i, j + 1] for j in range(ncols - 1)]

        for j, cell in enumerate(cells):
            # If not heatmap
            if j != 2:
                sharex = bottom_axes[j]
            else:
                sharex = None
            # axes.append(fig.add_subplot(cell, sharex=sharex))
            axes.append(fig.add_subplot(cell))

    axes = axes + list(bottom_axes)
    print(axes)

    axes[3].set_title(r'\textbf{a}', loc='left', y=1.2)
    axes[1].set_title(r'\textbf{b}', loc='left')
    axes[5].set_title(r'\textbf{d}', loc='left')
    axes[4].set_title(r'\textbf{c}', loc='left', y=1.2)

    # move entire first row up
    for i, ax in enumerate(axes[:3]):
        box = ax.get_position()
        box.y0 += .06
        box.y1 += .06
        ax.set_position(box)

    # Move legend left
    box = axes[0].get_position()
    # box.x0 -= 0.15
    box.x1 -= 0.15
    axes[0].set_position(box)

    number_line_axes = fig.add_subplot(gs[1:, 0])

    number_line_axes.xaxis.set_visible(False)
    number_line_axes.set_yticks([1, 2, 3, 4, 5])
    number_line_axes.set_ylim([1, 5])
    tick_labels = [
        r'$\frac{1}{4}$',
        r'$\frac{1}{2}$',
        r'$1$',
        r'$2$',
        r'$4$',
    ]

    number_line_axes.set_yticklabels(tick_labels)
    number_line_axes.set_ylabel(r'$\lambda$', rotation=0, loc='top')
    number_line_axes.invert_yaxis()
    # number_line_axes.yaxis.tick_right()

    for side in ['right', 'top', 'bottom']:
        number_line_axes.spines[side].set_visible(False)

    pos1 = number_line_axes.get_position()
    pos1.y1 -= .05
    pos1.y0 += .05
    pos1.x0 -= 0.05
    pos1.x1 -= 0.05
    number_line_axes.set_position(pos1)

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
    print(markers)
    colours = [palette[i] for i in range(len(results_df.protocol.unique()))]

    for i, val in enumerate(vals):
        ax = scatter_axes[i]
        sub_df = results_df[results_df[args.fixed_param] == val].copy()
        # make scatter plot

        legend = False

        sub_df[params[0]] *= 1000

        g = sns.scatterplot(data=sub_df, x=params[0], y=params[1],
                            hue='protocol', style='protocol', legend=legend,
                            palette=palette, size=1, markers=markers,
                            linewidth=0.5, ax=ax)

        # ax.axvline(val1*1000, linestyle='dotted', color='grey',
        #            lw=.3)

        # ax.axhline(val2, linestyle='dotted',
        #            lw=.3, color='grey')

        # g.set_xlabel(r'$p_1$ / $\textrm{ms}^{-1}$')
        g.set_ylabel(r'$p_2$ (V$^{-1})$')

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

    ticks = scatter_axes[0].get_xticks()
    ticks = [ticks[0], ticks[-1]]
    tick_labels = [f"{v:.2f}" for v in ticks]
    for i in range(4):
        scatter_axes[i].set_xticks([])
        scatter_axes[i].set_xlabel('')

    scatter_axes[4].set_xticks(ticks)
    scatter_axes[4].set_xticklabels(tick_labels)
    scatter_axes[4].set_xlabel(r'$p_1 \times 10^3$ (s$^{-1})$')

    handles = [mlines.Line2D(xdata=[1], ydata=[1], color=color, marker=marker,
                             linestyle=linestyles[i], markersize=5,
                             label=label, lw=.3) for i, (label, marker, color)
               in enumerate(zip(protocols, markers, colours))]

    handles, labels = list(handles), list(protocols)
    ax.legend(labels=labels, handles=handles, **legend_kws)
    ax.axis('off')

    # Draw trajectories
    traj_df = results_df.sort_values(args.fixed_param)
    traj_df = traj_df[~traj_df.protocol.isin(args.ignore_protocols)]

    print(sorted(traj_df.protocol.unique()))
    for i, protocol in enumerate(sorted(traj_df.protocol.unique())):
        xs = traj_df[traj_df.protocol == protocol][params[0]]
        ys = traj_df[traj_df.protocol == protocol][params[1]]
        # x_new = np.linspace(*np.quantile(traj_df[args.fixed_param], [0, 1]), 50)
        # y_new = np.polynomial.polynomial.polyval(x_new, coefs)
        tck, u = scipy.interpolate.splprep([xs, ys])
        x_new, y_new = scipy.interpolate.splev(np.linspace(0, 1, 500), tck)
        for ax in scatter_axes:
            ax.plot(x_new*1e3, y_new, lw=.3, color=colours[i], alpha=.75)

    for ax in scatter_axes:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

if __name__ == "__main__":
    main()
