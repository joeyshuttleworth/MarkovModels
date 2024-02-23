#!/usr/bin/env python3

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec

from markovmodels.ArtefactModel import ArtefactModel
from markovmodels.model_generation import make_model_of_class
from markovmodels.fitting import get_best_params
from markovmodels.utilities import setup_output_directory, get_data, get_all_wells_in_directory
from markovmodels.voltage_protocols import get_protocol_list, get_ramp_protocol_from_json
from markovmodels.fitting import compute_predictions_df, adjust_kinetics


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('data_directory', help='directory where data is stored')
    parser.add_argument('fitting_case', type=str)
    parser.add_argument('--subtraction_df')
    parser.add_argument('--model_class')
    parser.add_argument('--experiment_name', '-e', default='newtonrun4')
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('--figsize', '-f', nargs=2, type=float, default=[4.675, 2.8])
    parser.add_argument('--fig_title', '-t', default='')
    parser.add_argument('--parameter_file')
    parser.add_argument('--nolegend', action='store_true')
    parser.add_argument('--dpi', '-d', default=500, type=int)
    parser.add_argument('--fontsize', type=int)
    parser.add_argument('--show_uncertainty', action='store_true')
    parser.add_argument('--ignore_protocols', nargs='+', default=['longap'])
    parser.add_argument('--shared_plot_limits', action='store_true')
    parser.add_argument('--no_voltage', action='store_true')
    parser.add_argument('--file_format', default='')
    parser.add_argument('--reversal', default=-91.71, type=float)
    parser.add_argument('--additional_protocols', default=[], nargs='+')
    parser.add_argument('--vlim', type=float, nargs=2)

    global args
    args = parser.parse_args()

    output_dir = setup_output_directory(args.output, 'plot_heatmaps')

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    ax = fig.subplots()

    params_df = pd.read_csv(args.parameter_file)

    model = make_model_of_class(args.model_class)

    solver = model.make_forward_solver_current()

    protocol_dict = {}

    args.use_artefact_model = False
    if args.fitting_case == 'I':
        args.use_artefact_model = True

    if args.use_artefact_model:
        model = ArtefactModel(model)

    for protocol in np.unique(list(params_df.protocol) + args.additional_protocols):
        desc, _ = get_ramp_protocol_from_json(protocol, os.path.join(args.data_directory, 'protocols'),
                                              args.experiment_name)

        times = np.loadtxt(os.path.join(args.data_directory,
                                        f"{args.experiment_name}-{protocol}-times.csv")).astype(np.float64).flatten()

        protocol_dict[protocol] = desc, times


    if args.fitting_case in ['0b']:
        if not args.subtraction_df:
            raise Exception()
        subtraction_df = pd.read_csv(args.subtraction_df)
        E_rev_df = subtraction_df.set_index(['protocol', 'well', 'sweep'])['fitted_E_rev']
        params_df = adjust_kinetics(args.model_class, params_df, E_rev_df,
                                    args.E_rev)

    model = make_model_of_class(args.model_class)
    prediction_df = compute_predictions_df(params_df, output_dir,
                                           protocol_dict, solver=solver,
                                           args=args)

    plot_heatmap(ax, prediction_df)


def plot_heatmap(axes, prediction_df):

    averaged_df = prediction_df.groupby([args.fixed_param, 'fitting_protocol', 'validation_protocol']).mean().reset_index()

    vals = sorted(prediction_df[args.fixed_param].unique())

    if args.vlim is None:
        vmin, vmax = averaged_df['RMSE'].min(), averaged_df['RMSE'].max()
    else:
        vmin, vmax = args.vlim
    # vmin = 10**-1.5
    # vmax = 10**0

    assert(len(vals) == 5)

    cmap = sns.cm.mako_r
    norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)

    sub_df = sub_df[~sub_df.fitting_protocol.isin(args.ignore_protocols)]

    pivot_df = sub_df.pivot(columns='fitting_protocol',
                            index='validation_protocol', values='RMSE')

    hm = sns.heatmap(pivot_df, ax=ax, square=True, cbar=False, norm=norm,
                    cmap=cmap)
    hm.set_yticklabels(hm.get_yticklabels(), rotation=0)

    ax.set_xlabel('training', labelpad=0)
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

    plt.savefig(os.path.join(output_dir, "heatmap.png"))


if __name__ == '__main__':
    main()
