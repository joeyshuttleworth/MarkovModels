import argparse
import os

import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.gridspec import GridSpec

from matplotlib import rc

import markovmodels
from markovmodels.model_generation import make_model_of_class
from markovmodels.fitting import get_best_params, compute_predictions_df
from markovmodels.ArtefactModel import ArtefactModel
from markovmodels.utilities import setup_output_directory, get_data, get_all_wells_in_directory
from markovmodels.voltage_protocols import get_protocol_list, get_ramp_protocol_from_json, make_voltage_function_from_description
from markovmodels.voltage_protocols import remove_spikes, detect_spikes

rc('font', **{'size': 12})
# rc('text', usetex=True)
rc('figure', dpi=400, facecolor=[0]*4)
rc('axes', facecolor=[0]*4)
rc('savefig', facecolor=[0]*4)
rc('figure', autolayout=True)

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', help='directory where data is stored')
    parser.add_argument('fitting_results', type=str)
    parser.add_argument('subtraction_df')
    parser.add_argument('--removal_duration', type=float, default=5.0)
    parser.add_argument('--experiment_name', '-e', default='newtonrun4')
    parser.add_argument('--validation_protocols', default=['longap'], nargs='+')
    parser.add_argument('--figsize', '-f', nargs=2, type=float, default=[5.54, 7])
    parser.add_argument('--fig_title', '-t', default='')
    parser.add_argument('--nolegend', action='store_true')
    parser.add_argument('--dpi', '-d', default=500, type=int)
    parser.add_argument('--fontsize', type=int, default=12)
    parser.add_argument('--show_uncertainty', action='store_true')
    parser.add_argument('--shared_plot_limits', action='store_true')
    parser.add_argument('--no_voltage', action='store_true')
    parser.add_argument('--file_format', default='')
    parser.add_argument('--reversal', default=-91.71, type=float)
    parser.add_argument('--output')

    global args
    args = parser.parse_args()

    args.model_classes = ['model2', 'model3', 'model10', 'Wang']

    global output_dir
    output_dir = setup_output_directory(args.output, 'chapter_4_optimisation_results')

    subtraction_df = pd.read_csv(args.subtraction_df)

    if args.fontsize:
        matplotlib.rcParams.update({'font.size': args.fontsize})

    infer_reversal_params = np.loadtxt(os.path.join('data', 'BeattieModel_roomtemp_staircase_params.csv')).flatten().astype(np.float64)

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    axs = setup_grid(fig)

    well_fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    well_ax = fig.subplots()

    cases = ['0a', '0b', '0c']

    # Get fitting results (dict of dicts)
    results_dict = {}
    params_dfs = []
    for model in args.model_classes:
        results_dict[model] = {}
        dirnames = ['Case0a', 'Case0b', 'Case0b']
        for case, dirname in zip(cases, dirnames):
            fname = os.path.join(args.fitting_results,
                                 dirname,
                                 model,
                                 "combine_fitting_results",
                                 "combined_fitting_results.csv")

            params_df = pd.read_csv(fname)
            params_dfs.append(params_df)
            results_dict[model][case] = params_df

    protocol_dict = {}
    for protocol in np.unique(list(itertools.chain(*[list(params_df.protocol.unique()) for params_df in params_dfs])) + args.validation_protocols):
        v_func, desc = get_ramp_protocol_from_json(protocol, os.path.join(args.data_dir, 'protocols'),
                                              args.experiment_name)

        times = np.loadtxt(os.path.join(args.data_dir,
                                        f"{args.experiment_name}-{protocol}-times.csv")).astype(np.float64).flatten()

        protocol_dict[protocol] = desc, times

    model_axs, label_axs, colour_bar_ax = axs

    for i, model_class in enumerate(args.model_classes):
        for j, case in enumerate(cases):

            ax = model_axs[i, j]
            sub_df = results_dict[model_class][case]
            do_heatmap(ax, model_class, case, params_df, subtraction_df)

            for well in params_df.well.unique():
                do_heatmap(well_ax, model_class, case, sub_df,
                           subtraction_df, well=well)

                well_fig.savefig(os.path.join(output_dir,
                                              f"{model_class}_{case}_{well}"))
                well_ax.cla()


    validation_protocols = args.validation_protocols + list(param_df.protocol.unique())

    # for i, model_class in enumerate(args.model_classes):
    #     for j, case in enumerate(cases):
    #         for protocol in validation_protocols:
    #             ax = model_axs[i, j]
    #             sub = params_dfs[model_class][case]

    #             for well in params_df.well.unique():
    #                 do_heatmap(well_ax, model_class, case, sub_df, well=well)

    #                 well_fig.savefig(os.path.join(output_dir,
    #                                               f"{model_class}_{case}_{well}"))
    #                 well_ax.cla()



    fig.savefig(os.path.join(output_dir, "averaged_heatmaps.pdf"))

    plt.close(fig)


def do_spread_of_predictions(ax, model_class, fitting_case, params_df,
                             subtraction_df, validation_protocol, well=None):
    pass

def do_heatmap(ax, model_class, fitting_case, params_df, subtraction_df, well=None):

    prediction_df = compute_predictions_df(params_df, output_dir,
                                           protocol_dict, fitting_case,
                                           args.reversal, model_class=args.model_class,
                                           args=args)

    if well is not None:
        sub_df = prediction_df[prediction_df.well == well]

    else:
        # Average across wells
        sub_df = prediction_df.groupby([args.fixed_param, 'fitting_protocol', 'validation_protocol']).mean().reset_index()

    if args.vlim is None:
        vmin, vmax = sub_df['RMSE'].min(), sub_df['RMSE'].max()
    else:
        vmin, vmax = args.vlim

    cmap = sns.cm.mako_r
    norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)

    sub_df = sub_df[~sub_df.fitting_protocol.isin(args.ignore_protocols)]

    pivot_df = sub_df.pivot(columns='fitting_protocol',
                            index='validation_protocol', values='RMSE')

    hm = sns.heatmap(pivot_df, ax=ax, square=True, cbar=False, norm=norm,
                    cmap=cmap)
    hm.set_yticklabels(hm.get_yticklabels(), rotation=0)

    im = cax.imshow([[vmin, vmax]], cmap=cmap, norm=norm)
    im.set_visible(False)
    cax.plot([0], [0])


def setup_grid(fig):
    # Row for each model then colorbar
    no_rows = 5

    # Row for each 'case' and labels
    no_columns = 4

    gs = GridSpec(no_rows, no_columns, figure=fig, height_ratios=[1, 1, 1, 1, 0.15])

    current_ax = fig.add_subplot(gs[0, :])

    label_axs = [fig.add_subplot(gs[:, i]) for i in range(no_rows - 1)]

    colour_bar_ax = fig.add_subplot(gs[-1, :])

    model_axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(1, no_columns)]
                          for i in range(no_rows - 1)])

    return model_axs, label_axs, colour_bar_ax


if __name__ == "__main__":
    main()
