import argparse
import os

import itertools
import logging
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
from numba import njit
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.gridspec import GridSpec
from matplotlib.colors import SymLogNorm
from matplotlib import rc

import markovmodels
from markovmodels.model_generation import make_model_of_class
from markovmodels.fitting import get_best_params, make_prediction, adjust_kinetics, get_ensemble_of_predictions
from markovmodels.ArtefactModel import ArtefactModel
from markovmodels.utilities import setup_output_directory, get_data, get_all_wells_in_directory
from markovmodels.voltage_protocols import get_protocol_list, get_ramp_protocol_from_json, make_voltage_function_from_description
from markovmodels.voltage_protocols import remove_spikes, detect_spikes

from markovmodels.fitting import compute_predictions_df

multiprocessing_kws = {'maxtasksperchild': 1}

rc('font', **{'size': 11})
rc('figure', autolayout=True)

global model_names
model_names = {'model2': 'C-O-I',
               'model3': 'Beattie',
               'model10': 'Kemp',
               'Wang': 'Wang'}

model_colour_dict = {
    'model2': '#a6cee3',
    'Wang': '#1f78b4',
    'model10': '#b2df8a',
    'model3': '#33a02c',
}

case_colour_dict = {
    '0a':  '#1b9e77',
    '0b': '#d95f02',
    '0c': '#7570b3'
}

staircase_protocols = ['staircaseramp1', 'staircaseramp', 'staircaseramp2',
                       'staircaseramp1_2']

model_colours = sns.husl_palette(n_colors=4)

case_colours = sns.husl_palette(n_colors=3)

symlogthresh = 1

# Use perceptually-uniform diverging colour palette
global cmap
cmap = sns.color_palette("vlag", as_cmap=True)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('data_directory', help='directory where data is stored')
    parser.add_argument('fitting_results', type=str)
    parser.add_argument('subtraction_df')
    parser.add_argument('--fitting_cases', nargs='+', default=['0a', '0b', '0c',
                                                               'II', '0d'])
    parser.add_argument('--model_class', default='model3')
    parser.add_argument('--plot_all_predictions', action='store_true')
    parser.add_argument('--data_label', default='')
    parser.add_argument('--ignore_protocols', nargs='+', default=['longap'], type=str)
    parser.add_argument('--ignore_wells', nargs='+', default=['M06'], type=str)
    parser.add_argument('--vlims', nargs=2, type=float, default=(-100, 100))
    parser.add_argument('-w', '--wells', type=str, nargs='+', default=[])
    parser.add_argument('--removal_duration', type=float, default=5.0)
    parser.add_argument('--experiment_name', '-e', default='newtonrun4')
    parser.add_argument('--validation_protocols', nargs='+', default=['longap'])
    parser.add_argument('--figsize', '-f', nargs=2, type=float, default=[5.54, 6])
    parser.add_argument('--fig_title', '-t', default='')
    parser.add_argument('--nolegend', action='store_true')
    parser.add_argument('--dpi', '-d', default=500, type=int)
    parser.add_argument('--fontsize', type=int, default=12)
    parser.add_argument('--show_uncertainty', action='store_true')
    parser.add_argument('--shared_plot_limits', action='store_true')
    parser.add_argument('--file_format', default='')
    parser.add_argument('--reversal', default=-91.71, type=float)
    parser.add_argument('--output', '-o')
    parser.add_argument('--no_cpus', '-c', default=1, type=int)

    global args
    args = parser.parse_args()

    case = '0b'

    if args.fontsize:
        matplotlib.rcParams.update({'font.size': args.fontsize})

    data_label = ''

    global output_dir
    output_dir = setup_output_directory(args.output, 'prediction_comparison')

    fname = os.path.join(args.fitting_results,
                         'Case0b',
                         args.model_class,
                         "combine_fitting_results",
                         "combined_fitting_results.csv")

    params_df = pd.read_csv(fname)
    subtraction_df = pd.read_csv(args.subtraction_df)

    protocol_dict = {}
    v_func = None
    sweep = 0
    # prediction_df = compute_predictions_df(params_df, output_dir,
    #                                        protocol_dict, case,
    #                                        args.reversal, subtraction_df,
    #                                        model_class=args.model_class,
    #                                        args=args,
    #                                        label=f"{args.model_class}_{case}_predictions",
    #                                        data_label=data_label,
    #                                        hybrid=False,
    #                                        strict=False,
    #                                        )

    # best_well = prediction_df.groupby('well').agg(agg_dict).idxmin()['n_score']
    best_well = 'B20'

    cases = ['0a', '0b', '0c']
    dirnames_dict = { '0a': 'Case0a',
                      '0b': 'Case0b',
                      '0c': 'Case0b'}

    dirnames = [dirnames_dict[case] for case in cases]

    model_classes = ['model2', 'model3', 'model10', 'Wang']
    # Get fitting results (dict of dicts)
    results_dict = {}
    params_dfs = []
    params_df_dict = {}
    for model in model_classes:
        results_dict[model] = {}
        for case, dirname in zip(cases, dirnames):
            fname = os.path.join(args.fitting_results,
                                 dirname,
                                 model,
                                 "combine_fitting_results",
                                 "combined_fitting_results.csv")

            params_df = pd.read_csv(fname)

            # if args.protocols:
            #     params_df = params_df[params_df.protocol.isin(args.protocols)].copy()

            # if args.wells:
            #     params_df = params_df[params_df.well.isin(args.wells)].copy()

            params_df = params_df[~params_df.well.isin(args.ignore_wells)].copy()

            params_df['protocol'] = ['staircaseramp1_2' if protocol ==
                                     'staircaseramp2' else protocol for
                                     protocol in params_df.protocol]

            params_dfs.append(params_df)
            params_df_dict[(model, case)] = params_df


    protocols = list(np.unique(list(itertools.chain(*[list(params_df.protocol.unique()) for params_df in params_dfs])))) + args.validation_protocols

    for protocol in protocols:
        v_func, desc = get_ramp_protocol_from_json(protocol, os.path.join(args.data_directory, 'protocols'),
                                              args.experiment_name)

        times = np.loadtxt(os.path.join(args.data_directory,
                                        f"{args.experiment_name}-{protocol}-times.csv")).astype(np.float64).flatten()

        protocol_dict[protocol] = desc, times

    fig = plt.figure(figsize=args.figsize)
    axs = fig.subplots(3, 2, sharex=True, height_ratios=[0.3, 1, 1])


    # Staircase protocol, predictions from each model
    protocol = 'staircaseramp1'
    sweep = 0

    v_func, desc = get_ramp_protocol_from_json(protocol,
                                               os.path.join(args.data_directory, 'protocols'),
                                               args.experiment_name)
    best_data, vp = get_data(best_well, protocol,
                             args.data_directory, args.experiment_name, sweep=sweep)

    times = np.loadtxt(os.path.join(args.data_directory,
                                    f"{args.experiment_name}-{protocol}-times.csv")).astype(np.float64).flatten()

    Vcmd = np.array([v_func(t) for t in times])
    axs[0, 0].plot(times, Vcmd, color='black')
    axs[0, 0].plot(times, best_data, color='grey', alpha=.25)

    axs[0, 0].set_title(r'$d_1^{(1)}$')

    for ax, cap in zip(axs.flatten(), ['a', 'b', 'c', 'd', 'e', 'f']):
        ax.set_title(cap, loc='left', fontweight='bold')


    model_label_dict = {
        'model2': 'C-O-I',
        'model3': 'Beattie',
        'model10': 'Kemp',
        'Wang': 'Wang'
    }

    case = '0b'
    for model_class in model_classes:
        params_df = params_df_dict[(model_class, case)]
        print(params_df.columns)
        pred = make_prediction(model_class, args, best_well,
                               protocol, sweep,
                               protocol, sweep, params_df,
                               subtraction_df, case,
                               args.reversal, protocol_dict,
                               best_data, Vcmd,
                               label=data_label)

        axs[1, 0].plot(times, pred, label=model_label_dict[model])

    axs[1, 0].legend()


    # Model 3 prediction of all wells
    params_df = params_df_dict[('model3', '0b')]

    for well in params_df.well.unique():
        data, vp = get_data(best_well, protocol,
                            args.data_directory, args.experiment_name, sweep=sweep)

        pred = make_prediction(model_class, args, best_well,
                               protocol, sweep,
                               protocol, sweep, params_df,
                               subtraction_df, case,
                               args.reversal, protocol_dict,
                               data, Vcmd,
                               label=data_label)
        axs[2, 0].plot(times, pred, label='well')

    axs[2, 0].legend(ncol=5)

    fig.savefig(os.path.join(output_dir, "prediction_comparison"))


if __name__ == '__main__':
    main()
