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

from matplotlib import rc

import markovmodels
from markovmodels.model_generation import make_model_of_class
from markovmodels.fitting import get_best_params, compute_predictions_df, make_prediction
from markovmodels.ArtefactModel import ArtefactModel
from markovmodels.utilities import setup_output_directory, get_data, get_all_wells_in_directory
from markovmodels.voltage_protocols import get_protocol_list, get_ramp_protocol_from_json, make_voltage_function_from_description
from markovmodels.voltage_protocols import remove_spikes, detect_spikes

multiprocessing_kws = {'maxtasksperchild': 1}

plt.rcParams["axes.formatter.use_mathtext"] = True

rc('font', **{'size': 12})
# rc('text', usetex=True)
# rc('figure', dpi=400, facecolor=[0]*4)
# rc('axes', facecolor=[0]*4)
# rc('savefig', facecolor=[0]*4)
rc('figure', autolayout=True)

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('data_directory', help='directory where data is stored')
    parser.add_argument('fitting_results', type=str)
    parser.add_argument('subtraction_df')
    parser.add_argument('chrono_file')
    parser.add_argument('--prediciton_protocols', nargs='+', default=['longap', 'staircaseramp1'])
    parser.add_argument('--use_fake_results', action='store_true')
    parser.add_argument('--ignore_protocols', nargs='+', default=['longap'], type=str)
    parser.add_argument('-w', '--wells', type=str, nargs='+')
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
    parser.add_argument('--output', '-o')
    parser.add_argument('--no_cpus', '-c', default=1, type=int)

    global args
    args = parser.parse_args()

    args.model_classes = ['model2', 'model3', 'model10', 'Wang']

    global output_dir
    output_dir = setup_output_directory(args.output, 'chapter_4_heatmaps')

    if args.fontsize:
        matplotlib.rcParams.update({'font.size': args.fontsize})

    infer_reversal_params = np.loadtxt(os.path.join('data', 'BeattieModel_roomtemp_staircase_params.csv')).flatten().astype(np.float64)

    subtraction_df = pd.read_csv(args.subtraction_df)

    cases = ['0c', '0b', '0a']

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

            params_df = get_best_params(pd.read_csv(fname))

            if args.wells:
                params_df = params_df[params_df.well.isin(args.wells)].copy()

            params_df['protocol'] = ['staircaseramp1_2' if protocol ==
                                     'staircaseramp2' else protocol for
                                     protocol in params_df.protocol]

            params_dfs.append(params_df)
            results_dict[model][case] = params_df

    protocol_dict = {}
    for protocol in np.unique(list(itertools.chain(*[list(params_df.protocol.unique()) for params_df in params_dfs])) + args.validation_protocols):
        v_func, desc = get_ramp_protocol_from_json(protocol, os.path.join(args.data_directory, 'protocols'),
                                              args.experiment_name)

        times = np.loadtxt(os.path.join(args.data_directory,
                                        f"{args.experiment_name}-{protocol}-times.csv")).astype(np.float64).flatten()

        protocol_dict[protocol] = desc, times


    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    axs = setup_grid(fig)
    model_axs, model_label_axs, case_label_axs, colour_bar_ax = axs

    tasks = []
    for i, model_class in enumerate(args.model_classes):
        for j, case in enumerate(cases):
            sub_df = results_dict[model_class][case]
            tasks.append([model_class, case, sub_df, args, output_dir,
                          protocol_dict, case])

    cbar_kws = {
        'orientation': 'horizontal',
        'fraction': .75,
        'drawedges': False,
        'label': 'RMSE',
    }

    fig = plt.figure(figsize=args.figsize)
    ax = fig.subplots()

    wells = results_dict[args.model_classes[0]][cases[0]]

    if args.wells:
        [w for w in wells if w in args.wells]

    # Compare cases
    for protocol in args.validation_protocols:
        for well in wells:
            for model_class in args.model_classes:
                for case in cases:
                    do_spread_of_predictions(ax, model_class, case,
                                             results_dict[model_class][case]
                                             params_df, subtraction_df,
                                             protocol, well, protocol_dict,
                                             args, line_colour=colours[0])

            # Plot data
            # TODO

            fig.savefig(f"{well}_{model_class}_{protocol}_cases_sop.png")
            ax.cla()


    # Compare models
    for protocol in args.validation_protocols:
        for well in wells:
            for case in cases:
                for model_class in args.model_classes:
                    do_spread_of_predictions(ax, model_class, case,
                                             results_dict[model_class][case]
                                             params_df, subtraction_df,
                                             protocol, well, protocol_dict,
                                             args, line_colour=colours[0])

            # Plot data
            # TODO

            fig.savefig(f"{well}_{model_class}_{case}_protocols_sop.png")
            ax.cla()


def do_spread_of_predictions(ax, model_class, fitting_case, params_df,
                             subtraction_df, validation_protocol, well,
                             protocol_dict, args, line_colour='red', plot_kws):

    params_df = params_df[(params_df.well == well)
                          & (params_df.protocol == fitting_protocol)]

    # Ensure we have one parameter set per sweep (the one with the best score)
    params_df = get_best_params(params_df)

    predictions = []

    model = make_model_of_class(model_class)
    param_labels = model.get_parameter_labels()

    solver = model.make_hybrid_solver_current(njitted=True,
                                              hybrid=False)

    voltage_func = model.voltage

    for _, row in params_df.iterrows():
        protocol = row['protocol']
        sweep = row['sweep']

        data_label = '' if fitting_case in ['I', 'II'] else ''

        data, _ = get_data(well, protocol, args.data_directory,
                            args.experiment_name, label=data_label,
                            sweep=sweep)

        desc, times = protocol_dict[protocol]
        voltages = np.array([voltage_func(t, protocol_description=desc) for t in times])
        pred = make_prediction(model_class, args, well, validation_protocol, 0,
                               protocol, sweep, params_df, subtraction_df,
                               fitting_case, args.reversal, protocol_dict,
                               data, voltages, solver=solver)

        predictions.append(pred)

    predictions = np.vstack(predictions)

    ax.plot(predictions.max(axis=0), ls='--', color=line_colour)
    ax.plot(predictions.min(axis=0), ls='--', color=line_colour)

    ax.axvspan(predictions.min(axis=0), predictions.max(axis=0),
               color=line_colour, alpha=.25)



