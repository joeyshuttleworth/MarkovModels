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
from markovmodels.fitting import get_best_params, compute_predictions_df, make_prediction, adjust_kinetics
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

model_colours = sns.husl_palette(n_colors=4)

case_colours = sns.husl_palette(n_colors=3)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('data_directory', help='directory where data is stored')
    parser.add_argument('fitting_results', type=str)
    parser.add_argument('subtraction_df')
    parser.add_argument('chrono_file')
    parser.add_argument('--cases', nargs='+', default=['0a', '0b', '0c'])
    parser.add_argument('--model_classes', nargs='+',
                        default=['model2', 'model3', 'Wang', 'model10'])
    parser.add_argument('--plot_all_predictions', action='store_true')
    parser.add_argument('--data_label', default='')
    parser.add_argument('--ignore_protocols', nargs='+', default=['longap'], type=str)
    parser.add_argument('-w', '--wells', type=str, nargs='+')
    parser.add_argument('--removal_duration', type=float, default=5.0)
    parser.add_argument('--experiment_name', '-e', default='newtonrun4')
    parser.add_argument('--validation_protocols', nargs='+')
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

    # args.model_classes = ['model3', 'model10']

    global output_dir
    output_dir = setup_output_directory(args.output, 'chapter_4_sop')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.fontsize:
        matplotlib.rcParams.update({'font.size': args.fontsize})

    infer_reversal_params = np.loadtxt(os.path.join('data', 'BeattieModel_roomtemp_staircase_params.csv')).flatten().astype(np.float64)

    subtraction_df = pd.read_csv(args.subtraction_df)

    if not args.validation_protocols:
        args.validation_protocols = list(subtraction_df.protocol.unique())

    cases = ['0a', '0b', '0c']

    global case_relabel_dict
    case_relabel_dict = {
        '0a': 'Case I',
        '0b': 'Case II',
        '0c': 'Case III'
    }

    dirnames = ['Case0a', 'Case0b', 'Case0b']

    # Get fitting results (dict of dicts)
    results_dict = {}
    params_dfs = []
    for model in args.model_classes:
        results_dict[model] = {}
        for case, dirname in zip(cases, dirnames):
            if case not in args.cases:
                continue
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

            print('protocols: ', params_df.protocol.unique())

            params_dfs.append(params_df)
            results_dict[model][case] = params_df

    protocol_dict = {}
    for protocol in list(np.unique(list(itertools.chain(*[list(params_df.protocol.unique()) for params_df in params_dfs])))) + args.validation_protocols:
        v_func, desc = get_ramp_protocol_from_json(protocol, os.path.join(args.data_directory, 'protocols'),
                                              args.experiment_name)

        times = np.loadtxt(os.path.join(args.data_directory,
                                        f"{args.experiment_name}-{protocol}-times.csv")).astype(np.float64).flatten()

        protocol_dict[protocol] = desc, times

    voltage_func = make_model_of_class(args.model_classes[0]).voltage
    wells = results_dict[args.model_classes[0]][cases[0]].well.unique()

    if args.wells:
        [w for w in wells if w in args.wells]


    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    axs = fig.subplots(3, height_ratios=[1, 1, .25],
                       sharex=True)

    desc, times = protocol_dict[protocol]

    for ax in axs:
        ax.spines[['top', 'right']].set_visible(False)
    # Compare cases 0a, 0b for models 2, 3 (longap)
    prediction_protocol = 'longap'
    sweep = 0
    well = 'B09'

    for model_class, ax in zip(['model2', 'model3'], axs):
        desc, times = protocol_dict[prediction_protocol]
        voltages = np.array([voltage_func(t, protocol_description=desc) for t in times])

        ax.set_title(model_names[model_class])
        desc, times = protocol_dict[prediction_protocol]
        data, _ = get_data(well, prediction_protocol, args.data_directory,
                                           args.experiment_name, label=args.data_label,
                                           sweep=sweep)
        ax.plot(times, data, color='grey', label=args.data_label, alpha=.3)

        for case in ['0a', '0b']:
            params_df = results_dict[model_class][case].copy()
            do_spread_of_predictions(ax, model_class, case, params_df,
                                     subtraction_df, prediction_protocol, well, sweep,
                                     protocol_dict, args,
                                     line_colour=case_colour_dict[case],
                                     label=f"{case_relabel_dict[case]}",
                                     voltage_func=voltage_func)

    axs[0].legend()
    axs[2].plot(times, voltages, color='black', lw=.5)
    axs[0].set_ylabel(r'$I_\text{subtracted}$ (pA)')
    axs[1].set_ylabel(r'$I_\text{subtracted}$ (pA)')

    axs[2].set_ylabel(r'$V_\text{cmd}$ (mV)')
    axs[2].set_xlabel(r'$t$ (ms)')

    fig.savefig(os.path.join(output_dir, 'thesis_plots'
                             f"{well}_{model_class}_{prediction_protocol}_cases_I_II_models_2_3.png"))

    for ax in axs:
        ax.cla()

    # Compare cases 0a, 0b for models 2, 3 (longap)
    prediction_protocol = 'staircaseramp1'
    sweep = 0

    for model_class, ax in zip(['model2', 'model3'], axs):
        desc, times = protocol_dict[prediction_protocol]
        voltages = np.array([voltage_func(t, protocol_description=desc) for t in times])
        data, _ = get_data(well, prediction_protocol, args.data_directory,
                                           args.experiment_name, label=args.data_label,
                                           sweep=sweep)
        ax.plot(times, data, color='grey', label=args.data_label, alpha=.3)

        ax.set_title(model_names[model_class])

        for case in ['0a', '0b']:
            params_df = results_dict[model_class][case].copy()
            do_spread_of_predictions(ax, model_class, case, params_df,
                                     subtraction_df, prediction_protocol, well, sweep,
                                     protocol_dict, args,
                                     line_colour=case_colour_dict[case],
                                     label=f"{case_relabel_dict[case]}",
                                     voltage_func=voltage_func)
        ax.legend()

    axs[0].legend()
    axs[2].plot(times, voltages, color='black', lw=.5)
    axs[0].set_ylabel(r'$I_\text{subtracted}$ (pA)')
    axs[1].set_ylabel(r'$I_\text{subtracted}$ (pA)')

    axs[2].set_ylabel(r'$V_\text{cmd}$ (mV)')
    axs[2].set_xlabel(r'$t$ (ms)')


    fig.savefig(os.path.join(output_dir, 'thesis_plots'
                             f"{well}_{model_class}_{prediction_protocol}_cases_I_II_models_2_3.png"))

    fig.clf()
    current_ax, voltage_ax = setup_axes(fig)
    # Compare all cases
    for protocol in args.validation_protocols:
        desc, times = protocol_dict[protocol]
        voltages = np.array([voltage_func(t, protocol_description=desc) for t in times])
        for well in wells:
            sweeps = params_dfs[0].sweep.unique()
            for sweep in sweeps:
                for model_class in args.model_classes:
                    i = 0
                    for case in cases:
                        if case not in args.cases:
                            continue
                        params_df = results_dict[model_class][case].copy()
                        params_df = params_df[params_df.well == well]
                        if len(params_df.index) == 0:
                            logging.warning(f"{protocol} {well} {sweep} {model_class} {case}: empty dataframe")
                            continue

                        plotted = do_spread_of_predictions(current_ax, model_class,
                                                           case,
                                                           params_df,
                                                           subtraction_df, protocol, well,
                                                           sweep, protocol_dict, args,
                                                           line_colour=case_colour_dict[case],
                                                           label=f"{case_relabel_dict[case]}",
                                                           voltage_func=voltage_func)
                        voltage_ax.plot(times, voltages, color='black', lw=.3)
                        i += 1

                    # Plot ata
                    # TODO
                    if plotted:
                        data, _ = get_data(well, protocol, args.data_directory,
                                           args.experiment_name, label=args.data_label,
                                           sweep=sweep)
                        current_ax.plot(times, data, color='grey', label=well, alpha=.3)
                        current_ax.legend()

                        fig.savefig(os.path.join(output_dir, 'thesis_plots'
                                                 f"{well}_{model_class}_{prediction_protocol}_cases_I_II_models_2_3.png"))
                        current_ax.cla()
                    current_ax, voltage_ax = setup_axes(fig)

    # Compare all models
    for protocol in args.validation_protocols:
        desc, times = protocol_dict[prediction_protocol]
        voltages = np.array([voltage_func(t, protocol_description=desc) for t in times])
        for well in wells:
            sweeps = params_dfs[0].sweep.unique()
            for sweep in sweeps:
                for case in cases:
                    if case not in args.cases:
                        continue
                    i = 0
                    for model_class in args.model_classes:
                        params_df = results_dict[model_class][case].copy()
                        params_df = params_df[params_df.well == well]
                        if len(params_df.index) == 0:
                            logging.warning(f"{protocol} {well} {sweep} {model_class} {case}: empty dataframe")
                            continue

                        plotted = do_spread_of_predictions(current_ax, model_class,
                                                           case, params_df,
                                                           subtraction_df,
                                                           protocol, well,
                                                           sweep,
                                                           protocol_dict, args,
                                                           line_colour=model_colour_dict[model_class],
                                                           label=model_names[model_class],
                                                           voltage_func=voltage_func)
                        voltage_ax.plot(times, voltages, color='black', lw=.3)
                        i += 1

                    # Plot data
                    # TODO
                    if plotted:
                        times = np.loadtxt(os.path.join(args.data_directory,
                                                        f"{args.experiment_name}-{protocol}-times.csv")).astype(np.float64).flatten()
                        data, _ = get_data(well, protocol, args.data_directory,
                                           args.experiment_name,
                                           label=args.data_label, sweep=sweep)
                        current_ax.plot(times, data, color='grey', label=well, alpha=.3)
                        current_ax.legend()

                        fig.savefig(os.path.join(output_dir,
                                                 f"{well}_{case_relabel_dict[case]}_sweep{sweep}_{protocol}_sop.png"))
                    current_ax, voltage_ax = setup_axes(fig)


def do_spread_of_predictions(ax, model_class, fitting_case, params_df,
                             subtraction_df, validation_protocol, well, sweep,
                             protocol_dict, args, line_colour='red',
                             label=None, plot_kws={}, voltage_func=None):
    if (well, validation_protocol, sweep) not in \
       subtraction_df.set_index(['well', 'protocol', 'sweep']).index:
        return False

    params_df = params_df[params_df.well == well]

    new_E_rev = subtraction_df.set_index(['well', 'protocol', 'sweep']).loc[well, validation_protocol, sweep]['E_rev']

    if fitting_case == '0c':
        params_df = adjust_kinetics(model_class, params_df, subtraction_df, args.reversal,
                                    new_E_rev=new_E_rev)

    model = make_model_of_class(model_class, voltage=voltage_func)
    param_labels = model.get_parameter_labels()

    solver = model.make_hybrid_solver_current(njitted=True,
                                              hybrid=False,
                                              strict=False)

    voltage_func = model.voltage

    data, vp = get_data(well, validation_protocol, args.data_directory,
                       args.experiment_name, label=args.data_label,
                       sweep=sweep)

    desc, times = protocol_dict[validation_protocol]
    desc = np.vstack((desc, [[desc[-1, 1], np.inf, -80.0, -80.0]]))

    voltages = np.array([voltage_func(t, protocol_description=desc) for t in times])

    predictions = []
    for _, row in params_df.iterrows():
        protocol = row['protocol']
        fit_sweep = row['sweep']

        pred = make_prediction(model_class, args, well, validation_protocol, sweep,
                               protocol, fit_sweep, params_df, subtraction_df,
                               fitting_case, args.reversal, protocol_dict,
                               data, voltages, solver=solver, do_spike_removal=False)

        predictions.append(pred)

    predictions = np.vstack(predictions)

    if not args.plot_all_predictions:
        ax.plot(times, predictions.max(axis=0), lw=.3, color=line_colour)
        ax.plot(times, predictions.min(axis=0), lw=.3, color=line_colour)
        ax.fill_between(times, predictions.min(axis=0), predictions.max(axis=0),
                        color=line_colour, alpha=.1, label=label)

    if args.plot_all_predictions:
        for row in predictions:
            ax.plot(times, row, lw=.3, color=line_colour)

    return True


def setup_axes(fig):
    fig.clf()
    axs = fig.subplots(2, sharex=True, height_ratios=[1, .3])
    spines = ['top', 'right']

    for ax in axs:
        ax.spines[spines].set_visible(False)

    axs[1].set_xlabel('$t$ (ms)')
    axs[1].set_ylabel(r'$V_\text{cmd}$ (mV)')
    axs[0].set_ylabel(r'$I_\text{Kr}$ (nA)')

    return axs

if __name__ == '__main__':
    main()
