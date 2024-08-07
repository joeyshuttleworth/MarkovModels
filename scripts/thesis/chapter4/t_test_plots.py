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
    parser.add_argument('chrono_file')
    parser.add_argument('--fitting_cases', nargs='+', default=['0a', '0b', '0c',
                                                               'II', '0d'])
    parser.add_argument('--model_classes',
                        default=['model2', 'model3', 'model10', 'Wang'],
                        nargs='+')
    parser.add_argument('--plot_all_predictions', action='store_true')
    parser.add_argument('--data_label', default='')
    parser.add_argument('--ignore_protocols', nargs='+', default=['longap'], type=str)
    parser.add_argument('--ignore_wells', nargs='+', default=['M06'], type=str)
    parser.add_argument('--vlims', nargs=2, type=float, default=(-100, 100))
    parser.add_argument('-w', '--wells', type=str, nargs='+', default=[])
    parser.add_argument('--removal_duration', type=float, default=5.0)
    parser.add_argument('--experiment_name', '-e', default='newtonrun4')
    parser.add_argument('--validation_protocols', nargs='+')
    parser.add_argument('--figsize', '-f', nargs=2, type=float, default=[5.54, 6])
    parser.add_argument('--fig_title', '-t', default='')
    parser.add_argument('--nolegend', action='store_true')
    parser.add_argument('--dpi', '-d', default=500, type=int)
    parser.add_argument('--fontsize', type=int, default=8)
    parser.add_argument('--show_uncertainty', action='store_true')
    parser.add_argument('--shared_plot_limits', action='store_true')
    parser.add_argument('--file_format', default='')
    parser.add_argument('--reversal', default=-91.71, type=float)
    parser.add_argument('--output', '-o')
    parser.add_argument('--no_cpus', '-c', default=1, type=int)

    global args
    args = parser.parse_args()

    if args.fontsize:
        matplotlib.rcParams.update({'font.size': args.fontsize})

    global output_dir
    output_dir = setup_output_directory(args.output, 'chapter_4_t_test_plots')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    infer_reversal_params = np.loadtxt(os.path.join('data', 'BeattieModel_roomtemp_staircase_params.csv')).flatten().astype(np.float64)

    subtraction_df = pd.read_csv(args.subtraction_df)

    if not args.validation_protocols:
        args.validation_protocols = list(subtraction_df.protocol.unique())

    cases = ['0a', '0b', '0c', '0d', 'II']
    dirnames = ['Case0a', 'Case0b', 'Case0b', 'Case0d', 'CaseII']

    global case_relabel_dict
    case_relabel_dict = {
        '0a': 'Case I',
        '0b': 'Case II',
        '0c': 'Case III'
    }

    with open(args.chrono_file, 'r') as fin:
        lines = fin.read().splitlines()
        protocol_order = [line.split(' ')[0] for line in lines]

    global relabel_dict
    relabel_dict = {p: r"$d_{" f"{i+1}" r"}$" for i, p
                    in enumerate(protocol_order)}

    # Get fitting results (dict of dicts)
    params_dfs = []
    results_dict = {}

    for model_class in args.model_classes:
        for case, dirname in zip(cases, dirnames):
            if case not in args.fitting_cases:
                continue
            fname = os.path.join(args.fitting_results,
                                    dirname,
                                    model_class,
                                    "combine_fitting_results",
                                    "combined_fitting_results.csv")

            params_df = get_best_params(pd.read_csv(fname))

            if args.wells:
                params_df = params_df[params_df.well.isin(args.wells)].copy()

            params_df['protocol'] = ['staircaseramp1_2' if protocol ==
                                        'staircaseramp2' else protocol for
                                        protocol in params_df.protocol]

            params_dfs.append(params_df)
            results_dict[(model_class, case)] = params_df

    protocol_dict = {}
    v_func = None
    sweep = 0

    protocols = list(np.unique(list(itertools.chain(*[list(params_df.protocol.unique()) for params_df in params_dfs])))) + args.validation_protocols

    for protocol in protocols:
        v_func, desc = get_ramp_protocol_from_json(protocol, os.path.join(args.data_directory, 'protocols'),
                                              args.experiment_name)

        times = np.loadtxt(os.path.join(args.data_directory,
                                        f"{args.experiment_name}-{protocol}-times.csv")).astype(np.float64).flatten()

        protocol_dict[protocol] = desc, times

    for fitting_case in args.fitting_cases:
        for model_class in args.model_classes:
            print(f"plotting {fitting_case} {model_class}")

            params_df = results_dict[(model_class, fitting_case)]

            plot_fitting_z_scores(sweep, fitting_case, params_df, protocols,
                                  protocol_dict, protocol_order, results_dict,
                                  subtraction_df, model_class, mode='prediction',
                                  v_func=v_func)

            plot_fitting_z_scores(sweep, fitting_case, params_df, protocols,
                                  protocol_dict, protocol_order, results_dict,
                                  subtraction_df, model_class, mode='fitting',
                                  v_func=v_func)


def setup_axes(fig, no_protocols):
    fig.clf()

    no_rows = int(no_protocols / 2) + 1
    no_columns = 2

    gs = GridSpec(no_rows, no_columns, figure=fig, height_ratios=[1] * (no_rows - 1) + [0.2])

    axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(no_columns)] for i in range(no_rows - 1)])
    cbar_ax = fig.add_subplot(gs[-1, :])

    spines = ['top', 'right']

    for ax in axs.flatten():
        ax.spines[spines].set_visible(False)

    for ax in axs[-1, :]:
        ax.set_xlabel(r'$t$')
        ax.set_xticks([0, 1])
        ax.set_xticklabels([0, r'$t_\text{end}$'])

    for ax in axs[:, 0]:
        ax.set_ylabel(r'$V$ (mV)')

    return axs.flatten(), cbar_ax


def plot_fitting_z_scores(sweep, fitting_case, params_df, protocols,
                          protocol_dict, protocol_order, results_dict,
                          subtraction_df, model_class,
                          mode='prediction', v_func=None):

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    protocols = params_df.protocol.unique()
    no_protocols = len(protocols)
    axs, cbar_ax = setup_axes(fig, no_protocols)
    V_range = (-120, 60)

    wells = np.array([w for w in params_df.well.unique() if w not in args.ignore_wells])
    zs = {}

    v_func = make_voltage_function_from_description()

    max_z, min_z = -np.inf, np.inf
    for well in wells:
        if well in args.ignore_wells:
            continue
        zs[well] = {}
        axs, cbar_ax = setup_axes(fig, no_protocols)
        for ax, protocol in zip(axs, protocol_order):
            desc, times = protocol_dict[protocol]
            voltages = np.array([v_func(t, protocol_description=desc) for t in times])

            spike_times, spike_indices = \
            markovmodels.voltage_protocols.detect_spikes(times, voltages,
                                                         threshold=10)
            _, _, indices = \
            markovmodels.voltage_protocols.remove_spikes(times,
                                                         voltages, spike_times,
                                                         time_to_remove=args.removal_duration)
            params_df = results_dict[(model_class, fitting_case)]
            sub_df = params_df[params_df.protocol != protocol]

            z = get_t_test_statistic(model_class, fitting_case, params_df,
                                     subtraction_df, protocol, well, sweep,
                                     protocol_dict, args, mode=mode,
                                     voltage_func=v_func
                                     )
            zs[well][protocol] = z
            if np.any(z):
                max_z = max(z[indices].max(), max_z)
                min_z = min(z[indices].min(), min_z)

    # vmin = -np.max(np.abs([min_z, max_z]))
    # vmax = +np.max(np.abs([min_z, max_z]))

    vmin, vmax = args.vlims

    for well in wells:
        if well in args.ignore_wells:
            continue

        axs, cbar_ax = setup_axes(fig, no_protocols)
        for ax, protocol in zip(axs, protocol_order):
            desc, times = protocol_dict[protocol]
            voltages = np.array([v_func(t, protocol_description=desc) for t in times])
            xmin, xmax = (0, 1)
            ymin, ymax = V_range
            ax.plot(times/times.max(), voltages, color='black')

            if protocol not in zs[well]:
                ax.set_facecolor((105/256, 105/256, 105/256, .5))
                continue

            z = zs[well][protocol]
            if not np.any(z):
                ax.set_facecolor((105/256, 105/256, 105/256, .5))
                continue

            if np.all(np.isfinite(z)):
                X = z[None, :].astype(np.float64)
                im = ax.imshow(X, extent=(xmin, xmax, ymin, ymax), alpha=1,
                            aspect='auto', norm=SymLogNorm(symlogthresh, vmin=vmin, vmax=vmax),
                            cmap=cmap)

                # ax.plot(times[indices]/times.max(), z[indices], label=model_names[model_class],
                #         color=model_colour_dict[model_class], alpha=.5)
            else:
                # Grey out axes with no values
                ax.set_facecolor((105/256, 105/256, 105/256, .5))

            ax.set_title(relabel_dict[protocol])
        if mode == 'prediction':
            label = r'$Z_\text{T}$'
        else:
            label = r'$\frac{1}{\hat\sigma}(y_i - z_i)$'
        fig.colorbar(im, cax=cbar_ax, shrink=.75, orientation='horizontal',
                     label=label)
        fig.savefig(os.path.join(output_dir,
                                 f"{well}_sweep{sweep}_t_scores_{model_class}_{fitting_case}_{mode}"))
        for ax in axs:
            ax.cla()
        cbar_ax.cla()

    fig.clf()

    axs, cbar_ax = setup_axes(fig, no_protocols)
    for ax, protocol in zip(axs, protocol_order):
        desc, times = protocol_dict[protocol]
        voltages = np.array([v_func(t, protocol_description=desc) for t in times])
        spike_times, spike_indices = \
        markovmodels.voltage_protocols.detect_spikes(times, voltages,
                                                     threshold=10)
        _, _, indices = \
            markovmodels.voltage_protocols.remove_spikes(times, voltages, spike_times,
                                                         args.removal_duration)
        ax.plot(times/times.max(), voltages, color='black')

        w = wells[0]
        if protocol not in zs[w]:
            ax.set_facecolor((105/256, 105/256, 105/256, .5))
            continue
        if not np.any(zs[w][protocol]):
            ax.set_facecolor((105/256, 105/256, 105/256, .5))
            continue

        z = np.vstack([zs[w][protocol] for w in wells if np.any(zs[w][protocol])]).mean(axis=0)

        if np.all(np.isfinite(z)):
            X = z[None, :]
            # ax.plot(times[indices]/times.max(), z[indices], label=model_names[model_class],
            #         color=model_colour_dict[model_class], alpha=.5)
            im = ax.imshow(X, extent=(xmin, xmax, ymin, ymax), alpha=1,
                           aspect='auto', norm=SymLogNorm(symlogthresh, vmin=vmin,
                                                          vmax=vmax),
                           cmap=cmap, interpolation=None)
        else:
            # Grey out axes with no values
            ax.set_facecolor((105/256, 105/256, 105/256, .5))
        ax.set_title(relabel_dict[protocol])

    fig.colorbar(im, cax=cbar_ax, shrink=.75, orientation='horizontal',
                 norm=SymLogNorm(symlogthresh, vmin=vmin, vmax=vmax),
                 label=label)
    fig.savefig(os.path.join(output_dir,
                                f"average_sweep{sweep}_t_scores_{model_class}_{fitting_case}_{mode}"))
    plt.close(fig)


def get_t_test_statistic(model_class, fitting_case, params_df,
                         subtraction_df, validation_protocol, well, sweep,
                         protocol_dict, args,
                         voltage_func=None, mode='prediction'):

    if (well, validation_protocol, sweep) not in \
       subtraction_df.set_index(['well', 'protocol', 'sweep']).index:
        return False

    if mode == 'fitting' and (well, validation_protocol, sweep) not in \
       params_df.set_index(['well', 'protocol', 'sweep']).index:
        return False

    params_df = params_df[params_df.well == well]

    new_E_rev = subtraction_df.set_index(['well', 'protocol', 'sweep']).loc[well, validation_protocol, sweep]['E_rev']

    if fitting_case == '0c':
        params_df = adjust_kinetics(model_class, params_df, subtraction_df, args.reversal,
                                    new_E_rev=new_E_rev)

    model = make_model_of_class(model_class, voltage=voltage_func)

    if fitting_case in ['I', 'II']:
        model = ArtefactModel(model)

    solver = model.make_hybrid_solver_current(njitted=False,
                                              hybrid=False,
                                              strict=False)

    data, vp = get_data(well, validation_protocol, args.data_directory,
                       args.experiment_name, label=args.data_label,
                       sweep=sweep)

    desc, times = protocol_dict[validation_protocol]
    desc = np.vstack((desc, [[desc[-1, 1], np.inf, -80.0, -80.0]]))
    voltages = np.array([voltage_func(t, protocol_description=desc) for t in times])

    if validation_protocol in staircase_protocols:
        disallowed_protocols = staircase_protocols
    else:
        disallowed_protocols = [validation_protocol]

    # sub_df = params_df[~params_df.protocol.isin(disallowed_protocols)]

    noise = data[:200].std(ddof=1)

    if mode == 'prediction':
        predictions = get_ensemble_of_predictions(times, desc, params_df,
                                                  validation_protocol, well, sweep,
                                                  subtraction_df, fitting_case,
                                                  args.reversal, model_class, data,
                                                  args, protocol_dict,
                                                  solver=solver,
                                                  voltage_func=voltage_func,
                                                  ignore_fitted=True)
        # T-test statistic with null hypothesis being 0 error
        residuals = predictions - data[None, :]
        z = residuals.mean(axis=0) / (noise + residuals.std(axis=0, ddof=1) / residuals.shape[0])

    elif mode == 'fitting':
        pred = make_prediction(model_class, args, well, validation_protocol, sweep,
                               validation_protocol, sweep, params_df, subtraction_df,
                               fitting_case, args.reversal, protocol_dict,
                               data, voltages, solver=solver, do_spike_removal=False)
        residuals = (pred - data).flatten()
        z = residuals / noise
    else:
        raise ValueError()

    return z


def setup_axes(fig, no_protocols):
    fig.clf()

    no_rows = int(no_protocols / 2) + 1
    no_columns = 2

    gs = GridSpec(no_rows, no_columns, figure=fig, height_ratios=[1] * (no_rows - 1) + [0.2])

    axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(no_columns)] for i in range(no_rows - 1)])
    cbar_ax = fig.add_subplot(gs[-1, :])

    spines = ['top', 'right']

    for ax in axs.flatten():
        ax.spines[spines].set_visible(False)

    for ax in axs[:, :].flatten():
        ax.set_xlabel(r'$t$')
        ax.set_xticks([])
        ax.set_xlim([0, 1])
        ax.set_xlabel('')

    for ax in axs[-1, :].flatten():
        ax.set_xlim([0, 1])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['0', r'$t_\text{end}$'])

    for ax in axs[:, -1]:
        ax.set_ylabel('')
        ax.set_yticks([])

    for ax in axs[:, 0]:
        ax.set_ylabel(r'$V$ (mV)')

    return axs.flatten(), cbar_ax

if __name__ == '__main__':
    main()
