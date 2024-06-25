#!/usr/bin/env python3

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import itertools
from matplotlib import gridspec

from matplotlib import rc

import markovmodels
from markovmodels.model_generation import make_model_of_class
from markovmodels.fitting import get_best_params, make_prediction
from markovmodels.ArtefactModel import ArtefactModel
from markovmodels.utilities import setup_output_directory, get_data, get_all_wells_in_directory
from markovmodels.voltage_protocols import get_protocol_list, get_ramp_protocol_from_json, make_voltage_function_from_description

# rc('text', usetex=True)
rc('figure', dpi=400, facecolor=[0]*4)
rc('axes', facecolor=[0]*4)
rc('savefig', facecolor=[0]*4)
rc('figure', autolayout=True)

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('data_directory', help='directory where data is stored')
    parser.add_argument('fitting_case', type=str)
    parser.add_argument('subtraction_file')
    parser.add_argument('parameter_file')
    parser.add_argument('--model_class')
    parser.add_argument('--removal_duration', default=5.0, type=float)
    parser.add_argument('--experiment_name', '-e', default='newtonrun4')
    parser.add_argument('--wells', '-w', type=str, nargs='+')
    parser.add_argument('--protocols', type=str, nargs='+')
    parser.add_argument('--validation_protocols', type=str, nargs='+')
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('--figsizet', '-f', nargs=2, type=float, default=[5.54, 7])
    parser.add_argument('--fig_title', '-t', default='')
    parser.add_argument('--nolegend', action='store_true')
    parser.add_argument('--dpi', '-d', default=500, type=int)
    parser.add_argument('--fontsize', type=int)
    parser.add_argument('--show_uncertainty', action='store_true')
    parser.add_argument('--shared_plot_limits', action='store_true')
    parser.add_argument('--no_voltage', action='store_true')
    parser.add_argument('--file_format', default='')
    parser.add_argument('--E_rev', default=-91.71, type=float)

    global args
    args = parser.parse_args()

    output_dir = setup_output_directory(args.output, 'plot_data')

    subtraction_df = pd.read_csv(args.subtraction_file)

    # Case describing how was the was model fitted
    if args.fitting_case == '0a':
        args.adjust_kinetics = False
        args.infer_reversal_potential = False
        args.use_artefact_model = False
    elif args.fitting_case == '0b':
        args.adjust_kinetics = False
        args.infer_reversal_potential = True
        args.use_artefact_model = False
    elif args.fitting_case == '0c':
        args.adjust_kinetics = True
        args.infer_reversal_potential = True
        args.use_artefact_model = False
    elif args.fitting_case == 'I':
        args.adjust_kinetics = False
        args.infer_reversal_potential = False
        args.use_artefact_model = True
    elif args.case == 'II':
        args.adjust_kinetics = False
        args.infer_reversal_potential = False
        args.use_artefact_model = True

    if args.fontsize:
        matplotlib.rcParams.update({'font.size': args.fontsize})

    if args.protocols is None:
        args.protocols = get_protocol_list()

    params_df = get_best_params(pd.read_csv(args.parameter_file))

    if params_df is not None:
        args.protocols = [protocol for protocol in args.protocols if protocol
                          in params_df['protocol'].unique()]
        wells = [well for well in params_df['well'].unique()]
        if args.wells:
            wells = [w for w in wells if w in args.wells]
    else:
        wells = []

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    axs = fig.subplots(2)

    for ax in axs:
        ax.spines[['top', 'right']].set_visible(False)

    cm = sns.husl_palette(len(args.protocols))

    if len(args.protocols) == 1:
        lw = 1
    else:
        lw = .5

    protocol_dict = {}
    for protocol in args.protocols + args.validation_protocols:
        v_func, desc = get_ramp_protocol_from_json(protocol, os.path.join(args.data_directory, 'protocols'),
                                              args.experiment_name)

        times = np.loadtxt(os.path.join(args.data_directory,
                                        f"{args.experiment_name}-{protocol}-times.csv")).astype(np.float64).flatten()

        protocol_dict[protocol] = desc, times

    param_labels = make_model_of_class(args.model_class).get_parameter_labels()

    for well in wells:
        sweep = params_df.sweep.unique()[0]

        for prediction_protocol in args.validation_protocols + args.protocols:
            current, _ = get_data(well, prediction_protocol, args.data_directory,
                                        args.experiment_name, sweep=sweep)

            desc, times  = protocol_dict[prediction_protocol]

            model = make_model_of_class(args.model_class, voltage=v_func, times=times,
                                        protocol_description=desc, E_rev=args.E_rev)

            voltages = np.array([v_func(t, protocol_description=desc) for t in times])

            if args.use_artefact_model:
                model = ArtefactModel(model)
                param_labels = model.get_parameter_labels()

            fit = make_prediction(args.model_class, args, well,
                                  prediction_protocol, sweep, prediction_protocol,
                                  sweep, params_df, subtraction_df,
                                  args.fitting_case, args.E_rev,
                                  protocol_dict, current, voltages,
                                  label='', solver=None,
                                  do_spike_removal=True,
                                  return_states=False, strict=True,
                                  tolerances=(None, None))

            axs[0].plot(times*1e-3, current, color='grey', alpha=0.5)
            axs[0].plot(times*1e-3, fit*1e-3, label='fit')
            axs[0].legend()

            if args.use_artefact_model:
                axs[0].set_ylabel(r'$I_\text{post}$ (nA)')
            else:
                axs[0].set_ylabel(r'$I_\text{Kr}$ (nA)')

            if args.use_artefact_model:
                axs[1].set_ylabel(r'$V$ (mV)')
                axs[1].plot(times*1e-3, state_solver(fit_params,
                                                     protocol_description=desc)[:, -1],
                            label=r'$V_\text{m}$')
                axs[1].plot(times*1e-3, voltages, label=r'$V_\text{cmd}$')
                axs[1].legend()
            else:
                axs[1].set_ylabel(r'$V_\text{cmd} (mV)$')
                axs[1].plot(times*1e-3, voltages)

            axs[1].set_xlabel(r'$t$ (ms)')
            fig.savefig(os.path.join(output_dir, f"{prediction_protocol}-{well}-sweep{sweep}-fit.pdf"))

            for ax in axs:
                ax.cla()

            V_off = 0

            predictions = []
            for i, protocol in enumerate(args.protocols):
                if params_df is not None:
                    if protocol not in params_df.protocol.unique():
                        continue

                    parameters = params_df[(params_df.well == well) &
                                            (params_df.protocol == protocol)].head(1)[param_labels].values.flatten()
                else:
                    parameters = None

                if len(args.protocols) == 1:
                    color = 'green'
                    label = model.get_model_name()
                elif args.show_uncertainty:
                    color = 'grey'
                    label = None
                else:
                    color = cm[i]
                    label = f"{protocol} fit"

                if model:
                    prediction = make_prediction(args.model_class, args, well,
                                                 prediction_protocol, sweep,
                                                 protocol, sweep, params_df,
                                                 subtraction_df,
                                                 args.fitting_case, args.E_rev,
                                                 protocol_dict, current,
                                                 voltages, label='',
                                                    solver=None,
                                                 do_spike_removal=True,
                                                 return_states=False,
                                                 strict=True, tolerances=(None,
                                                                          None))

                    color = 'red'
                    axs[0].plot(times*1e-3, prediction, color=color,
                                label=label, linewidth=lw,
                                linestyle='--')
                    predictions.append(prediction)

            if not args.use_artefact_model:
                axs[1].plot(times*1e-3, voltages, linewidth=lw)
                axs[1].set_xlabel('time (ms)')
                axs[1].set_ylabel(r'$V_\text{cmd}$ (mV)')
            else:
                print('voltages', voltages)
                axs[1].set_ylabel(r'$V$ (mV)')
                axs[1].plot(times*1e-3, voltages, linewidth=lw, label=r'$V_\text{cmd}$')
                Vm = state_solver(parameters, protocol_description=desc,
                                  times=times)[:, -1]
                axs[1].plot(times*1e-3, Vm, label=r'$V_\text{m}$')
                axs[1].set_xlabel('time (ms)')

                axs[1].legend()

            data_alpha = 1 if model is None else .5
            axs[0].plot(times*1e-3, current, color='grey', label='data', alpha=data_alpha, linewidth=lw)
            axs[0].set_ylabel(r'$I_{Kr}$ (nA)')

            if predictions:
                predictions = np.stack(predictions)
                min_pred = predictions.min(axis=0)
                max_pred = predictions.max(axis=0)

                axs[0].fill_between(times*1e-3, min_pred, max_pred, color='orange', alpha=.4)

            if not args.nolegend:
                axs[0].legend(prop={'size': 6})
                axs[0].set_xticks([])

            axs[0].set_ylim(np.min(current), np.max(current))

            axs[0].set_title(args.fig_title)
            # fig.tight_layout()
            fig.savefig(os.path.join(output_dir,
                                        f"{well}_{prediction_protocol}_{args.experiment_name}.pdf"),
                        dpi=args.dpi)

            for ax in axs:
                ax.cla()

    else:
        lw = 1

        if args.shared_plot_limits:
            time_range = (0, 0)
            current_range = (0, 0)
            voltage_range = (0, 0)
            for well in wells:
                for protocol in args.protocols:
                    data, voltages, fit, times = get_data_voltages_fit_times(protocol, well, params_df, model_class)
                    time_range = (0, max(np.max(times), time_range[1]))
                    voltage_range = (min(np.min(voltages), voltage_range[0]),
                                     max(np.max(voltages), voltage_range[1]))
                    current_range = (min(np.min(data), np.min(fit), current_range[0]),
                                     max(np.max(data), np.max(fit), current_range[1]))
        else:
            time_range = None
            current_range = None
            voltage_range = None


def get_data_voltages_fit_times(protocol, well, params_df, model_class):
    sweep = 0


    if os.path.exists(os.path.join(args.data_directory,
                                   f"{args.experiment_name}-{protocol}-times.csv")):
        current, vp = get_data(well, protocol, args.data_directory,
                                      args.experiment_name, sweep=sweep)

        desc = vp.get_all_sections()

        v_func = make_voltage_function_from_description(desc,
                                                        holding_potential=-80.0)

        times = np.loadtxt(os.path.join(args.data_directory,
                                        f"{args.experiment_name}-{protocol}-times.csv")).astype(np.float64).flatten()
        voltages = np.array([v_func(t,
                                    protocol_description=desc) for t in times])

        fit = None

        if params_df is not None:
            model = make_model_of_class(args.model_class)
            param_labels = model.get_parameter_labels()
            parameters = params_df[(params_df.well == well) &
                                   (params_df.protocol == protocol)].head(1)[param_labels].values.flatten()
            model = make_model_of_class(args.model_class, voltage=v_func,
                                        times=times, protocol_description=desc)
            fit = model.SimulateForwardModel(parameters)

        else:
            fit = None
    else:
        raise Exception('could not open data')

    times = times * 1e-3
    current = current * 1e-3

    return current, voltages, fit, times



if __name__ == "__main__":
    main()
