#!/usr/bin/env python3

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec

from matplotlib import rc

import markovmodels
from markovmodels.model_generation import make_model_of_class
from markovmodels.fitting import get_best_params
from markovmodels.ArtefactModel import ArtefactModel
from markovmodels.utilities import setup_output_directory, get_data, get_all_wells_in_directory
from markovmodels.voltage_protocols import get_protocol_list, get_ramp_protocol_from_json, make_voltage_function_from_description

rc('font', **{'size': 12})
# rc('text', usetex=True)
rc('figure', dpi=400, facecolor=[0]*4)
rc('axes', facecolor=[0]*4)
rc('savefig', facecolor=[0]*4)
rc('figure', autolayout=True)

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('data_directory', help='directory where data is stored')
    parser.add_argument('fitting_case', type=str)
    parser.add_argument('--model_class')
    parser.add_argument('--experiment_name', '-e', default='newtonrun4')
    parser.add_argument('--wells', '-w', type=str, nargs='+')
    parser.add_argument('--protocols', type=str, nargs='+')
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('--figsize', '-f', nargs=2, type=float, default=[5.54, 7])
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

    global args
    args = parser.parse_args()

    output_dir = setup_output_directory(args.output, 'plot_data')

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

    if args.parameter_file and args.model_class:
        params_df = get_best_params(pd.read_csv(args.parameter_file))
    else:
        params_df = None

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

    cm = sns.husl_palette(len(args.protocols))

    if len(args.protocols) == 1:
        lw = 1
    else:
        lw = .5

    args.protocols = [p for p in args.protocols if p not in
                      args.ignore_protocols]

    infer_reversal_params = np.loadtxt(os.path.join('data', 'BeattieModel_roomtemp_staircase_params.csv')).flatten().astype(np.float64)

    prediction_protocols = list(params_df.protocol.unique()) + ['longap']

    protocol_directory = os.path.join(args.data_directory, 'protocols')


    param_labels = make_model_of_class(args.model_class).get_parameter_labels()

    for well in wells:
        sweep = params_df.sweep.unique()[0]

        for prediction_protocol in prediction_protocols:
            # Don't plot anything which was fitted to validation protocols
            if prediction_protocol in args.ignore_protocols:
                continue

            current, vp = get_data(well, prediction_protocol, args.data_directory,
                                        args.experiment_name, sweep=sweep)
            desc = vp.get_all_sections()
            prot_func = make_voltage_function_from_description(desc)

            current = current * 1e-3

            times = np.loadtxt(os.path.join(args.data_directory,
                                            f"{args.experiment_name}-{prediction_protocol}-times.csv")).astype(np.float64).flatten()


            model = make_model_of_class(args.model_class, voltage=prot_func, times=times,
                                        protocol_description=desc, E_rev=args.reversal)
            if args.use_artefact_model:
                model = ArtefactModel(model)
                param_labels = model.get_parameter_labels()

            solver = model.make_forward_solver_current()
            state_solver = model.make_hybrid_solver_states(hybrid=False)

            V_off = 0
            # TODO get reversal potential automatically
            E_rev = args.reversal
            if args.use_artefact_model and args.infer_reversal_potential:
                V_off = \
                    markovmodels.fitting.find_V_off(prediction_protocol, times,
                                                    current, 'model3', default_parameters=infer_reversal_params,
                                                    E_rev=args.reversal)
            elif args.infer_reversal_potential:
                _, pred_desc = get_ramp_protocol_from_json(prediction_protocol,
                                                           protocol_directory,
                                                           args.experiment_name
                                                           )
                E_rev = \
                markovmodels.fitting.infer_reversal_potential(pred_desc,
                                                              current,
                                                              times,
                                                              known_Erev=args.reversal)
                model.set_E_rev(E_rev)
                solver = model.make_forward_solver_current()
                state_solver = model.make_hybrid_solver_states(hybrid=False)

            fit_params = params_df[(params_df.well==well)\
                                   & (params_df.protocol == prediction_protocol)
                                   & (params_df.sweep == sweep)][param_labels].values[0, :].flatten()

            fit = solver(fit_params, protocol_description=desc, times=times)

            voltages = np.array([prot_func(t) for t in times])
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
            # fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"{prediction_protocol}-{well}-sweep{sweep}-fit.pdf"))

            for ax in axs:
                ax.cla()


            if args.use_artefact_model:
                model.channel_model.default_parameters[-3] = V_off

            predictions =[]

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
                    # if args.use_artefact_model:
                    #     params_from_fitted_trace = params_df[(params_df.well==well)
                    #                                             & (params_df.protocol == prediction_protocol)].head(1)[param_labels].values_flatten()

                    #     params[-7:] = params_from_fitted_trace[artefact_params]

                    prediction = model.SimulateForwardModel(parameters,
                                                            protocol_description=desc,
                                                            times=times)
                    print(model.E_rev)
                    prediction = prediction * 1e-3
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
                axs[1].plot(times*1e-3, Vm, label=r'$V_\text{m}$', lw=lw,
                            color='red', linestyle='--')
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

        # for well in wells:
        #     for protocol in args.protocols:
        #         data, voltages, fit, times = get_data_voltages_fit_times(protocol, well, params_df, model_class)
        #         if not args.no_voltage:
        #             axs[1].plot(times, voltages, linewidth=lw)
        #             axs[1].set_xlabel('time (ms)')
        #             axs[1].set_ylabel(r'$V_{in}$ (mV)')


        #         # Set plot limits
        #         if time_range is not None:
        #             axs[0].set_xlim(time_range)
        #             if not args.no_voltage:
        #                 axs[1].set_xlim(time_range)

        #         if current_range is not None:
        #             axs[0].set_ylim(current_range)

        #         if voltage_range is not None and not args.no_voltage:
        #             axs[1].set_ylim(voltage_range)

        #         data_alpha = .5
        #         axs[0].plot(times*1e-3, data, color='grey', label='data', alpha=data_alpha, linewidth=.5)
        #         axs[0].plot(times*1e-3, fit,)
        #         axs[0].set_ylabel(r'$I_{Kr}$ (nA)')

        #         colour = 'green'
        #         axs[0].plot(times*1e-3, fit, color=colour, linewidth=lw)

        #         axs[0].set_title(args.fig_title)
        #         # fig.tight_layout()
        #         fig.savefig(os.path.join(output_dir,
        #                                  f"{protocol}_{well}_fit.pdf"), dpi=args.dpi)
        #         for ax in axs:
        #             ax.cla()


# def get_data_voltages_fit_times(protocol, well, params_df, model_class):
#     sweep = 0
#     if os.path.exists(os.path.join(args.data_directory,
#                                    f"{args.experiment_name}-{protocol}-times.csv")):
#         current, prot_desc = get_data(well, protocol, args.data_directory,
#                                       args.experiment_name, sweep=sweep)
#         times = np.loadtxt(os.path.join(args.data_directory,
#                                         f"{args.experiment_name}-{protocol}-times.csv")).astype(np.float64).flatten()
#         voltages = np.array([prot_func(t) for t in times])

#         fit = None

#         if params_df is not None:
#             model = make_model_of_class(args.model_class)
#             param_labels = model.get_parameter_labels()
#             parameters = params_df[(params_df.well == well) &
#                                    (params_df.protocol == protocol)].head(1)[param_labels].values.flatten()
#             model = make_model_of_class(args.model_class, voltage=prot_func,
#                                         times=times, parameters=parameters,
#                                         protocol_description=desc)
#             fit = model.SimulateForwardModel()

#         else:
#             fit = None
#     else:
#         raise Exception('could not open data')

#     times = times * 1e-3
#     current = current * 1e-3

#     return current, voltages, fit, times



if __name__ == "__main__":
    main()
