#!/usr/bin/env python3

from markovmodels import common
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import gridspec


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('data_directory', help='directory where data is stored')
    parser.add_argument('--experiment_name', '-e', default='newtonrun4')
    parser.add_argument('--wells', '-w', type=str, nargs='+')
    parser.add_argument('--protocols', type=str, nargs='+')
    parser.add_argument('--prediction_protocols', type=str, nargs='+', default=[])
    parser.add_argument('--output', type=str)
    parser.add_argument('--figsize', '-f', nargs=2, type=float)
    parser.add_argument('--fig_title', '-t', default='')
    parser.add_argument('--parameter_file')
    parser.add_argument('--model', default='Beattie')
    parser.add_argument('--nolegend', action='store_true')
    parser.add_argument('--dpi', '-d', default=500, type=int)
    parser.add_argument('--fontsize', type=int)
    parser.add_argument('--show_uncertainty', action='store_true')
    parser.add_argument('--ignore_protocols', nargs='+', default=[])
    parser.add_argument('--shared_plot_limits', action='store_true')
    parser.add_argument('--no_voltage', action='store_true')
    parser.add_argument('--file_format', default='')

    global args
    args = parser.parse_args()

    output_dir = common.setup_output_directory(args.output, 'plot_data')

    if args.fontsize:
        matplotlib.rcParams.update({'font.size': args.fontsize})

    if args.wells is None:
        args.wells = common.get_all_wells_in_directory(args.data_directory)

    if args.protocols is None:
        args.protocols = common.get_protocol_list()

    if args.parameter_file and args.model:
        params_df = pd.read_csv(args.parameter_file)
        model_class = common.get_model_class(args.model)
    else:
        params_df = None

    if params_df is not None:
        args.protocols = [protocol for protocol in args.protocols if protocol
                          in params_df['protocol'].unique()]
        args.wells = [well for well in args.wells if well in params_df['well'].unique()]

    # if args.prediction_protocols is None:
    #     args.prediction_protocols = args.protocols

    fig = plt.figure(figsize=args.figsize)

    if not args.no_voltage:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 2])
        axs = [fig.add_subplot(ax) for ax in gs]
    else:
        axs = [fig.subplots()]

    cm = sns.husl_palette(len(args.protocols))

    if len(args.protocols) == 1:
        lw = 1
    else:
        lw = .5

    args.protocols = [p for p in args.protocols if p not in
                      args.ignore_protocols]

    if args.prediction_protocols:
        for well in args.wells:
            for prediction_protocol in args.prediction_protocols:
                predictions = []
                prot_func, _, desc = common.get_ramp_protocol_from_csv(prediction_protocol)
                if os.path.exists(os.path.join(args.data_directory,
                                               f"{args.experiment_name}-{prediction_protocol}-times.csv")):
                    current = common.get_data(well, prediction_protocol, args.data_directory,
                                              args.experiment_name)
                    times = pd.read_csv(os.path.join(args.data_directory,
                                                     f"{args.experiment_name}-{prediction_protocol}-times.csv"))['time'].values.astype(np.float64).flatten()

                voltages = np.array([prot_func(t) for t in times])
                model = model_class(voltage=prot_func, times=times,
                                    protocol_description=desc)
                print(params_df)
                for i, protocol in enumerate(args.protocols):
                    if params_df is not None:
                        param_labels = model_class().get_parameter_labels()
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
                        label = f"{prediction_protocol} fit"

                    if model:
                        prediction = model.SimulateForwardModel(parameters)
                        color = 'red'
                        axs[0].plot(times, prediction, color=color,
                                    label=label, linewidth=lw,
                                    linestyle='--')
                        predictions.append(prediction)

                if not args.no_voltage:
                    axs[1].plot(times, voltages, linewidth=lw)
                    axs[1].set_xlabel('time / ms')
                    axs[1].set_ylabel(r'$V_{in}$ / mV')
                else:
                    axs[0].set_xlabel('time / ms')
                    axs[0].set_ylabel(r'$V_in$ / mV')

                data_alpha = 1 if model is None else .5
                axs[0].plot(times, current, color='grey', label='data', alpha=data_alpha, linewidth=lw)
                axs[0].set_ylabel(r'$I_{Kr}$ / nA')

                if predictions:
                    predictions = np.stack(predictions)
                    min_pred = predictions.min(axis=0)
                    max_pred = predictions.max(axis=0)
                    # mean_pred = predictions.mean(axis=0)

                    # plot min and max
                    # axs[0].plot(times, min_pred, linestyle='--', color='red', linewidth=lw, label='min/max prediction')
                    # axs[0].plot(times, max_pred, linestyle='--', color='red', linewidth=lw)
                    # # axs[0].plot(times, mean_pred, color='red', linewidth=lw, label='mean prediction')
                    axs[0].fill_between(times, min_pred, max_pred, color='orange', alpha=.4)

                if not args.nolegend:
                    axs[0].legend(prop={'size': 6})
                    axs[0].set_xticks([])

                axs[0].set_ylim(np.min(current), np.max(current))

                axs[0].set_title(args.fig_title)
                fig.savefig(os.path.join(output_dir,
                                         f"{well}_{prediction_protocol}_{args.experiment_name}.{args.file_format}"),
                            dpi=args.dpi)

                for ax in axs:
                    ax.cla()

    else:
        lw = 1

        if args.shared_plot_limits:
            time_range = (0, 0)
            current_range = (0, 0)
            voltage_range = (0, 0)
            for well in args.wells:
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

        for well in args.wells:
            for protocol in args.protocols:
                data, voltages, fit, times = get_data_voltages_fit_times(protocol, well, params_df, model_class)
                if not args.no_voltage:
                    axs[1].plot(times, voltages, linewidth=lw)
                    axs[1].set_xlabel('time / ms')
                    axs[1].set_ylabel(r'$V_{in}$ / mV')


                # Set plot limits
                if time_range is not None:
                    axs[0].set_xlim(time_range)
                    if not args.no_voltage:
                        axs[1].set_xlim(time_range)

                if current_range is not None:
                    axs[0].set_ylim(current_range)

                if voltage_range is not None and not args.no_voltage:
                    axs[1].set_ylim(voltage_range)

                data_alpha = .5
                axs[0].plot(times, data, color='grey', label='data', alpha=data_alpha, linewidth=.5)
                axs[0].plot(times, fit,)
                axs[0].set_ylabel(r'$I_{Kr}$ / nA')

                colour = 'green'
                axs[0].plot(times, fit, color=colour, linewidth=lw)

                # remove frame
                # for ax in axs:
                #     for side in ['top', 'right', 'bottom', 'left']:
                #         ax.spines[side].set_visible(False)

                axs[0].set_title(args.fig_title)
                fig.savefig(os.path.join(output_dir,
                                         f"{protocol}_{well}_fit.{args.file_format}"), dpi=args.dpi)
                for ax in axs:
                    ax.cla()


def get_data_voltages_fit_times(protocol, well, params_df, model_class):
    if os.path.exists(os.path.join(args.data_directory,
                                   f"{args.experiment_name}-{protocol}-times.csv")):
        current = common.get_data(well, protocol, args.data_directory,
                                  args.experiment_name)
        times = pd.read_csv(os.path.join(args.data_directory,
                                         f"{args.experiment_name}-{protocol}-times.csv"))['time'].values.astype(np.float64).flatten()

        prot_func, _, desc = common.get_ramp_protocol_from_csv(protocol)
        voltages = np.array([prot_func(t) for t in times])

        fit = None

        if params_df is not None:
            param_labels = model_class().get_parameter_labels()
            parameters = params_df[(params_df.well == well) &
                                   (params_df.protocol == protocol)].head(1)[param_labels].values.flatten()
            model = model_class(voltage=prot_func, times=times,
                                parameters=parameters, protocol_description=desc)
            fit = model.SimulateForwardModel()

        else:
            fit = None
    else:
        raise Exception('could not open data')

    return current, voltages, fit, times



if __name__ == "__main__":
    main()
