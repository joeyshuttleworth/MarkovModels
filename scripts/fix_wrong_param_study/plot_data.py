#!/usr/bin/env python3

from MarkovModels import common
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

    parser.add_argument('datafile', help='directory where data is stored')
    parser.add_argument('--experiment_name', '-e', default='newtonrun4')
    parser.add_argument('--wells', '-w', type=str, nargs='+')
    parser.add_argument('--protocols', type=str, nargs='+')
    parser.add_argument('--prediction_protocols', type=str, nargs='+', default=['longap'])
    parser.add_argument('--output', type=str)
    parser.add_argument('--figsize', '-f', nargs=2, type=float)
    parser.add_argument('--fig_title', '-t', default='')
    parser.add_argument('--parameter_file')
    parser.add_argument('--model_class', default='Beattie')
    parser.add_argument('--nolegend', action='store_true')
    parser.add_argument('--dpi', '-d', default=500, type=int)
    parser.add_argument('--fontsize', type=int)
    parser.add_argument('--show_uncertainty', action='store_true')
    parser.add_argument('--ignore_protocols', nargs='+', default=[])
    parser.add_argument('--no_voltage', action='store_true')
    parser.add_argument('--file_format', default='')
    parser.add_argument('--fix_param')
    parser.add_argument('--sampling_timestep', type=np.float64, default=0.1)
    parser.add_argument('--ylims', type=np.float64, nargs=2)

    global args
    args = parser.parse_args()

    model_class = common.get_model_class(args.model_class)

    if not args.fix_param:
        args.fix_param = model_class().get_parameter_labels()[-1]

    output_dir = common.setup_output_directory(args.output, 'plot_data')

    if args.fontsize:
        matplotlib.rcParams.update({'font.size': args.fontsize})

    if args.protocols is None:
        args.protocols = common.get_protocol_list()

    params_df = pd.read_csv(args.datafile)

    if args.protocols:
        args.protocols = [protocol for protocol in args.protocols if protocol
                          in params_df['protocol'].unique()]
    else:
        args.protocols = params_df['protocol'].unique()

    if args.wells:
        args.wells = [well for well in args.wells if well in params_df['well'].unique()]
    else:
        args.wells = params_df['well'].unique()

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

    for lab in model_class().get_parameter_labels():
        params_df[lab] = params_df[lab].astype(np.float64)

    print(params_df)

    for well in args.wells:
        for val in params_df[args.fix_param].unique():
            for prediction_protocol in args.prediction_protocols:

                prot, tstart, tend, tstep, desc = common.get_ramp_protocol_from_csv(prediction_protocol)
                times = np.linspace(tstart, tend, int((tend - tstart)/args.sampling_timestep))
                voltages = np.array([prot(t) for t in times])
                model = model_class(prot, times,
                                    protocol_description=desc)
                solver = model.make_forward_solver_current()

                predictions = []

                print(params_df[(params_df.well == well) &
                                (params_df[args.fix_param] == val)])

                print([params_df[args.fix_param]])

                for i, protocol in enumerate(args.protocols):
                    param_labels = model_class().get_parameter_labels()
                    parameters = params_df[(params_df.well == well) &
                                           (params_df.protocol == protocol) &
                                           (params_df[args.fix_param] == val)]\
                                           .head(1)[param_labels].values.flatten()

                    if len(args.protocols) == 1:
                        color = 'green'
                        label = model.get_model_name()
                    elif args.show_uncertainty:
                        color = 'grey'
                        label = None
                    else:
                        color = cm[i]
                        label = f"{protocol} fit"

                    prediction = solver(parameters)
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

                # data_alpha = 1 if model is None else .5

                axs[0].set_ylabel(r'$I_{Kr}$ / nA')

                if predictions:
                    predictions = np.stack(predictions)
                    min_pred = predictions.min(axis=0)
                    max_pred = predictions.max(axis=0)

                    axs[0].fill_between(times, min_pred, max_pred, color='orange', alpha=.4)

                axs[0].set_title(args.fig_title)

                if args.ylims:
                    axs[0].set_ylim(*args.ylims)

                axs[0].plot(times, solver(), label='blue', lw=lw)
                fig.savefig(os.path.join(output_dir,
                                         f"{well}_{prediction_protocol}_{args.experiment_name}_{val:.4f}.{args.file_format}"),
                            dpi=args.dpi)

                for ax in axs:
                    ax.cla()


if __name__ == "__main__":
    main()
