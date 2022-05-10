#!/usr/bin/env python3

from MarkovModels import common
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('data_directory', help='directory where data is stored')
    parser.add_argument('--experiment_name', '-e', default='newtonrun4')
    parser.add_argument('--wells', '-w', type=str, nargs='+')
    parser.add_argument('--protocols', type=str, nargs='+')
    parser.add_argument('--output', type=str)
    parser.add_argument('--figsize', '-f', nargs=2, type=float)
    parser.add_argument('--parameter_file')
    parser.add_argument('--model', default='Beattie')

    args = parser.parse_args()

    output_dir = common.setup_output_directory(args.output, 'plot_data')

    if args.wells is None:
        args.wells = common.get_all_wells_in_directory(args.data_directory)

    if args.protocols is None:
        args.protocols = common.get_protocol_list()

    if args.parameter_file and args.model:
        params_df = pd.read_csv(args.parameter_file)
        model_class = common.get_model_class(args.model)
    else:
        params_df = None

    fig = plt.figure(figsize=args.figsize)
    axs = fig.subplots(2)
    for well in args.wells:
        for protocol in args.protocols:
            if os.path.exists(os.path.join(args.data_directory,
                                           f"{args.experiment_name}-{protocol}-times.csv")):
                current = common.get_data(well, protocol, args.data_directory,
                                          args.experiment_name)
                times = pd.read_csv(os.path.join(args.data_directory,
                                                 f"{args.experiment_name}-{protocol}-times.csv"))['time'].values.flatten()

                prot_func, tstart, tend, tstep, desc = common.get_ramp_protocol_from_csv(protocol)

                if params_df is not None:
                    param_labels = model_class().get_parameter_labels()
                    parameters = params_df[(params_df.well == well) &
                                           (params_df.protocol == protocol)].head(1)[param_labels].values.flatten()
                    model = model_class(voltage=prot_func, times=times,
                                        parameters=parameters, protocol_description=desc)

                else:
                    model = None

                voltages = np.array([prot_func(t) for t in times])

                data_alpha = 1 if model is None else .5
                axs[0].plot(times, current, color='grey', label='data', alpha=data_alpha)
                axs[1].plot(times, voltages)
                axs[1].set_xlabel('time / ms')
                axs[1].set_ylabel('V_in / mV')
                axs[0].set_ylabel('I_Kr / nA')

                if model:
                    axs[0].plot(times, model.SimulateForwardModel(), color='green', label='Beattie model')
                    axs[0].legend()

                fig.savefig(os.path.join(output_dir, f"{well}_{protocol}_{args.experiment_name}.png"))

            for ax in axs:
                ax.cla()




if __name__ == "__main__":
    main()

