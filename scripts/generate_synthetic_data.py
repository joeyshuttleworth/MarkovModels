#! /usr/bin/env python3

from MarkovModels import common
from MarkovModels.BeattieModel import BeattieModel
from MarkovModels.KempModel import KempModel
from MarkovModels.ClosedOpenModel import ClosedOpenModel

import argparse
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import multiprocessing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", default=None)
    parser.add_argument("--model", default='Beattie')
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--prefix", default='synthetic')
    parser.add_argument('-P', '--protocols', nargs='+', default=None)
    parser.add_argument('-p', '--plot', action='store_true', default=False)
    parser.add_argument('--noise', default=0.01, type=float)
    parser.add_argument('--Erev', '-e', default=None)
    parser.add_argument('--cpus', '-c', default=1, type=int)
    parser.add_argument('--repeats', '-c', default=1, type=int)

    global args
    args = parser.parse_args()

    global Erev
    Erev = common.calculate_reversal_potential()\
        if args.Erev is None\
        else args.Erev

    global output_dir
    output_dir = common.setup_output_directory(args.output, 'synthetic_data_%s' % args.model)

    global sigma
    sigma = args.noise

    global model_class
    model_class = common.get_model_class(args.model)

    global parameters
    if args.parameters is not None:
        param_labels = model_class().get_parameter_labels()
        parameters = pd.read_csv(args.parameters)[param_labels].values[0, :]
    else:
        parameters = model_class().get_default_parameters()

    protocols = common.get_protocol_list()\
        if args.protocols is None\
        else args.protocols

    tasks = [(p, args.repeats) for p in protocols]
    with multiprocessing.Pool(args.cpus) as pool:
        pool.starmap(generate_data, tasks)


def generate_data(protocol, no_repeats):
    prot, tstart, tend, tstep, desc = common.get_ramp_protocol_from_csv(protocol)

    times = np.linspace(tstart, tend, int((tend - tstart)/tstep))
    # voltages = [prot(t) for t in times]

    times_df = pd.DataFrame(times.T, columns=('time',))

    times_df.to_csv(os.path.join(output_dir, f"{args.prefix}-{protocol}-times.csv"))
    model = model_class(prot, times, Erev=Erev, parameters=parameters)
    model.protocol_description = desc

    mean = model.SimulateForwardModel()

    for repeat in range(no_repeats):
        data = mean + np.random.normal(0, sigma, times.shape)

        # Output data
        out_fname = os.path.join(output_dir, f"{args.prefix}_{protocol}_{repeat}.csv")
        pd.DataFrame(data.T, columns=('current',)).to_csv(out_fname)

        if args.plot:
            fig = plt.figure(figsize=(14, 12))
            axs = fig.subplots(3)
            axs[0].plot(times, mean, label='mean')
            axs[0].plot(times, data, label='data', color='grey', alpha=0.5)
            axs[1].plot(times, model.GetStateVariables())
            axs[1].legend(model.state_labels)
            axs[0].legend()
            axs[1].set_xlabel('time / ms')
            axs[0].set_ylabel('current / nA')
            axs[2].plot(times, [model.voltage(t) for t in times], label='voltage / mV')
            fig.savefig(os.path.join(output_dir, f"%s_plot.png" % protocol))
            plt.close(fig)



if __name__ == "__main__":
    main()
