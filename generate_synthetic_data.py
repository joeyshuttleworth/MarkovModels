#! /usr/bin/env python3

from MarkovModels import common
from MarkovModels.BeattieModel import BeattieModel
from MarkovModels.KempModel import KempModel

import argparse
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", default=None)
    parser.add_argument("--model", default='Beattie')
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--prefix", default='synthetic')
    parser.add_argument('-P', '--protocols', default=None)
    parser.add_argument('-p', '--plot', action='store_true', default=False)
    parser.add_argument('--noise', default=0.01)
    parser.add_argument('--Erev', '-e', default=None)

    args = parser.parse_args()

    Erev = common.calculate_reversal_potential()\
        if args.Erev is None\
        else args.Erev

    if args.parameters is not None:
        parameters = pd.read_csv(args.parameters).values
    else:
        parameters = None

    output_dir = common.setup_output_directory(args.output, 'synthetic_data')

    sigma = args.noise

    if args.model == 'Beattie':
        model_class = BeattieModel
    elif args.model == 'Kemp':
        model_class = KempModel
    else:
        assert False

    protocols = common.get_protocol_list()\
        if args.protocols is None\
        else args.protocols

    print(protocols)

    fig = plt.figure()
    ax = plt.subplots()

    for protocol in protocols:
        prot, tstart, tend, tstep, desc = common.get_ramp_protocol_from_csv(protocol)

        times = np.linspace(tstart, tend, int((tend - tstart)/tstep))
        # voltages = [prot(t) for t in times]

        times_df = pd.DataFrame(times.T, columns=('time',))

        times_df.to_csv(os.path.join(output_dir, f"{args.prefix}-{protocol}-times.csv"))
        model = model_class(prot, times, Erev=Erev, parameters=parameters)
        model.protocol_description = desc

        mean = model.SimulateForwardModel()
        data = mean + np.random.normal(0, sigma, times.shape)

        # Output data
        out_fname = os.path.join(output_dir, f"{args.prefix}-{protocol}-A01.csv")
        pd.DataFrame(data.T, columns=('current',)).to_csv(out_fname)

        if args.plot:
            ax.plot(times, mean, label='mean')
            ax.plot(times, data, label='data', color='grey', alpha='0.5')
            ax.legend()
            fig.savefig(os.path.join(output_dir, f"%s_plot.png" % protocol))



if __name__ == "__main__":
    main()
