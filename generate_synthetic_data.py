#! /usr/bin/env python3

from MarkovModels import common
from MarkovModels.BeattieModel import BeattieModel

import argparse
import numpy as np
import pandas as pd
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", default=None)
    parser.add_argument("--model", default='BeattieModel')
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--prefix", default='synthetic')
    parser.add_argument('-p', '--protocols', default=None)
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

    if args.model == 'BeattieModel':
        model_class = BeattieModel
    else:
        assert False

    protocols = common.get_protocol_list()\
        if args.protocols is None\
        else args.protocols

    print(protocols)

    for protocol in protocols:
        prot, tstart, tend, tstep, desc = common.get_ramp_protocol_from_csv(protocol)

        times = np.linspace(tstart, tend, int((tend - tstart)/tstep))
        # voltages = [prot(t) for t in times]

        times_df = pd.DataFrame(times.T, columns=('time',))

        times_df.to_csv(os.path.join(output_dir, f"{args.prefix}-{protocol}-times.csv"))
        model = model_class(prot, times, Erev=Erev, parameters=parameters)
        model.protocol_description = desc

        data = model.SimulateForwardModel() + np.random.normal(0, sigma, times.shape)

        # Output data
        out_fname = os.path.join(output_dir, f"{args.prefix}-{protocol}-A01.csv")
        pd.DataFrame(data.T, columns=('current',)).to_csv(out_fname)


if __name__ == "__main__":
    main()
