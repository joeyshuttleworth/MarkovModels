#!/usr/bin/env python3

from MarkovModels import common
import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import pints
import pints.plot
from pathos.multiprocessing import ProcessPool as Pool

from MarkovModels.BeattieModel import BeattieModel

from numba import njit

import matplotlib.pyplot as plt
import matplotlib as mpl
sigma2 = 0.01**2


def main():
    plt.style.use('classic')

    parser = common.get_parser(description="Fit a synthetic model many times to check for convergence")
    parser.add_argument("-n", "--no_samples", type=int, default=1000)
    parser.add_argument("-c", "--cpus", type=int, default=1)
    parser.add_argument("-i", "--max_iterations", type=int, default=None)

    global args
    args = parser.parse_args()

    global optimiser
    optimiser = pints.CMAES

    # Setup a pool for parallel computation
    pool = Pool(args.cpus)

    output_dir = common.setup_output_directory(args.output, "test_optimisation")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    protocol_func, tstart, tend, tstep, desc = common.get_ramp_protocol_from_csv('staircase')
    times = np.linspace(tstart, tend, int((tend-tstart)/tstep))
    model = BeattieModel(voltage=protocol_func, times=times)
    model.protocol_description = desc

    starting_positions = list(10**(np.random.uniform(-9, 1, (args.no_samples, model.get_no_parameters())))) +\
    [model.get_default_parameters()]

    print(starting_positions)

    data = model.SimulateForwardModel() + np.random.normal(0, np.sqrt(sigma2), times.shape)
    arg_list = [(pos, data) for pos in starting_positions]

    fits = pool.map(fit_func, *zip(*arg_list))

    found_parameters, scores = list(zip(*fits))
    found_parameters = np.row_stack(found_parameters)

    param_labels = model.param_labels

    columns = ["initial_%s" % lab for lab in param_labels]\
        + ["fitted_%s" % lab for lab in param_labels]\
        + ["score"]

    fits_df = pd.DataFrame(np.column_stack((starting_positions, found_parameters, scores)), columns=columns)

    print(fits_df)

    fits_df.to_csv(os.path.join(output_dir, "optimsation_results"))


def fit_func(starting_position, data):
    protocol_func, tstart, tend, tstep, desc = common.get_ramp_protocol_from_csv('staircase')
    print(starting_position)
    times = np.linspace(tstart, tend, int((tend-tstart)/tstep))
    model = BeattieModel(voltage=protocol_func, times=times)
    model.protocol_description = desc

    return common.fit_model(model, data, starting_position)


if __name__ == "__main__":
    main()
