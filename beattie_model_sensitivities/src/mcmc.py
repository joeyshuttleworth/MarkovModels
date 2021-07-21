#!/usr/bin/env python3

import os
import math
import pints
import pints.plot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import scipy.integrate as integrate
import symengine as se
import matplotlib.pyplot as plt
import matplotlib
import argparse

from settings import Params
from sensitivity_equations import GetSensitivityEquations, CreateSymbols
from common import *

class PintsWrapper(pints.LogPDF):
    # A class used by pints to compute the log likelihood of the model
    def __init__(self, data, settings, args, times_to_use):
        par = Params()
        self.data = data
        self.times_to_use = times_to_use
        self.starting_parameters = [2.26E-04, 0.0699, 3.45E-05, 0.05462, 0.0873, 8.92E-03, 5.150E-3, 0.03158, 0.1524]
        # Create symbols for symbolic functions
        p, y, v = CreateSymbols(settings)

        # Choose starting parameters (from J Physiol paper)
        para = [2.26E-04, 0.0699, 3.45E-05, 0.05462, 0.0873, 8.92E-03, 5.150E-3, 0.03158, 0.1524]

        # Define system equations and initial conditions
        k1 = p[0] * se.exp(p[1] * v)
        k2 = p[2] * se.exp(-p[3] * v)
        k3 = p[4] * se.exp(p[5] * v)
        k4 = p[6] * se.exp(-p[7] * v)

        # Write in matrix form taking y = ([C], [O], [I])^T

        A = se.Matrix([[-k1 - k3 - k4, k2 -  k4, -k4], [k1, -k2 - k3, k4], [-k1, k3 - k1, -k2 - k4 - k1]])
        B = se.Matrix([k4, 0, k1])

        rhs = np.array(A * y + B)

        voltage = None
        protocol = args.protocol
        if protocol == "sine_wave":
            voltage = beattie_sine_wave
        elif protocol == "staircase":
            voltage = get_staircase_protocol()
        else:
            assert(False)

        self.funcs = GetSensitivityEquations(par, p, y, v, A, B, para, times_to_use, voltage=voltage)

    def __call__(self, p):
        # Fix all parameters except p_5 and p_7
        p_vec = np.copy(self.starting_parameters)
        p_vec[4] = p[0]
        p_vec[6] = p[1]
        pred = self.funcs.SimulateForwardModel(p_vec)

        # compute sample variance
        errors = pred - self.data
        s_var = 1

        # Compute the log likelihood (assuming i.i.d Gaussian noise)
        n = pred.shape[0]
        ll = -n*0.5*np.log(2*np.pi) - n*0.5*np.log(s_var) - 1/(2*s_var)*(errors**2).sum()
        return ll

    def n_parameters(self):
        return 2

def extract_times(lst, time_ranges, step):
    """
    Take values from a list, lst which are indexes between the upper and lower
    bounds provided by time_ranges. Each element of time_ranges specifies an
    upper and lower bound.

    Returns a 2d numpy array containing all of the relevant data points
    """
    ret_lst = []
    for time_range in time_ranges:
        ret_lst.extend(lst[time_range[0]:time_range[1]:step].tolist())
    return np.array(ret_lst)

def main():
    #constants
    indices_to_use = [[1,2499], [2549,2999], [3049,4999], [5049,14999], [15049,19999], [20049,29999], [30049,64999], [65049,69999], [70049,-1]]

    starting_parameters = np.array([2.26E-04, 0.0699, 3.45E-05, 0.05462, 0.0873, 8.92E-03, 5.150E-3, 0.03158, 0.1524])

    plt.rcParams['axes.axisbelow'] = True

    # Check input arguments
    parser = argparse.ArgumentParser(
        description='Plot sensitivities of the Beattie model')
    parser = get_parser(data_reqd=True)
    parser.add_argument("-n", "--no_chains",type=int, default=3, help="number of chains to use")
    parser.add_argument("-l", "--chain_length", type=int, default=1000, help="length of chain to use")
    parser.add_argument("-v", "--protocol", default=None, help="name of the protocol to use")
    parser.add_argument("-b", "--burn_in", type=int, default=None, help="amount of burn in to use")

    args = parser.parse_args()

    output_dir = os.path.join(args.output, "{}_mcmc".format(args.protocol))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    data = pd.read_csv(args.data_file_path, delim_whitespace=True)
    times = data["time"]
    data = data["current"].values

    print("outputting to {}".format(args.output))

    # Create output directory
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    if not os.path.exists(args.data_file_path):
        print("Input file not provided. Doing nothing.")
        return

    par = Params()

    skip = int(par.timestep/0.1)

    dat = extract_times(data, indices_to_use, skip)

    ll = PintsWrapper(data, par, args, times)
    prior = pints.UniformLogPrior([0,0], [1,1])
    posterior = pints.LogPosterior(ll, prior)
    mcmc = pints.MCMCController(posterior, args.no_chains, np.tile(starting_parameters[[4,6]], [args.no_chains,1]), method=pints.HaarioBardenetACMC)
    mcmc.set_max_iterations(args.chain_length)

    chains = mcmc.run()
    if args.burn_in is not None:
        chains = chains[:, args.burn_in:, :]
    for i, chain in enumerate(chains):
        np.savetxt(os.path.join(output_dir, "chain_{}.csv".format(i)), chain, delimiter=",")

    # Plot histograms
    pints.plot.histogram(chains, parameter_names=["p5","p7"], kde=True)
    if args.plot:
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, "histogram.pdf"))

if __name__ == "__main__":
    main()
    print("done")
