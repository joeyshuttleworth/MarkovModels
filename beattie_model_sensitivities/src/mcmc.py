#!/usr/bin/env python3

import os
import math
import pints
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

    def __init__(self, data, settings, args, times_to_use, protocol=None):
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

        if protocol == "sine_wave":
            voltage = beattie_sine_wave
        elif protocol == "staircase":
            voltage = get_staircase_protocol()

        self.funcs = GetSensitivityEquations(par, p, y, v, A, B, para, times_to_use, voltage=voltage)

    def __call__(self, p):
        pred = self.funcs.SimulateForwardModel(p)
        return sum(np.log((pred - self.data)**2))

    def n_parameters(self):
        return len(self.starting_parameters)

    def evaluateS1(self, parameters):
        return self(parameters), self.funcs.GetErrorSensitivities(parameters, self.data)

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

    self.starting_parameters = [2.26E-04, 0.0699, 3.45E-05, 0.05462, 0.0873, 8.92E-03, 5.150E-3, 0.03158, 0.1524]

    plt.rcParams['axes.axisbelow'] = True

    # Check input arguments
    parser = argparse.ArgumentParser(
        description='Plot sensitivities of the Beattie model')
    parser = get_parser(data_reqd=True)
    parser.add_argument("-n", "--no_chains", default=5, help="number of chains to use")
    parser.add_argument("-l", "--chain_length", default=1000, help="length of chain to use")
    parser.add_argument("-p", "--protocol", default=None, help="length of chain to use")
    args = parser.parse_args()

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

    model = PintsWrapper(data, par, args, times)

    mcmc = pints.MCMCController(model, args.no_chains, args.no_chains*[starting_parameters], method=pints.HaarioBardenetACMC)
    mcmc.set_max_iterations(args.chain_length)

    chains = mcmc.run()
    print(chains)

if __name__ == "__main__":
    main()
    print("done")
