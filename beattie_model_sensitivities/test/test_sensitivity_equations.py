#!/usr/bin/env python3

import os
import pints
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.integrate as integrate
import symengine as se
import argparse

import argparse
import matplotlib.pyplot as plt

from context import Params, CreateSymbols, GetSensitivityEquations


def test_run_functions():
    # Check input arguments
    parser = argparse.ArgumentParser(
        description='Plot sensitivities of the Beattie model')
    parser.add_argument("-s", "--sine_wave", action='store_true', help="whether or not to use sine wave protocol",
        default=False)
    parser.add_argument("-p", "--plot", action='store_true', help="whether to plot figures or just save",
        default=False)
    parser.add_argument("--dpi", type=int, default=100, help="what DPI to use for figures")
    args = parser.parse_args()


    par = Params()
    starting_parameters = [2.26E-04, 0.0699, 3.45E-05, 0.05462, 0.0873, 8.92E-03, 5.150E-3, 0.03158, 0.1524]
    # Create symbols for symbolic functions
    p, y, v = CreateSymbols(par)

    # Choose starting parameters (from J Physiol paper)
    para = [2.26E-04, 0.0699, 3.45E-05, 0.05462, 0.0873, 8.92E-03, 5.150E-3, 0.03158, 0.1524]

    # Create symbols for symbolic functions
    p, y, v = CreateSymbols(par)

    # Define system equations and initial conditions
    k1 = p[0] * se.exp(p[1] * v)
    k2 = p[2] * se.exp(-p[3] * v)
    k3 = p[4] * se.exp(p[5] * v)
    k4 = p[6] * se.exp(-p[7] * v)

    # Write in matrix form taking y = ([C], [O], [I])^T
    A = se.Matrix([[-k1 - k3 - k4, k2 -  k4, -k4], [k1, -k2 - k3, k4], [-k1, k3 - k1, -k2 - k4 - k1]])
    B = se.Matrix([k4, 0, k1])
    rhs = np.array(A * y + B)

    data  = pd.read_csv("../data/averaged-data.txt", delim_whitespace=True)
    times = data["time"].values
    data  = data["current"].values

    funcs = GetSensitivityEquations(par, p, y, v, A, B, para, times, sine_wave=args.sine_wave)

    # Run Functions
    funcs.SimulateForwardModel(starting_parameters)
    funcs.SimulateForwardModelSensitivities(starting_parameters, data)


if __name__ == "__main__":
    test_run_functions()
