#!/usr/bin/env python3

import os
import unittest
import pints
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.integrate as integrate
import symengine as se
import argparse

import argparse
import matplotlib.pyplot as plt

import MarkovModels


# TODO Rewrite these tests using methods from code
class Test(unittest.TestCase):
    def test_run_functions(self):
        # Check input arguments
        parser = argparse.ArgumentParser(
            description='Plot sensitivities of the Beattie model')
        parser.add_argument("-p", "--plot", action='store_true', help="whether to plot figures or just save",
                            default=False)
        parser.add_argument("--dpi", type=int, default=100,
                            help="what DPI to use for figures")
        args = parser.parse_args()

        par = MarkovModels.Params()
        starting_parameters = [2.26E-04, 0.0699, 3.45E-05,
                               0.05462, 0.0873, 8.92E-03, 5.150E-3, 0.03158, 0.1524]
        # Create symbols for symbolic functions
        p, y, v = MarkovModels.common.CreateSymbols(par)

        # Choose starting parameters (from J Physiol paper)
        para = [2.26E-04, 0.0699, 3.45E-05, 0.05462,
                0.0873, 8.92E-03, 5.150E-3, 0.03158, 0.1524]

        # Create symbols for symbolic functions
        p, y, v = MarkovModels.CreateSymbols(par)

        # Define system equations and initial conditions
        k1 = p[0] * se.exp(p[1] * v)
        k2 = p[2] * se.exp(-p[3] * v)
        k3 = p[4] * se.exp(p[5] * v)
        k4 = p[6] * se.exp(-p[7] * v)

        # Write in matrix form taking y = ([C], [O], [I])^T
        A = se.Matrix([[-k1 - k3 - k4, k2 - k4, -k4],
                       [k1, -k2 - k3, k4], [-k1, k3 - k1, -k2 - k4 - k1]])
        B = se.Matrix([k4, 0, k1])
        rhs = np.array(A * y + B)

        data = pd.read_csv("data/averaged-data.txt", delim_whitespace=True)
        times = data["time"].values
        data = data["current"].values

        funcs = MarkovModels.MarkovModel(
            par, p, y, v, A, B, para, times)

        # Run Functions
        funcs.SimulateForwardModel(starting_parameters)
        funcs.SimulateForwardModelSensitivities(starting_parameters)

    def test_sens_limits(self):
        par = MarkovModels.Params()
        p, y, v = MarkovModels.CreateSymbols(par)
        reversal_potential = par.Erev
        para = np.array([2.07, 7.17E1, 3.44E-2, 6.18E1,
                         4.18E2, 2.58E1, 4.75E1, 2.51E1, 3.33E1])
        para = para*1E-3

        # Define system equations and initial conditions
        k1 = p[0] * se.exp(p[1] * v)
        k2 = p[2] * se.exp(-p[3] * v)
        k3 = p[4] * se.exp(p[5] * v)
        k4 = p[6] * se.exp(-p[7] * v)

        current_limit = (p[-1]*(par.holding_potential - reversal_potential)
                         * k1/(k1+k2) * k4/(k3+k4)).subs(v, par.holding_potential)
        print("{} Current limit computed as {}".format(
            __file__, current_limit.subs(p, para).evalf()))

        sens_inf = [float(se.diff(current_limit, p[j]).subs(
            p, para).evalf()) for j in range(0, par.n_params)]
        print("{} sens_inf calculated as {}".format(__file__, sens_inf))

        k = se.symbols('k1, k2, k3, k4')

        A = se.Matrix([[-k1 - k3 - k4, k2 - k4, -k4],
                       [k1, -k2 - k3, k4], [-k1, k3 - k1, -k2 - k4 - k1]])
        B = se.Matrix([k4, 0, k1])

        # Use results from HH equations
        current_limit = (p[-1]*(par.holding_potential - reversal_potential)
                         * k1/(k1+k2) * k4/(k3+k4)).subs(v, par.holding_potential)

        funcs = MarkovModels.MarkovModel(par, p, y, v, A, B, para, [0])
        sens_inf = [float(se.diff(current_limit, p[j]).subs(
            p, para).evalf()) for j in range(0, par.n_params)]

        sens = funcs.SimulateForwardModelSensitivities(para)[1]

        # Check sens = sens_inf
        error = np.abs(sens_inf - sens)

        equal = np.all(error < 1e-10)
        assert(equal)
        return


if __name__ == "__main__":
    unittest.main()
