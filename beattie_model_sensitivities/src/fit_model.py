#!/usr/bin/env python3

import os
import math
import pints
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.integrate as integrate
import symengine as se
import matplotlib.pyplot as plt
import matplotlib
import argparse

from   settings import Params
from   sensitivity_equations import GetSensitivityEquations, CreateSymbols
from   scipy.stats import chi2

class PintsWrapper(pints.ForwardModelS1):

    def __init__(self, settings, args, times_to_use):
        par = Params()
        self.times_to_use = times_to_use
        self.starting_parameters = [2.26E-04, 0.0699, 3.45E-05, 0.05462, 0.0873, 8.92E-03, 5.150E-3, 0.03158, 0.1524]
        # Create symbols for symbolic functions
        p, y, v = CreateSymbols(settings)

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

        self.funcs = GetSensitivityEquations(par, p, y, v, A, B, para, times_to_use, sine_wave=args.sine_wave)

    def n_parameters(self):
        return len(self.starting_parameters)

    def simulate(self, parameters, times):
        ret = self.funcs.SimulateForwardModel(parameters)
        # print(ret.shape)
        return ret

    def simulateS1(self, parameters, times):
        return self.funcs.SimulateForwardModelSensitivites(parameters, data), self.times_to_use, 1



class Boundaries(pints.Boundaries):
    def check(self, parameters):
        '''Check that each rate constant lies in the range 1.67E-5 < A*exp(B*V) < 1E3
        '''

        for i in range(0, 4):
            alpha = parameters[2*i]
            beta  = parameters[2*i + 1]

            vals = [0,0]
            vals[0] = alpha * np.exp(beta * -90 * 1E-3)
            vals[1] = alpha * np.exp(beta *  50 * 1E-3)

            for val in vals:
                if val < 1.67E-5 or val > 1E3:
                    return False
        # Check maximal conductance
        if parameters[8] > 0 and parameters[8] < 2:
            return True
        else:
            return False

    def n_parameters(self):
        return 9

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
    starting_parameters = [3.87068845e-04, 5.88028759e-02, 6.46971727e-05, 4.87408447e-02, 8.03073893e-02, 7.36295506e-03, 5.32908518e-03, 3.32254316e-02, 6.56614672e-02]

    plt.rcParams['axes.axisbelow'] = True

    # Check input arguments
    parser = argparse.ArgumentParser(
        description='Plot sensitivities of the Beattie model')
    parser.add_argument("data_file_path", help="path to csv data for the model to be fit to")
    parser.add_argument("-s", "--sine_wave", action='store_true', help="whether or not to use sine wave protocol",
        default=False)
    parser.add_argument("-p", "--plot", action='store_true', help="whether to plot figures or just save",
        default=False)
    parser.add_argument("--dpi", type=int, default=100, help="what DPI to use for figures")
    parser.add_argument("-o", "--output", type=str, default="output", help="The directory to output figures and data to")
    args = parser.parse_args()

    # Create output directory
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    if not os.path.exists(args.data_file_path):
        print("Input file not provided. Doing nothing.")
        return

    par = Params()

    skip = int(par.timestep/0.1)

    data  = pd.read_csv(args.data_file_path, delim_whitespace=True)
    dat = extract_times(data.values, indices_to_use, skip)
    times=dat[:,0]
    values=dat[:,1]

    model = PintsWrapper(par, args, times)

    current = model.simulate(starting_parameters, times)
    problem = pints.SingleOutputProblem(model, times, values)
    error = pints.SumOfSquaresError(problem)
    boundaries  = Boundaries()
    x0 = np.array([0.1]*9)
    # found_parameters, found_value = pints.optimise(error, starting_parameters, boundaries=boundaries)
    found_parameters = [1.68482689e-05, -1.19434719e-01, 1.72795030e-05, 3.34388390e-01, 4.42337711e-01,  1.56335228e-01, 1.76629398e-03, -6.59798458e-01, 2.45905373e-01]
    found_value = 166.01906668991023

    print("finished! found parameters : {} ".format(found_parameters, found_value))

    # Find error sensitivities
    funcs = model.funcs
    sens = funcs.SimulateForwardModelSensitivities(found_parameters)[1]
    FIM = sens @ sens.T
    cov = FIM**-1
    eigvals = np.linalg.eigvals(FIM)
    cov_ellipse(cov[0:2,0:2])
    print('Eigenvalues of FIM:\n{}'.format(eigvals))
    print("Covariance matrix is {}".format(cov))

def cov_ellipse(cov, q=0.95, nsig=None, **kwargs):
    """
    Parameters
    ----------
    copied from stackoverflow


    cov : (2, 2) array
        Covariance matrix.
    q : float, optional
        Confidence level, should be in (0, 1)
    nsig : int, optional
        Confidence level in unit of standard deviations.
        E.g. 1 stands for 68.3% and 2 stands for 95.4%.

    Returns
    -------
    width, height, rotation :
         The lengths of two axises and the rotation angle in degree
    for the ellipse.
    """

    if q is not None
        q = np.asarray(q)
    elif nsig is not None:
        q = 2 * norm.cdf(nsig) - 1
    else:
        raise ValueError('One of `q` and `nsig` should be specified.')
    r2 = chi2.ppf(q, 2)

    val, vec = np.linalg.eigh(cov)
    print(val)
    width, height = 2 * np.sqrt(val[:, None] * r2)
    rotation = np.degrees(np.arctan2(*vec[::-1, 0]))

    fig = plt.figure(0)
    e = matplotlib.patches.Ellipse([0,0], width, height, rotation)
    ax = fig.add_subplot(111, aspect='equal')
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    plt.show()
    print(width, height)
    return width, height, rotation


if __name__ == "__main__":
    main()
    print("done")
