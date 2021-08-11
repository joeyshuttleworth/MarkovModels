#!/usr/bin/env python3

# Found optimal parameters [2.73318378e+07 3.44828704e+06 8.02615766e+05 2.52224782e+05 9.42360426e+04 2.67322394e+04 1.53260080e+03 6.84216241e+03 4.49633961e+03]

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

from  settings import Params
from  sensitivity_equations import GetSensitivityEquations, CreateSymbols
from  common import *
from  simulate_beattie_model_sensitivities import simulate_sine_wave_sensitivities

def main(args, output_dir="", ms_to_remove_after_spike=50):
    # Load defaults
    par = Params()

    # Load data
    if not os.path.exists(args.data_file_path):
        print("Input file not provided. Doing nothing.")
        return
    data = pd.read_csv(args.data_file_path, delim_whitespace=True)
    skip = int(par.timestep/0.1)

    starting_parameters = [2.26E-04, 0.0699, 3.45E-05, 0.05462, 0.0873, 8.92E-03, 5.150E-3, 0.03158, 0.1524]

    # Process data
    if ms_to_remove_after_spike == 0:
        indices_to_remove = None
    else:
        spikes = [2500, 3000, 5000, 15000, 20000, 30000, 65000, 70000]
        indices_to_remove = [[spike, spike + ms_to_remove_after_spike*10] for spike in spikes]
    indices_to_use = remove_indices(list(range(80000)), indices_to_remove)

    dat = data.values[indices_to_use]
    times=dat[:,0]
    values=dat[:,1]
    nobs = times.shape[0]
    print("Number of observations is {}".format(nobs))

    output_dir = os.path.join(args.output, output_dir)
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

    voltage=None
    if args.protocol=="sine_wave":
        voltage = beattie_sine_wave

    funcs = GetSensitivityEquations(par, p, y, v, A, B, starting_parameters, times, voltage=voltage)

    # Ensure directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.rcParams['axes.axisbelow'] = True

    found_parameters, found_value = fit_model(funcs, values, starting_parameters, par)

    print("finished! found parameters : {} ".format(found_parameters, found_value))
    s_variance = found_value/(nobs-1)
    print("Sample variance of residues is {}".format(s_variance))

    # plot fit
    current = funcs.SimulateForwardModel(starting_parameters)
    plt.plot(times, values)
    plt.plot(funcs.times, current)

    if args.plot:
       plt.show()

    else:
        plt.savefig(os.path.join(output_dir, "original_plot"))


    # Find error sensitivities
    current, sens = funcs.SimulateForwardModelSensitivities(found_parameters)
    sens = (sens * found_parameters[None,:]).T

    for i, vec in enumerate(sens):
        plt.plot(times, vec, label="state_variable".format(i))

    plt.title("Output sensitivities for four gate Markov Model")
    if args.plot:
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, "output_sensitivities"))
        plt.clf()

    # Estimate the various of the i.i.d Gaussian noise
    nobs = len(times)
    sigma2 = sum((current - values)**2)/(nobs-1)
    print("sigma2 = {}".format(sigma2))

    plt.plot(data["time"], data["current"], label="averaged data")
    plt.plot(times, current, label="current")
    plt.legend()
    if args.plot:
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, "fit"))
        plt.clf()

    return times, found_parameters

if __name__ == "__main__":
    spike_removal_durations=[0,50,100,150,200,1000]
    parser = get_parser(data_reqd=True)
    parser.add_argument("-r", "--remove", default=spike_removal_durations, help="ms of data to ignore after each capacitive spike", nargs='+', type=int)
    parser.add_argument("-v", "--protocol", default=None, help="name of the protocol to use")

    args = parser.parse_args ()
    data  = pd.read_csv(args.data_file_path, delim_whitespace=True)
    for val in args.remove:
        output_dir = os.path.join("{}ms_removed".format(val))
        times, params = main(args, output_dir, val)
        simulate_sine_wave_sensitivities(args, times, dirname=output_dir, para=params, data=data)
    print("done")
