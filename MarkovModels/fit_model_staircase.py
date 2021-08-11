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

from settings import Params
from sensitivity_equations import GetSensitivityEquations, CreateSymbols
from common import get_args, get_parser, cov_ellipse, extract_times, remove_indices
from simulate_beattie_model_sensitivities import simulate_sine_wave_sensitivities


class PintsWrapper(pints.ForwardModelS1):

    def __init__(self, settings, args, times_to_use):
        par = Params()
        self.times_to_use = times_to_use
        self.starting_parameters = [
            2.26E-04, 0.0699, 3.45E-05, 0.05462, 0.0873, 8.92E-03, 5.150E-3, 0.03158, 0.1524]
        # Create symbols for symbolic functions
        p, y, v = CreateSymbols(settings)

        # Choose starting parameters (from J Physiol paper)
        para = [2.26E-04, 0.0699, 3.45E-05, 0.05462,
                0.0873, 8.92E-03, 5.150E-3, 0.03158, 0.1524]

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
        protocol = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(
            os.path.realpath(__file__))), "protocols", "protocol-staircaseramp.csv"))
        times = 10000*protocol["time"].values
        voltages = protocol["voltage"].values

        staircase_protocol = scipy.interpolate.interp1d(
            times, voltages, kind="linear")

        def staircase_protocol_safe(t): return staircase_protocol(
            t) if t < times[-1] else par.holding_potential

        self.funcs = GetSensitivityEquations(
            par, p, y, v, A, B, para, times_to_use, voltage=staircase_protocol_safe)

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
            beta = parameters[2*i + 1]

            vals = [0, 0]
            vals[0] = alpha * np.exp(beta * -90 * 1E-3)
            vals[1] = alpha * np.exp(beta * 50 * 1E-3)

            for val in vals:
                if val < 1E-7 or val > 1E3:
                    return False
        # Check maximal conductance
        if parameters[8] > 0 and parameters[8] < 2:
            return True
        else:
            return False

    def n_parameters(self):
        return 9


def main(args, output_dir="", ms_to_remove_after_spike=50):
    output_dir = os.path.join(args.output, output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # Constants
    if ms_to_remove_after_spike == 0:
        indices_to_remove = None
    else:
        spikes = [2500, 3000, 5000, 15000, 20000, 30000, 65000, 70000]
        indices_to_remove = [
            [spike, spike + ms_to_remove_after_spike*10] for spike in spikes]

    indices_to_use = remove_indices(list(range(80000)), indices_to_remove)
    # indices_to_use = [[1,2499], [2549,2999], [3049,4999], [5049,14999], [15049,19999], [20049,29999], [30049,64999], [65049,69999], [70049,-1]]
    starting_parameters = [3.87068845e-04, 5.88028759e-02, 6.46971727e-05, 4.87408447e-02,
                           8.03073893e-02, 7.36295506e-03, 5.32908518e-03, 3.32254316e-02, 6.56614672e-02]

    plt.rcParams['axes.axisbelow'] = True

    data = pd.read_csv(args.data_file_path, delim_whitespace=True)

    print("outputting to {}".format(args.output))

    if not os.path.exists(args.data_file_path):
        print("Input file not provided. Doing nothing.")
        return

    par = Params()

    skip = int(par.timestep/0.1)
    dat = data.values[indices_to_use]

    times = dat[:, 0]
    values = dat[:, 1]

    model = PintsWrapper(par, args, times)

    current = model.simulate(starting_parameters, times)

    if args.plot:
        plt.plot(times, values)
        plt.plot(model.times_to_use, current)
        plt.show()

    problem = pints.SingleOutputProblem(model, times, values)
    error = pints.SumOfSquaresError(problem)
    boundaries = Boundaries()
    x0 = starting_parameters

    found_parameters, found_value = pints.optimise(
        error, starting_parameters, boundaries=boundaries)
    # found_parameters = np.array([2.26E-04, 0.0699, 3.45E-05, 0.05462, 0.0873, 8.92E-03, 5.150E-3, 0.03158, 0.1524])
    # found_value = 100

    print("finished! found parameters : {} ".format(
        found_parameters, found_value))

    # Find error sensitivities
    funcs = model.funcs
    current, sens = funcs.SimulateForwardModelSensitivities(found_parameters)
    sens = (sens * found_parameters[None, :]).T

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

    # Compute the Fischer information matrix
    FIM = sens @ sens.T/sigma2
    cov = FIM**-1
    eigvals = np.linalg.eigvals(FIM)
    for i in range(0, par.n_params):
        for j in range(i+1, par.n_params):
            parameters_to_view = np.array([i, j])
            sub_cov = cov[parameters_to_view[:, None], parameters_to_view]
            eigen_val, eigen_vec = np.linalg.eigh(sub_cov)
            eigen_val = eigen_val.real
            if eigen_val[0] > 0 and eigen_val[1] > 0:
                print("COV_{},{} : well defined".format(i, j))
                cov_ellipse(sub_cov, q=[0.75, 0.9, 0.99])
                plt.ylabel("parameter {}".format(i))
                plt.xlabel("parameter {}".format(j))
                if args.plot:
                    plt.show()
                else:
                    plt.savefig(os.path.join(
                        output_dir, "covariance_for_parameters_{}_{}".format(i, j)))
                plt.clf()
            else:
                print("COV_{},{} : negative eigenvalue".format(i, j))

    print('Eigenvalues of FIM:\n{}'.format(eigvals))
    print("Covariance matrix is: \n{}".format(cov))

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
    spike_removal_durations = [0, 50, 100, 150, 200, 1000]
    parser = get_parser(data_reqd=True)
    parser.add_argument("-r", "--remove", default=spike_removal_durations,
                        help="ms of data to ignore after each capacitive spike", nargs='+', type=int)
    args = parser.parse_args()
    data = pd.read_csv(args.data_file_path, delim_whitespace=True)
    for val in args.remove:
        output_dir = "{}ms_removed".format(val)
        times, params = main(args, output_dir, val)
        simulate_sine_wave_sensitivities(
            args, times, dirname=output_dir, para=params, data=data)
    print("done")
