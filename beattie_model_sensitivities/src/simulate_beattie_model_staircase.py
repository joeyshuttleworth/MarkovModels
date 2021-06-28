#!/usr/bin/env python3

import numpy as np
import pandas as pd
import scipy.interpolate
import symengine as se
import sympy
import matplotlib.pyplot as plt
import argparse
import os

from settings import Params
from sensitivity_equations import GetSensitivityEquations, CreateSymbols
from common import calculate_reversal_potential

def main():
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

    # Choose parameters (make sure conductance is the last parameter)
    para = np.array([2.07, 7.17E1, 3.44E-2, 6.18E1, 4.18E2, 2.58E1, 4.75E1, 2.51E1, 3.33E1])
    para = para*1E-3

    para = np.array([1.87451202e-03, 1.36254787e-02, 1.68324276e-05, 8.77532812e-02, 5.67114947e-02, 2.66069061e-02, 1.21159939e-03, 7.96959925e-03, 5.49219181e-02])

    # Compute reversal potential for 37 degrees C
    reversal_potential = calculate_reversal_potential(temp=37)
    par.Erev = reversal_potential
    print("reversal potential is {}".format(reversal_potential))

    # Create symbols for symbolic functions
    p, y, v = CreateSymbols(par)

    k = se.symbols('k1, k2, k3, k4')
    print(k)

    # Define system equations and initial conditions
    k1 = p[0] * se.exp(p[1] * v)
    k2 = p[2] * se.exp(-p[3] * v)
    k3 = p[4] * se.exp(p[5] * v)
    k4 = p[6] * se.exp(-p[7] * v)

    # Notation is consistent between the two papers
    A = se.Matrix([[-k1 - k3 - k4, k2 -  k4, -k4], [k1, -k2 - k3, k4], [-k1, k3 - k1, -k2 - k4 - k1]])
    B = se.Matrix([k4, 0, k1])

    rhs = np.array(A * y + B)

    protocol = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "protocols", "protocol-staircaseramp.csv"))
    times = 10000*protocol["time"].values
    voltages = protocol["voltage"].values

    staircase_protocol = scipy.interpolate.interp1d(times, voltages, kind="linear")
    staircase_protocol_safe = lambda t : staircase_protocol(t) if t < times[-1] else par.holding_potential

    funcs = GetSensitivityEquations(par, p, y, v, A, B, para, times, voltage=staircase_protocol_safe)
    ret = funcs.SimulateForwardModelSensitivities(para),
    current = ret[0][0]
    S1 = ret[0][1]

    S1n = S1 * np.array(para)[None, :]

    state_variables = funcs.GetStateVariables(para)
    state_labels = ['C', 'O', 'I', 'IC']

    param_labels = ['S(p' + str(i + 1) + ',t)' for i in range(par.n_params)]

    fig = plt.figure(figsize=(8, 8), dpi=args.dpi)
    ax1 = fig.add_subplot(411)
    ax1.plot(funcs.times, funcs.GetVoltage())
    ax1.grid(True)
    ax1.set_xticklabels([])
    ax1.set_ylabel('Voltage (mV)')
    ax2 = fig.add_subplot(412)
    ax2.plot(funcs.times, funcs.SimulateForwardModel(para))
    ax2.grid(True)
    ax2.set_xticklabels([])
    ax2.set_ylabel('Current (nA)')
    ax3 = fig.add_subplot(413)
    for i in range(par.n_state_vars + 1):
        ax3.plot(funcs.times, state_variables[:, i], label=state_labels[i])
    ax3.legend(ncol=4)
    ax3.grid(True)
    ax3.set_xticklabels([])
    ax3.set_ylabel('State occupancy')
    ax4 = fig.add_subplot(414)
    for i in range(par.n_params):
        ax4.plot(funcs.times, S1n[:, i], label=param_labels[i])
    ax4.legend(ncol=3)
    ax4.grid(True)
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Sensitivities')
    plt.tight_layout()

    if not args.plot:
        plt.savefig('ForwardModel_SW_' + str(args.sine_wave) + '.png')

    H = np.dot(S1n.T, S1n)
    print(H)
    # eigvals = np.linalg.eigvals(H)
    # print('Eigenvalues of H:\n{}'.format(eigvals.real))

    if not args.plot:
        plt.savefig('Eigenvalues_SW_' + str(args.sine_wave) + '.png')

    if args.plot:
        plt.show()

if __name__=="__main__":
    main()
