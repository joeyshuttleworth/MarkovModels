#!/usr/bin/env python3

import numpy as np
import symengine as se
import matplotlib.pyplot as plt
import argparse
import os

from settings import Params
from sensitivity_equations import GetSensitivityEquations, CreateSymbols

def main():
    plt.rcParams['axes.axisbelow'] = True
    # Check input arguments
    parser = argparse.ArgumentParser(
        description='Plot sensitivities of the Beattie model')
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

    par = Params()

    # Choose starting parameters (from J Physiol paper)
    para = [2.26E-04, 0.0699, 3.45E-05, 0.05462, 0.0873, 8.92E-03, 5.150E-3, 0.03158, 0.1524]

    # Create symbols for symbolic functions
    p, y, v = CreateSymbols(par)

    # Define system equations and initial conditions
    k1 = p[0] * se.exp(p[1] * v)
    k2 = p[2] * se.exp(-p[3] * v)
    k3 = p[4] * se.exp(p[5] * v)
    k4 = p[6] * se.exp(-p[7] * v)

    # C, I, O
    rhs = [k2 * y[2] + k4 * (p[8] - y[0] - y[1] - y[2]) - (k1 + k3) * y[0],
        k1 * (p[8] - y[0] - y[1] - y[2]) + k3 * y[2] - (k2 + k4) * y[1],
        k1 * y[0] + k4 * y[1] - (k2 + k3) * y[2]]

    ICs = [0.99, 0.01, 0.01]

    funcs = GetSensitivityEquations(par, p, y, v, rhs, ICs, sine_wave=args.sine_wave)

    S1 = funcs.SimulateForwardModelSensitivities(para)
    S1n = funcs.NormaliseSensitivities(S1, para)

    param_labels = ['S(p' + str(i + 1) + ',t)' for i in range(par.n_params)]

    fig = plt.figure(figsize=(8, 8), dpi=args.dpi)
    ax1 = fig.add_subplot(411)
    ax1.plot(funcs.GetVoltage())
    ax1.grid(True)
    ax1.set_xticklabels([])
    ax1.set_ylabel('Voltage (mV)')
    ax2 = fig.add_subplot(412)
    ax2.plot(funcs.SimulateForwardModel(para))
    ax2.grid(True)
    ax2.set_xticklabels([])
    ax2.set_ylabel('Current (nA)')
    ax3 = fig.add_subplot(413)
    for i in range(par.n_params):
        ax3.plot(S1n[:, i], label=param_labels[i])

    ax3.legend(ncol=3)
    ax3.grid(True)
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Sensitivities')
    plt.tight_layout()

    # Plot sensitivities w.r.t the initial conditions
    ## Get machine precision
    ##h = np.sqrt(np.finfo(np.float).eps)
    h=0.01

    ax4 = fig.add_subplot(414)
    for i in range(par.n_state_vars):
        new_ICs = ICs
        new_ICs[i] += h
        funcs.SetInitialConditions(new_ICs)
        results = []
        results.append(funcs.SimulateForwardModel(para))

        new_ICs[i] -= 2*h
        funcs.SetInitialConditions(new_ICs)
        results.append(funcs.SimulateForwardModel(para))

        derivative = ICs[i]*(results[1] - results[0])/(2*h)
        ax4.plot(derivative, label="{}".format(i))

    funcs.SetInitialConditions(ICs)

    ax4.legend(ncol=3)
    ax4.grid(True)
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Sensitivitiy of state variables')
    plt.tight_layout()


    if not args.plot:
        plt.savefig(os.path.join(args.output, 'ForwardModel_SW_' + str(args.sine_wave) + '.png'))

    H = np.dot(S1n.T, S1n)
    evals = np.linalg.eigvals(H)
    print('Eigenvalues of H:')
    eigvals = evals/np.max(evals)
    print(eigvals)

    fig = plt.figure(figsize=(6, 6), dpi=args.dpi)
    ax = fig.add_subplot(111)
    for i in eigvals:
        ax.axhline(y=i, xmin=0.25, xmax=0.75)
    ax.set_yscale('log')
    ax.set_xticks([])
    ax.grid(True)

    if not args.plot:
        plt.savefig('Eigenvalues_SW_' + str(args.sine_wave) + '.png')

    if args.plot:
        plt.show()

if __name__=="__main__":
    main()
