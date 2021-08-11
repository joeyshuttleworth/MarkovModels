#!/usr/bin/env python3

import numpy as np
import symengine as se
import matplotlib.pyplot as plt
import argparse
import os

from settings import Params
from sensitivity_equations import GetSensitivityEquations, CreateSymbols
from common import *


def simulate_sine_wave_sensitivities(args, times=[], dirname="", para=[], data=None):
    # Check input arguments
    par = Params()

    dirname = os.path.join(args.output, dirname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Choose parameters (from J Physiol paper)
    if para == []:
        para = [2.26E-04, 0.0699, 3.45E-05, 0.05462,
                0.0873, 8.92E-03, 5.150E-3, 0.03158, 0.1524]

    # Sigma copied from output of fit_model
    sigma2 = 0.0006

    # Create symbols for symbolic functions
    p, y, v = CreateSymbols(par)

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

    if times == []:
        times = np.linspace(0, par.tmax, par.tmax + 1)

    spikes = None
    if args.sine_wave:
        voltage = beattie_sine_wave
        # Capacitive spikes
        # spikes = [250, 300, 500, 1500, 2000, 3000, 6500, 7000]
    else:
        voltage = None

    funcs = GetSensitivityEquations(
        par, p, y, v, A, B, para, times, voltage=voltage)

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
    if spikes is not None:
        [ax1.axvline(spike, linestyle="--", color="red", alpha=0.25)
         for spike in spikes]
    ax2 = fig.add_subplot(412)
    ax2.plot(funcs.times, funcs.SimulateForwardModel(para), label="model fit")
    if data is not None:
        ax2.plot(data["time"], data["current"], label="data")
    ax2.legend()
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
        plt.savefig(os.path.join(dirname, 'ForwardModel_SW_' +
                                 str(args.sine_wave) + '.png'))

    H = np.dot(S1n.T, S1n)
    eigvals = np.linalg.eigvals(H)
    print('Eigenvalues of H:\n{}'.format(eigvals.real))

    fig = plt.figure(figsize=(6, 6), dpi=args.dpi)
    ax = fig.add_subplot(111)
    for i in eigvals:
        ax.axhline(y=i, xmin=0.25, xmax=0.75)
    ax.set_yscale('log')
    ax.set_xticks([])
    ax.grid(True)

    if not args.plot:
        plt.savefig(os.path.join(dirname, 'Eigenvalues_SW_' +
                                 str(args.sine_wave) + '.png'))

    if args.plot:
        plt.show()

    draw_cov_ellipses(para, par, S1n=S1n, sigma2=sigma2, plot_dir=dirname)


if __name__ == "__main__":
    parser = get_parser(data_reqd=False)
    args = parser.parse_args()
    simulate_sine_wave_sensitivities(args, dirname="sine_wave")
