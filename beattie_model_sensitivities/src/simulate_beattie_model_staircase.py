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
from common import calculate_reversal_potential, get_parser

def main():
    # Check input arguments
    parser = get_parser()
    args = parser.parse_args()

    par = Params()

    plot_dir = os.path.join(args.output, "staircase")

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Choose parameters (make sure conductance is the last parameter)
    para = np.array([2.07E-3, 7.17E-2, 3.44E-5, 6.18E-2, 4.18E-1, 2.58E-2, 4.75E-2, 2.51E-2, 3.33E-2])

    # Compute resting potential for 37 degrees C
    reversal_potential = calculate_reversal_potential(temp=37)
    par.Erev = reversal_potential
    print("reversal potential is {}".format(reversal_potential))

    # Create symbols for symbolic functions
    p, y, v = CreateSymbols(par)

    k = se.symbols('k1, k2, k3, k4')

    # Define system equations and initial conditions
    k1 = p[0] * se.exp(p[1] * v)
    k2 = p[2] * se.exp(-p[3] * v)
    k3 = p[4] * se.exp(p[5] * v)
    k4 = p[6] * se.exp(-p[7] * v)

    # Notation is consistent between the two papers
    A = se.Matrix([[-k1 - k3 - k4, k2 -  k4, -k4], [k1, -k2 - k3, k4], [-k1, k3 - k1, -k2 - k4 - k1]])
    B = se.Matrix([k4, 0, k1])

    current_limit = (p[-1]*(par.holding_potential - reversal_potential) * k1/(k1+k2) * k4/(k3+k4)).subs(v, par.holding_potential)
    print("{} Current limit computed as {}".format(__file__, current_limit.subs(p, para).evalf()))

    sens_inf = [float(se.diff(current_limit, p[j]).subs(p, para).evalf()) for j in range(0, par.n_params)]
    print("{} sens_inf calculated as {}".format(__file__, sens_inf))

    protocol = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "protocols", "protocol-staircaseramp.csv"))

    times = 1000*protocol["time"].values
    voltages = protocol["voltage"].values

    spikes = 1000*detect_spikes(protocol["time"], protocol["voltage"])

    staircase_protocol = scipy.interpolate.interp1d(times, voltages, kind="linear")
    staircase_protocol_safe = lambda t : staircase_protocol(t) if t < times[-1] else par.holding_potential

    funcs = GetSensitivityEquations(par, p, y, v, A, B, para, times, voltage=staircase_protocol_safe)
    ret = funcs.SimulateForwardModelSensitivities(para),
    current = ret[0][0]
    S1 = ret[0][1]

    S1n = S1 * np.array(para)[None, :]
    sens_inf_N = sens_inf * np.array(para)[None,:]

    param_labels = ['S(p' + str(i + 1) + ',t)' for i in range(par.n_params)]
    [plt.plot(funcs.times, sens, label=param_labels[i]) for i, sens in enumerate(S1n.T)]
    [plt.axhline(s) for s in sens_inf_N[0,:]]
    plt.legend()
    plt.xlabel("time /ms")
    plt.ylabel("dI(t)/dp")
    if args.plot:
        plt.show()
    else:
        plt.savefig(os.path.join(plot_dir, "sensitivities_plot"))

    state_variables = funcs.GetStateVariables(para)
    state_labels = ['C', 'O', 'I', 'IC']

    param_labels = ['S(p' + str(i + 1) + ',t)' for i in range(par.n_params)]

    fig = plt.figure(figsize=(8, 8), dpi=args.dpi)
    ax1 = fig.add_subplot(411)
    ax1.plot(funcs.times, funcs.GetVoltage())
    ax1.grid(True)
    ax1.set_xticklabels([])
    ax1.set_ylabel('Voltage (mV)')
    [ax1.axvline(spike, color='red') for spike in spikes]
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
        plt.savefig(os.path.join(plot_dir, 'ForwardModel_SW_{}.png'.format(args.sine_wave)))

    # Only take every 100th point
    # S1n = S1n[0:-1:10]
    H = np.dot(S1n.T, S1n)
    print(H)
    eigvals = np.linalg.eigvals(H)
    #Sigma2 - the observed variance. 1885 is the value taken from a fit
    sigma2 = 1885/(len(funcs.times) - 1)
    print('Eigenvalues of H:\n{}'.format(eigvals.real))

    # Plot the eigenvalues of H, shows the condition of H
    fig = plt.figure(figsize=(6, 6), dpi=args.dpi)
    ax = fig.add_subplot(111)
    for i in eigvals:
        ax.axhline(y=i, xmin=0.25, xmax=0.75)
    ax.set_yscale('log')
    ax.set_xticks([])
    ax.grid(True)

    if not args.plot:
        plt.savefig(os.path.join(plot_dir, 'Eigenvalues_SW_{}.png'.format(args.sine_wave)))

    if args.plot:
        plt.show()

    cov = np.linalg.inv(H*sigma2)
    for j in range(0, par.n_params):
        for i in range(j+1, par.n_params):
            parameters_to_view = np.array([i,j])
            # sub_sens = S1n[:,[i,j]]
            sub_cov = cov[parameters_to_view[:,None], parameters_to_view]
            # sub_cov = np.linalg.inv(np.dot(sub_sens.T, sub_sens)*sigma2)
            eigen_val, eigen_vec = np.linalg.eigh(sub_cov)
            eigen_val=eigen_val.real
            if eigen_val[0] > 0 and eigen_val[1] > 0:
                print("COV_{},{} : well defined".format(i, j))
                cov_ellipse(sub_cov, q=[0.75, 0.9, 0.99], offset=para[[i,j]])
                plt.ylabel("parameter {}".format(i+1))
                plt.xlabel("parameter {}".format(j+1))
                if args.plot:
                    plt.show()
                else:
                    plt.savefig(os.path.join(output_dir, "covariance_for_parameters_{}_{}".format(j+1,i+1)))
                plt.clf()
            else:
                print("COV_{},{} : negative eigenvalue: {}".format(i,j, eigen_val))


if __name__=="__main__":
    main()
