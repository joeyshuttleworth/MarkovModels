#!/usr/bin/env python3

import numpy as np
import pandas as pd
import scipy.interpolate
import symengine as se
import sympy
import math
import matplotlib.pyplot as plt
import argparse
import os

import pints
import pints.plot

import scipy.optimize

from settings import Params
from sensitivity_equations import GetSensitivityEquations, CreateSymbols
from common import *

# Set noise level
sigma2 = 0.01

def draw_likelihood_surface(funcs, paras, params_to_change, ranges, data, output_dir=None):
    """
    Draw a heatmap of the log-likelihood surface when two parameters are changed
    The likeihood is assumed to be based of i.i.d Gaussian error
    """
    true_vals = paras[params_to_change[0]], paras[params_to_change[1]]
    n = len(funcs.times)
    if data is None:
        data = np.linspace(0, 0, n)
    labels = ["p{}".format(p+1) for p in params_to_change]

    def llxy(x,y):
        p = np.copy(paras)
        p[params_to_change[0]] = x
        p[params_to_change[1]] = y

        y = funcs.SimulateForwardModel(p)
        # sample_var = (y - data).var()
        sample_var = sigma2
        ll = -n*0.5*np.log(2*np.pi) - n*0.5*np.log(sample_var) -1/(2*sample_var)*sum(y - data)**2
        return max(ll, -10000)

    # log likelihood of true parameters values
    print("log likelihood of true values {}".format(llxy(*paras[params_to_change])))

    xs = np.linspace(ranges[0][0], ranges[0][1], 25)
    ys = np.linspace(ranges[1][0], ranges[1][1], 25)

    fix_params = [i for i in range(len(paras)) if i not in params_to_change]
    print(fix_params)

    mle = fit_model(funcs, data, paras, Params(), fix_parameters=fix_params)[0]
    # res = scipy.optimize.minimize(lambda p : -llxy(*p), true_vals)
    # mle = res.x

    print("mle is {}".format(mle))

    # log likelihood of true parameters values
    print("log likelihood of mle values {}".format(llxy(*mle)))

    zs = np.array([[llxy(x,y) for x in xs] for y in ys])

    l_a=xs.min()
    r_a=xs.max()
    l_b=ys.min()
    r_b=ys.max()
    l_z,r_z  = zs.min(), zs.max()

    figure, axes = plt.subplots()

    c = axes.pcolormesh(xs, ys, zs, vmin=l_z, vmax=r_z, label="Unnormalised log likelihood", shading="auto")
    axes.set_title('Log Likelihood Surface')
    axes.axis([l_a, r_a, l_b, r_b])

    plt.plot(mle[0], mle[1], "x", label="Maximum likelihood estimator", color="red")

    figure.colorbar(c)

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    # Draw confidence region
    confidence_levels = [0.5, 0.95]
    mle_paras = np.copy(paras)

    for i,j in enumerate(params_to_change):
        mle_paras[j] = mle[i]

    pred, S1n = funcs.SimulateForwardModelSensitivities(mle_paras)
    # S1n = S1n * mle_paras[None,:]
    sample_var = sigma2

    # Discard rows that are fixed
    S1n = S1n[:,params_to_change]
    H   = np.dot(S1n.T, S1n)
    print(H)

    cov = sample_var*np.linalg.inv(H)
    cov_ellipse(cov, q=confidence_levels, offset=mle, new_figure=False) # Parameters have been normalised to 1

    plt.plot(true_vals[0], true_vals[1], "x", label="true value of parameters", color="black")
    plt.legend()

    if output_dir is None:
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, "heatmap_{}_{}.pdf".format(*(np.array(params_to_change)+1))))
        plt.clf()

    plt.plot(funcs.times, data, "x", label="synthetic data")
    plt.plot(funcs.times, funcs.SimulateForwardModel(paras), label="True model response")
    plt.plot(funcs.times, funcs.SimulateForwardModel(mle_paras), label="Fitted model response")
    plt.legend()

    if output_dir is None:
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, "synthetic_data_generation.pdf"))

    eigvals, eigvecs = np.linalg.eigh(H)
    max(eigvals)

    if np.abs(eigvals[0]) < np.abs(eigvals[1]):
        eigvec = eigvecs[0]
    else:
        eigvec = eigvecs[1]
    # Plot the true values

    if eigvec[0] < 0 and eigvec[1] < 0:
        eigvec = -eigvec
    elif eigvec [0] < 0 or eigvec[1] < 0:
        assert(False)

    scale_factor = (20 - mle[0])/eigvec[0]
    start_point = mle
    end_point = mle + eigvec*scale_factor
    print("start and end points for plot: {}, {}".format(start_point, end_point))

    class likelihood(pints.LogPDF):
        def __call__(self, p):
            return max(llxy(*p), -100)
        def n_parameters(self):
            return 2

    pints.plot.function_between_points(likelihood(), start_point, end_point, evaluations=500, padding=0.01)

    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, "likelihood_1d_plot.pdf"))
        plt.clf()
    else:
        plt.show()

    return

def generate_synthetic_data(funcs, para, sigma2):
    nobs = len(funcs.times)
    y = funcs.SimulateForwardModel(para)
    z = np.random.normal(0, sigma2, size=nobs)
    obs = y + z
    return obs

def main():
    # Check input arguments
    parser = get_parser()
    parser.add_argument("-m", "--heatmap", action='store_true')
    args = parser.parse_args()

    par = Params()

    plot_dir = os.path.join(args.output, "staircase")

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Choose parameters (make sure conductance is the last parameter)
    # para = np.array([2.07E-3, 7.17E-2, 3.44E-5, 6.18E-2, 4.18E-1, 2.58E-2, 4.75E-2, 2.51E-2, 3.33E-2])
    para = np.array([2.07E-3, 7.17E-2, 3.44E-5, 6.18E-2, 20, 2.58E-2, 2, 2.51E-2, 3.33E-2])

    # Compute resting potential for 37 degrees C
    # reversal_potential = calculate_reversal_potential(temp=37)
    # par.Erev = reversal_potential
    reversal_potential = par.Erev
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

    staircase_protocol = get_staircase_protocol(par)
    times = np.linspace(0, 15000, 1000)
    funcs = GetSensitivityEquations(par, p, y, v, A, B, para, times, voltage=staircase_protocol)
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
    spikes = detect_spikes(funcs.times, funcs.GetVoltage())
    [ax1.axvline(spike, "--", color='red', alpha=0.3) for spike in spikes]
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

    cov = np.linalg.inv(H/sigma2)
    print("Covariance matrix is {}".format(cov))

    # Output covariance matrix to file
    cols = ["\hat q_{}".format(i + 1) for i in range(0, cov.shape[0])]
    df_cov = pd.DataFrame(data=cov, columns=cols, index=cols)
    print(df_cov)
    print(df_cov.to_latex())

    evals, evecs = np.linalg.eig(cov)
    print(evals, evecs)

    draw_cov_ellipses(para, par, S1n=S1n, plot_dir=plot_dir, sigma2=sigma2)

    # Draw log-likelihood surface using synthetic data
    para = np.array([2.26E-04, 0.0699, 3.45E-05, 0.05462, 0.0873, 8.92E-03, 5.150E-3, 0.03158, 0.1524])
    synthetic_data = generate_synthetic_data(funcs, para, sigma2)

    if args.heatmap:
        draw_likelihood_surface(funcs, para, [4,6], [[0, 0.2],[0, 0.02]], synthetic_data, output_dir=plot_dir)

if __name__=="__main__":
    # Seed numpy
    np.random.seed(20211507)
    main()
