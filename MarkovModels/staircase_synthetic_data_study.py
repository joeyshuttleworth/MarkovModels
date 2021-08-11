#!/usr/bin/env python3

import numpy as np
import pandas as pd
import scipy
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
import plotly.graph_objects as go

from settings import Params
from common import *

from MarkovModel import MarkovModel

# Set noise level - based on results from fitting the sine_wave model
sigma2 = 0.006


def draw_likelihood_surface(
        funcs, paras, params_to_change, ranges, data, args, output_dir=None):
    """
    Draw a heatmap of the log-likelihood surface when two parameters are changed
    The likeihood is assumed to be based of i.i.d Gaussian error
    """
    true_vals = paras[params_to_change[0]], paras[params_to_change[1]]
    print("True values of parameters are {}".format(true_vals))
    n = len(funcs.times)
    if data is None:
        data = np.linspace(0, 0, n)
    labels = ["p{}".format(p + 1) for p in params_to_change]

    def llxy(x, y):
        p = np.copy(paras)
        p[params_to_change[0]] = x
        p[params_to_change[1]] = y

        y = funcs.SimulateForwardModel(p)

        ll = -n * 0.5 * np.log(2 * np.pi) - n * 0.5 * np.log(sigma2) - \
            ((y - data)**2).sum() / (2 * sigma2)
        return ll

    # log likelihood of true parameters values
    print("log likelihood of true values {}".format(
        llxy(*paras[params_to_change])))

    fix_params = [i for i in range(len(paras)) if i not in params_to_change]
    print(fix_params)

    mle, val = fit_model(
        funcs, data, paras, fix_parameters=fix_params, max_iterations=args.max_iterations)

    # Compute SSE
    p = np.copy(paras)
    p[params_to_change[0]] = mle[0]
    p[params_to_change[1]] = mle[1]
    pred, S1 = funcs.SimulateForwardModelSensitivities(p)
    noise_level = (pred - data).var()
    print("sigma2 = {}".format(noise_level))
    print("sum square errors for mle is {}".format(((pred - data)**2).sum()))

    print("mle is {}".format(mle))

    ll_of_mle = llxy(*mle)
    print("log likelihood of mle values {}".format(ll_of_mle))

    # Discard columns that are fixed
    S1 = S1[:, params_to_change]
    H = np.dot(S1.T, S1)
    cov = noise_level * np.linalg.inv(H)
    eigvals, eigvecs = np.linalg.eigh(cov)
    assert(np.all(eigvals > 0))

    # TODO Make this work in general
    major_axis_angle = np.arctan2(min(eigvals), max(eigvals))
    window_size = [10 * eigvecs[0, 1] * np.max(eigvals), 0.002]

    def paraboloid(x,y):
        return ll_of_mle - (0.5 * H[0, 0] * x**2 + H[1, 0] * x * y + 0.5 * H[1, 1] * y**2) / noise_level

    if args.heatmap:
        xs = np.linspace(
            mle[0] -
            window_size[0] /
            2,
            mle[0] +
            window_size[0] /
            2,
            100)
        ys = np.linspace(
            mle[1] -
            window_size[1] /
            2,
            mle[1] +
            window_size[1] /
            2,
            100)
        zs = np.array([[max(ll_of_mle - 100, llxy(x, y))
                        for x in xs] for y in ys])

        approximate_zs = np.array([[paraboloid(x, y)
                                    for x in xs - mle[0]] for y in ys - mle[1]])

        # Plot surface
        # fig = go.Figure(data=[go.Surface(z=zs,x=xs,y=ys, opacity=0.5), go.Surface(z=approximate_zs,x=xs,y=ys, opacity=0.5, showscale=False)])
        # fig.show()

        l_a = xs.min()
        r_a = xs.max()
        l_b = ys.min()
        r_b = ys.max()
        l_z, r_z = ll_of_mle - 50, ll_of_mle

        figure, axes = plt.subplots()

        c = axes.pcolormesh(
            xs,
            ys,
            zs,
            vmin=l_z,
            vmax=r_z,
            label="Unnormalised log likelihood",
            shading="auto")
        axes.axis([l_a, r_a, l_b, r_b])

        plt.plot(mle[0], mle[1], "o", label="maximum likelihood estimator",
                 color="red", fillstyle="none")

        figure.colorbar(c)

        plt.xlabel(labels[0])
        plt.ylabel(labels[1])

        # Draw confidence region
        confidence_levels = [0.5, 0.95]
        mle_paras = np.copy(paras)

        for i, j in enumerate(params_to_change):
            mle_paras[j] = mle[i]

        # Parameters have been normalised to 1
        cov_ellipse(cov, q=confidence_levels, offset=mle, new_figure=False)

        plt.plot(true_vals[0], true_vals[1], "x",
                 label="true value of parameters", color="black")
        plt.legend()

        if output_dir is None:
            plt.show()
        else:
            plt.savefig(os.path.join(output_dir, "heatmap_{}_{}.pdf".format(
                *(np.array(params_to_change) + 1))))
            plt.clf()

        plt.plot(funcs.times, data, "x", label="synthetic data")
        plt.plot(funcs.times, funcs.SimulateForwardModel(
            paras), label="True model response")
        plt.plot(funcs.times, funcs.SimulateForwardModel(
            mle_paras), label="Fitted model response")
        plt.legend()

        if output_dir is None:
            plt.show()
        else:
            plt.savefig(os.path.join(
                output_dir, "synthetic_data_generation.pdf"))
            plt.clf()

    if args.heatmap:
        plot_likelihood_cross_sections(llxy, paraboloid, cov, mle, output_dir)

    if args.mcmc:
        do_mcmc(llxy, mle, args, output_dir)
    return mle


def do_mcmc(llxy, starting_pos, args, output_dir=None):
    class pints_likelihood(pints.LogPDF):
        def __call__(self, p):
            return llxy(*p)

        def n_parameters(self):
            return 2
    prior = pints.UniformLogPrior([0, 0], [1, 1])
    posterior = pints.LogPosterior(pints_likelihood(), prior)
    mcmc = pints.MCMCController(posterior, args.no_chains, np.tile(
        starting_pos, [args.no_chains, 1]), method=pints.HaarioBardenetACMC)
    mcmc.set_max_iterations(args.chain_length)
    chains = mcmc.run()
    if args.burn_in is not None:
        chains = chains[:, args.burn_in:, :]
    for i, chain in enumerate(chains):
        np.savetxt(os.path.join(output_dir, "chain_{}.csv".format(i)),
                   chain, delimiter=",")

    # Plot histograms
    try:
        pints.plot.histogram(chains, parameter_names=["p5", "p7"], kde=True)
    except(np.LinAlgError):
        print("Failed to draw histograms")
    if args.plot:
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, "histogram.pdf"))
    return


def plot_likelihood_cross_sections(
        likelihood, paraboloid, cov, mle, output_dir=None):
    assert(cov.shape == (2, 2))
    eigvals, eigvecs = np.linalg.eigh(cov)
    ll_of_mle = likelihood(*mle)
    for i, [e, l] in enumerate(zip(eigvecs, eigvals.T)):
        print(e, l)
        # Orientate nicely
        if e[0] < 0:
            e = -e
        # Plot across a 99 percent confidence interval
        q_val = 0.975
        r2 = scipy.stats.chi2.ppf(q_val, 2)
        radius = np.sqrt(l * r2)
        angle = np.arctan2(*e)
        print(e, [np.cos(angle), np.sin(angle)])
        start_point = mle - radius * e
        end_point = mle + radius * e
        print("start and end points for plot: {}, {}".format(
            start_point, end_point))

        vecs = np.linspace(start_point, end_point, 100)
        plt.plot(vecs[:, 0], [paraboloid(*(vec - mle)) for vec in vecs],
                 "--", color="grey", label="parabola approximation")
        plt.plot(vecs[:, 0], [val if val > 0 else np.nan for val in [
                 likelihood(*vec) for vec in vecs]], label="likelihood")
        plt.axvline(mle[0], linestyle="--", label="mle")

        # Plot confidence region lines
        q_vals = [0.5, 0.95]
        colours = ["pink", "red"]
        for q, colour in zip(q_vals, colours):
            r2 = scipy.stats.chi2.ppf(q, 2)
            radius = np.sqrt(l * r2)
            plt.axvline(mle[0] + radius * e[0], linestyle="--", color=colour,
                        label="Approximate {}% confidence region boundary".format(int(q * 100)))
            plt.axvline(mle[0] - radius * e[0], linestyle="--", color=colour,
                        label="Approximate {}% confidence region boundary".format(int(q * 100)))
            plt.xlabel("p5")
        plt.legend()
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        if output_dir is not None:
            plt.savefig(os.path.join(
                output_dir, "likelihood_1d_plot_{}.pdf".format(i)))
            plt.clf()
        else:
            plt.show()
    return


def generate_synthetic_data(funcs, para, sigma2):
    nobs = len(funcs.times)
    rng = np.random.default_rng(2021)
    z = np.sqrt(sigma2) * rng.standard_normal(nobs)
    y = funcs.SimulateForwardModel(para)
    obs = y + z
    return obs


def main():
    # Check input arguments
    parser = get_parser()
    parser.add_argument("-m", "--heatmap", action='store_true')
    parser.add_argument("-n", "--no_chains", type=int,
                        default=3, help="number of chains to use")
    parser.add_argument("-l", "--chain_length", type=int,
                        default=1000, help="length of chain to use")
    parser.add_argument("-v", "--protocol", default=None,
                        help="name of the protocol to use")
    parser.add_argument("-b", "--burn_in", type=int,
                        default=None, help="amount of burn in to use")
    parser.add_argument(
        "-r",
        "--remove",
        default=50,
        help="ms of data to ignore after each capacitive spike",
        type=int)
    parser.add_argument(
        "-M",
        "--mcmc",
        default=50,
        help="Whether or not to perform mcmc on the synthetic data example.",
        action='store_true')
    parser.add_argument(
        "--max_iterations",
        default=None,
        help="The maximum number of iterations to use when performing the optimisation",
        type=int)
    args = parser.parse_args()

    par = Params()

    plot_dir = os.path.join(args.output, "staircase")

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Choose parameters (make sure conductance is the last parameter)
    para = np.array([2.07E-3, 7.17E-2, 3.44E-5, 6.18E-2,
                     20, 2.58E-2, 2, 2.51E-2, 3.33E-2])

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
    A = se.Matrix([[-k1 - k3 - k4, k2 - k4, -k4],
                   [k1, -k2 - k3, k4], [-k1, k3 - k1, -k2 - k4 - k1]])
    B = se.Matrix([k4, 0, k1])

    current_limit = (p[-1] * (par.holding_potential - reversal_potential)
                     * k1 / (k1 + k2) * k4 / (k3 + k4)).subs(v, par.holding_potential)
    print("{} Current limit computed as {}".format(
        __file__, current_limit.subs(p, para).evalf()))

    sens_inf = [float(se.diff(current_limit, p[j]).subs(p, para).evalf())
                for j in range(0, par.n_params)]
    print("{} sens_inf calculated as {}".format(__file__, sens_inf))

    staircase_protocol = get_staircase_protocol(par)
    times = np.linspace(0, 15000, 1000)
    funcs = MarkovModel(par, p, y, v, A, B, para, times,
                        voltage=staircase_protocol)
    ret = funcs.SimulateForwardModelSensitivities(para),
    current = ret[0][0]
    S1 = ret[0][1]

    S1n = S1 * np.array(para)[None, :]
    sens_inf_N = sens_inf * np.array(para)[None, :]

    param_labels = ['S(p' + str(i + 1) + ',t)' for i in range(par.n_params)]
    [plt.plot(funcs.times, sens, label=param_labels[i])
     for i, sens in enumerate(S1n.T)]
    [plt.axhline(s) for s in sens_inf_N[0, :]]
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
        plt.savefig(os.path.join(plot_dir, 'ForwardModel_SW.pdf'))

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
        plt.savefig(os.path.join(plot_dir, 'Eigenvalues_SW.png'))

    if args.plot:
        plt.show()

    cov = np.linalg.inv(H / sigma2)
    print("Covariance matrix is {}".format(cov))

    # Output covariance matrix to file
    cols = [r"\hat q_{}".format(i + 1) for i in range(0, cov.shape[0])]
    df_cov = pd.DataFrame(data=cov, columns=cols, index=cols)
    print(df_cov)
    print(df_cov.to_latex())

    evals, evecs = np.linalg.eig(cov)
    print(evals, evecs)

    draw_cov_ellipses(S1=S1n, plot_dir=plot_dir, sigma2=sigma2)

    # Draw log-likelihood surface using synthetic data
    para = np.array([2.26E-04, 0.0699, 3.45E-05, 0.05462,
                     0.0873, 8.92E-03, 5.150E-3, 0.03158, 0.1524])
    synthetic_data = generate_synthetic_data(funcs, para, sigma2)

    if args.heatmap or args.mcmc:
        likelihood, mle = draw_likelihood_surface(funcs, para, [4, 6], [[0, 0.2], [
                                                  0, 0.02]], synthetic_data, args, output_dir=plot_dir)
    elif args.mcmc:
        assert(false)


if __name__ == "__main__":
    plt.rcParams["figure.figsize"] = (8, 8)
    main()
