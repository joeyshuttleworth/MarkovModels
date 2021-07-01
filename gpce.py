import pints
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import scipy.integrate as integrate
import chaospy
from hh_markov_model import ChannelModel


def simulate(parameters, ts, first_V, second_V):
    mdl = ChannelModel(parameters, lambda t : first_V)

    ts = np.array(ts)

    # Set model to its steady state
    CK, eigenvalues, X2 = mdl.getLinearSolution(0)

    # Do a step to the second voltage
    mdl.setStates(np.array(X2))
    mdl.setProtocol(lambda t: second_V)

    # Get solution
    CK, eigenvalues, X2 = mdl.getLinearSolution(0)

    # Calculate current after 0.5 seconds
    states = [CK * np.matrix(np.exp(eigenvalues*t)).T + X2 for t in ts]
    I = [mdl.calculateCurrent(state[1])[0,0] for state in states]

    return I

def do_gpce(forward_problem, starting_parameters, first_V, second_V, distribution, quad_order=20, poly_order=30, plot=True):

    # Do the pseudo spectral projection
    abscissas, weights = chaospy.generate_quadrature(quad_order, distribution, rule="gaussian")

    polynomial_expansion = chaospy.generate_expansion(poly_order, distribution)
    evaluations = [forward_problem(abscissa[0]) for abscissa in abscissas.T]

    foo_approx = chaospy.fit_quadrature(polynomial_expansion, abscissas, weights, evaluations)
    expected = chaospy.E(foo_approx, distribution)
    std = chaospy.Std(foo_approx, distribution)

    print(expected, std)
    print(foo_approx)
    if plot:
        xs = np.linspace(distribution.mom(1)-math.sqrt(distribution.mom(2))*5,distribution.mom(1) + math.sqrt(distribution.mom(2))*5, 50)
        fig, ax = plt.subplots()
        ax.set_xlabel("p8")
        ax.axvline(distribution.mom(1), linestyle="--", label="mean parameter value")
        ax.plot(xs, [foo_approx(x)[0].sum() for x in xs], label="gpce approximation" ,color="blue")
        ax.plot(xs, [forward_problem(x)[0] for x in xs], label="true value", color="red")
        ax.set_ylabel("current /nA")

        ax2 = ax.twinx()
        x2s = np.linspace(xs[0], xs[-1], 10000)
        ax2.plot(xs, distribution.pdf(xs), "--", label="probability density", color="green")

        ax.legend()
        fig.savefig("normal_pdf_{}mV_{}mV.pdf".format(first_V, second_V))

        fig, ax = plt.subplots()
        ax.fill_between(coordinates, expected-std, expected+std, alpha=0.3)
        ax.plot(coordinates, expected)
        ax.set_xlabel("time /ms")
        ax.set_ylabel("current /nA")
        fig.savefig("pseudo_spectral_plot_{}mV_{}mV.pdf".format(first_V, second_V))
        print("plotted")

    return expected, std

def get_forward_problem(starting_parameters, index_of_parameter, coordinates, first_V, second_V):
    lst1 = starting_parameters[0:index_of_parameter]
    lst2 = starting_parameters[index_of_parameter+1:-1]
    return lambda x : simulate(lst1 + [x] + lst2, coordinates, first_V, second_V)

if __name__ == "__main__":
    # Put in plots folder
    dirname = os.path.join("output", "gpce_plots")
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    os.chdir(dirname)

    index_of_parameter = 5

    first_V  = -80
    second_V = 20
    coordinates = np.linspace(20, 500, 100)
    starting_parameters = [3.87068845e-04, 5.88028759e-02, 6.46971727e-05, 4.87408447e-02, 8.03073893e-02, 7.36295506e-03, 5.32908518e-03, 3.32254316e-02, 6.56614672e-02]

    distribution = chaospy.Normal(starting_parameters[index_of_parameter], starting_parameters[index_of_parameter])
    forward_problem = get_forward_problem(starting_parameters, index_of_parameter, coordinates, first_V, second_V)
    mean, std = do_gpce(forward_problem, starting_parameters, first_V, second_V, distribution, quad_order=50, poly_order=10, plot=True)

    forward_problem = get_forward_problem(starting_parameters, index_of_parameter, coordinates, second_V, first_V)
    mean, std = do_gpce(forward_problem, starting_parameters, second_V, first_V, distribution, quad_order=50, poly_order=10, plot=True)


    # # Compute the 'true' mean and std using many monte carlo samples
    # n_samples = 10000
    # sample = np.array(distribution.sample(n_samples))
    # sample = np.array([forward_problem(x) for x in sample])

    # base_mean = sample.mean()
    # base_std  = sample.std()

    # mean_error = []
    # std_error  = []
    # n_range = range(20, int(n_samples/10))
    # for n in n_range:
    #     mean_error.append(sample[0:n].mean())
    #     std_error.append(sample[0:n].std())
    # mean_error = np.array(mean_error) - base_mean
    # std_error  = np.array(std_error)  - base_std

    # plt.plot(n_range, np.log10(mean_error), label="mean_error")
    # plt.plot(n_range, np.log10(std_error),  label = "std_error")
    # plt.ylabel("log10")
    # plt.axhline(np.log10(mean-base_mean))
    # plt.axhline(np.log10(std - base_std))
    # plt.legend()
    # plt.savefig("monte_carlo_convergence.pdf")
    # plt.clf()

