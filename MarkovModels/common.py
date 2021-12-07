#!/usr/bin/env python3

import argparse
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import numpy as np
import pandas as pd
import math
import os
import pints
import regex as re

def get_protocol_directory():
    return os.path.join(os.path.dirname( os.path.realpath(__file__)), "protocols")

def get_protocol_list():
    directory = get_protocol_directory()

    files = os.listdir(directory)
    regex = re.compile("^([a-z|0-9]*)\.csv$")

    protocols = []
    for fname in filter(regex.match, files):
        name = re.search(regex, fname).group(1)
        protocols.append(name)
    return protocols

def get_args(data_reqd=False, description=None):
    """Get command line arguments from using get_parser


    Params:

    data_reqd: is a flag which is set to True when a positional argument
    giving a path to some input data is needed

    description: is a string describing what the program should be used for. If
    this is none, get_parser uses a default description

    """
    # Check input arguments
    parser = get_parser(data_reqd, description=description)
    args = parser.parse_args()

    # Create output directory
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    return args


def get_parser(data_reqd=False, description=None):
    """Create an ArgumentParser object with common command line arguments already included

    Params:

    data_reqd: a flag which is set to True when a positional argument
    giving a path to some input data is needed

    description: is a string describing what the program should be used for. If
    this is none, get_parser uses a default description

    """

    if description is not None:
        description = 'Plot sensitivities of a Markov model'

    # Check input arguments
    parser = argparse.ArgumentParser(description=description)
    if data_reqd:
        parser.add_argument(
            "data_file_path",
            help="path to csv data for the model to be fit to",
            default=False)
    parser.add_argument(
        "-p",
        "--plot",
        action='store_true',
        help="whether to plot figures or just save",
        default=False)
    parser.add_argument("--dpi", type=int, default=100,
                        help="what DPI to use for figures")
    parser.add_argument("-o", "--output", type=str, default="output",
                        help="The directory to output figures and data to")
    return parser


def calculate_reversal_potential(T=293.15, K_out=120, K_in=5):
    """
    Compute the Nernst potential of a potassium channel.

    """
    # E is the Nernst potential for potassium ions across the membrane
    # Gas constant R, temperature T, Faradays constat F
    R = 8314.55
    F = 96485

    # valency of ions (1 in the case of K^+)
    z = 1

    # Nernst potential
    E = R * T / (z * F) * np.log(K_in / K_out)
    return E


def cov_ellipse(cov, offset=[0, 0], q=None,
                nsig=None, ax: plt.axes = None,
                label_arg: str = None,
                rotate: float = None,
                resize_axes: bool=False,
                color: str = None):
    """
    copied from stackoverflow
    Parameters
    ----------


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

    if q is not None:
        q = np.asarray(q)
        qs = np.sort(q)
    elif nsig is not None:
        q = 2 * scipy.stats.norm.cdf(nsig) - 1
        qs = [q]
    else:
        raise ValueError('One of `q` and `nsig` should be specified.')

    if ax is None:
        ax = plt.gca()

    if color is None:
        color = np.random.rand(3)

    val, vec = np.linalg.eigh(cov)
    for q in qs:
        r2 = scipy.stats.chi2.ppf(q, 2)
        width, height = 2 * np.sqrt(val[:, None] * r2)
        rotate = np.arctan2(*vec[::-1, 0]) if rotate is None else rotate
        if label_arg is None:
            label = "{}% confidence region".format(int(q * 100))
        else:
            label = label_arg

        e = matplotlib.patches.Ellipse(offset,
                                       width[0],
                                       height[0],
                                       math.degrees(rotate),
                                       fill=False,
                                       label=label,
                                       color=color)
        ax.add_patch(e)
        e.set_clip_box(ax.bbox)

        window_width  = max(np.abs(width[0] * math.cos(rotate)), np.abs(height[0] * math.sin(rotate)))
        window_height = max(np.abs(width[0] * math.sin(rotate)), np.abs(height[0] * math.cos(rotate)))
        # window_height = np.abs(height[0] * np.sin(rotate))
        # max_dim = max(window_width, window_height)

    # Change plot extents
    if resize_axes:
        if rotate != 0:
            ax.set_xlim(offset[0] - window_width, offset[0] + window_width)
            ax.set_ylim(offset[1] - window_height, offset[1] + window_height)
        else:
            ax.set_xlim(offset[0] - width/2, offset[0] + width/2)
            ax.set_ylim(offset[1] - height/2, offset[1] + height/2)

    return ax

def remove_spikes(times, voltages, spike_times, time_to_remove):
    lst = np.column_stack((times, voltages))
    indices_to_remove = []
    for spike in spike_times:

        spike_iter = list(filter(lambda v: v[1] > spike, enumerate(lst[:, 0])))
        end_iter = list(filter(lambda v: v[1] > spike + time_to_remove, enumerate(lst[:, 0])))

        if len(spike_iter) == 0 or len(end_iter) == 0:
            break

        spike_index = spike_iter[0][0]
        end_index = end_iter[0][0]

        if spike_index > len(lst) or end_index > len(lst):
            break
        indices_to_remove.append((spike_index, end_index))

    lst = np.array(remove_indices(np.column_stack((times, voltages)),
                                  indices_to_remove))
    return lst[:,0], lst[:,1]

def remove_indices(lst, indices_to_remove):
    """Remove a list of indices from some list-like object

    Params:

    lst: a list-like object

    indices_to_remove: A list of pairs of indices (a,b) with a<b such that we
    remove indices strictly between a and b


    returns a new list

    """
    if len(indices_to_remove) == 0:
        return lst
    if indices_to_remove is None:
        return lst

    first_lst = lst[0:indices_to_remove[0][0]]

    lsts = []
    for i in range(1, len(indices_to_remove)):
        lsts.append(lst[indices_to_remove[i - 1][1]: indices_to_remove[i][0] + 1])

    lsts.append(lst[indices_to_remove[-1][1]:-1])

    lst = list(first_lst) + [index for lst in lsts for index in lst]
    return lst


def detect_spikes(x, y, threshold=100, window_size=250):
    """
    Find the points where time-series 'jumps' suddenly. This is useful in order
    to find 'capacitive spikes' in protocols.

    Params:
    x : the independent variable (usually time)
    y : the dependent variable (usually voltage)

    """
    dx = np.diff(x)
    dy = np.diff(y)

    deriv = dy / dx
    spike_indices = np.argwhere(np.abs(deriv) > threshold)[:, 0]

    spike_indices = [index - window_size + np.argmax(
        np.abs(y[(index - window_size):(index + window_size)]))
                     for index in spike_indices]
    spike_indices = np.unique(spike_indices)

    if(len(spike_indices) == 0):
        return [], np.array([])

    return x[spike_indices], np.array(spike_indices)


def beattie_sine_wave(t):
    """
    The sine-wave protocol from https://doi.org/10.1113/JP276068.

    Params:

    t: The time at which to compute the voltage


    Returns:
    V: the voltage at time t
    """

    # This shift is needed for simulated protocol to match the protocol
    # recorded in experiment, which is shifted by 0.1ms compared to the
    # original input protocol. Consequently, each step is held for 0.1ms
    # longer in this version of the protocol as compared to the input.
    shift = 0.1
    C = [54.0,
         26.0,
         10.0,
         0.007 / (2 * np.pi),
         0.037 / (2 * np.pi),
         0.19 / (2 * np.pi)]

    if t >= 250 + shift and t < 300 + shift:
        V = -120
    elif t >= 500 + shift and t < 1500 + shift:
        V = 40
    elif t >= 1500 + shift and t < 2000 + shift:
        V = -120
    elif t >= 3000 + shift and t < 6500 + shift:
        V = -30 + C[0] * (np.sin(2 * np.pi * C[3] * (t - 2500 - shift))) + C[1] * \
            (np.sin(2 * np.pi * C[4] * (t - 2500 - shift))) + \
            C[2] * (np.sin(2 * np.pi * C[5] * (t - 2500 - shift)))
    elif t >= 6500 + shift and t < 7000 + shift:
        V = -120
    else:
        V = -80
    return V


def get_protocol_from_csv(protocol_name : str, directory=None, holding_potential=-80):
    """Generate a function by interpolating
    time-series data.

    Params:
    Holding potential: the value to return for times outside of the
    range

    Returns:
    Returns a function float->float which returns the voltage (in mV)
    at any given time t (in ms)

    """

    if directory is None:
        directory = get_protocol_directory()

    protocol = pd.read_csv(os.path.join(directory, protocol_name+".csv"))

    times = protocol["time"].values.flatten()
    voltages = protocol["voltage"].values.flatten()

    staircase_protocol = scipy.interpolate.interp1d(
        times, voltages, kind="linear")

    def staircase_protocol_safe(t): return staircase_protocol(
            t) if t < times[-1] and t > times[0] else holding_potential

    return staircase_protocol_safe, times[0], times[-1], times[1]-times[0]

def draw_cov_ellipses(mean=[0, 0], S1=None, sigma2=None, cov=None, plot_dir=None):
    """Plot confidence intervals using a sensitivity matrix or covariance matrix.

    In the case of a sensitivity matrix, i.i.d Guassian additive errors are
    assumed with variance sigma2. Exactly one of cov and S1 must not be None.
    In the case that S1 is provided, the confidence regions are calculated
    under the assumption that all other variables are fixed. However, if cov is
    provided, the confidence regions correspond a marginal distribution.

    Params:

    S1: A sensitivity matrix where S_i,j corresponds to the derivative
    of the (scalar) observation at the ith timepoint with respect to the jth
    parameter.

    sigma2: The variance of the Gaussian additive noise - only required if S1
    is provided

    cov: A covariance matrix

    plot_dir: The directory to store the plots in. When this defaults to None, the
    plots will be displayed using plt.show()

    """

    # TODO improve exception handling
    if S1 is not None:
        if cov is not None:
            Raise()
        else:
            n_params = S1.shape[1]
    else:
        if cov is None:
            raise ValueError()
        if sigma2 is not None:
            raise ValueError()
        else:
            n_params = cov.shape[0]

    for j in range(0, n_params - 1):
        for i in range(j + 1, n_params):
            if S1 is not None:
                if sigma2 is None:
                    raise
                sub_sens = S1[:, [i, j]]
                sub_cov = sigma2 * np.linalg.inv(np.dot(sub_sens.T, sub_sens))
            # Else use cov
            else:
                sub_cov = cov[parameters_to_view[:, None], np.array((i,j))]
            eigen_val, eigen_vec = np.linalg.eigh(sub_cov)
            eigen_val = eigen_val.real
            if eigen_val[0] > 0 and eigen_val[1] > 0:
                # Parameters have been normalised to 1
                cov_ellipse(sub_cov, offset=mean, q=[0.5, 0.95],
                            resize_axes=True)
                plt.ylabel("parameter {}".format(i + 1))
                plt.xlabel("parameter {}".format(j + 1))
                plt.legend()
                if plot_dir is None:
                    plt.show()
                else:
                    plt.savefig(
                        os.path.join(
                            plot_dir,
                            "covariance_for_parameters_{}_{}".format(
                                j + 1,
                                i + 1)))
                plt.clf()
            else:
                print(
                    "COV_{},{} : negative eigenvalue: {}".format(
                        i, j, eigen_val))


def fit_model(funcs, data, starting_parameters=None, fix_parameters=[],
              max_iterations=None, method=pints.CMAES):
    """
    Fit a MarkovModel to some dataset using pints.

    Params:

    funcs: A MarkovModel

    data: The data set to fit to: a (1,n) numpy array
    consiting of observations corresponding to the times in funcs.times.

    starting_parameters: An initial guess for the optimal parameters

    fix_parameters: Which parameters (if any) should be ignored and set to fixed values

    max_iterations: An optional upper bound on the number of evaluations PINTS should perform

    method: Which optimisation method should be used

    returns: A pair containing the optimal parameters and the corresponding sum of square errors.

    """
    if starting_parameters is None:
        starting_parameters = funcs.get_default_parameters()

    class Boundaries(pints.Boundaries):
        def __init__(self, parameters, fix_parameters=None):
            self.fix_parameters = fix_parameters
            self.parameters = parameters

        def check(self, parameters):
            '''Check that each rate constant lies in the range 1.67E-5 < A*exp(B*V) < 1E3
            '''
            sim_params = self.parameters

            if np.any(sim_params) <= 0:
                return False

            c = 0
            for i in fix_parameters:
                if i not in self.fix_parameters:
                    sim_params[i] = parameters[c]
                    c += 1
                if c == len(parameters):
                    break

            # TODO Rewrite this for other models
            for i in range(0, 4):
                alpha = sim_params[2 * i]
                beta = sim_params[2 * i + 1]

                vals = [0, 0]
                vals[0] = alpha * np.exp(beta * -90 * 1E-3)
                vals[1] = alpha * np.exp(beta * 50 * 1E-3)
                for val in vals:
                    if val < 1E-7 or val > 1E3:
                        return False

            return True

        def n_parameters(self):
            return 9 - \
                len(self.fix_parameters) if self.fix_parameters is not None else 9

    class PintsWrapper(pints.ForwardModelS1):
        def __init__(self, funcs, parameters, fix_parameters=None):
            self.funcs = funcs
            self.parameters = parameters

            self.fix_parameters = fix_parameters

            if fix_parameters is not None:
                self.free_parameters = [i for i in range(
                    0, len(starting_parameters)) if i not in fix_parameters]
            else:
                self.free_parameters = range(0, len(starting_parameters))

        def n_parameters(self):
                return len(self.parameters) - len(self.fix_parameters)

        def simulate(self, parameters, times):
            self.funcs.times = times
            if self.fix_parameters is None:
                return self.funcs.SimulateForwardModel(parameters, times)
            else:
                sim_params = np.copy(self.parameters)
                c = 0
                for i, parameter in enumerate(self.parameters):
                    if i not in self.fix_parameters:
                        sim_params[i] = parameters[c]
                        c += 1
                    if c == len(parameters):
                        break
                return self.funcs.SimulateForwardModel(sim_params, times)

        def simulateS1(self, parameters, times):
            if fix_parameters is None:
                return self.funcs.SimulateForwardModelSensitivities(parameters)
            else:
                sim_params = np.copy(self.parameters)
                c = 0
                for i, parameter in enumerate(self.parameters):
                    if i not in fix_parameters:
                        sim_params[i] = parameters[c]
                        c += 1
                    if c == len(parameters):
                        break
                current, sens = self.funcs.SimulateForwardModelSensitivities(
                    sim_params, times)
                sens = sens[:, self.free_parameters]
                return current, sens

    model = PintsWrapper(funcs, starting_parameters,
                         fix_parameters=fix_parameters)
    problem = pints.SingleOutputProblem(model, funcs.times, data)
    error = pints.SumOfSquaresError(problem)
    boundaries = Boundaries(starting_parameters, fix_parameters)

    print("data size is {}".format(data.shape))

    if fix_parameters is not None:
        params_not_fixed = [starting_parameters[i] for i in range(
            len(starting_parameters)) if i not in fix_parameters]
    else:
        params_not_fixed = starting_parameters
    controller = pints.OptimisationController(
        error, params_not_fixed, boundaries=boundaries, method=method)
    if max_iterations is not None:
        print("Setting max iterations = {}".format(max_iterations))
        controller.set_max_iterations(max_iterations)

    found_parameters, found_value = controller.run()
    return found_parameters, found_value


def get_protocol(protocol_name: str):
    """Returns a function describing the voltage trace.

    params:

    protocol_name: A string used to select the protocol

    returns: A function, v(t), which returns the voltage (mV)
    at a time t (ms).

    """
    v = None
    if protocol_name == "sine-wave":
        v = beattie_sine_wave
        t_start = 0
        t_end   = 15000
        t_step  = 0.1
    else:
        # Check the protocol folders for a protocol with the same name
        protocol_dir = get_protocol_directory()
        possible_protocol_path = os.path.join(protocol_dir, protocol_name+".csv")
        if os.path.exists(possible_protocol_path):
            try:
                v, t_start, t_end, t_step = get_protocol_from_csv(protocol_name)
            except:
                # TODO
                raise
        else:
            # Protocol not found
            raise Exception("Protocol not found at " + possible_protocol_path)
    return v, t_start, t_end, t_step

def fit_well_to_data(model_class, well, protocol, data_directory, max_iterations, output_dir = None, T=298, K_in=120, K_out=5, default_parameters: float = None):

    # Ignore files that have been commented out
    voltage_func, t_start, t_end, t_step = get_protocol(protocol)
    # Find data
    regex = re.compile(f"^newtonrun4-{protocol}-{well}.csv$")
    fname = next(filter(regex.match, os.listdir(data_directory)))
    data = pd.read_csv(os.path.join(data_directory, fname))['current'].values

    times = pd.read_csv(os.path.join(data_directory, f"newtonrun4-{protocol}-times.csv"))['time'].values
    print(times.shape)

    Erev = calculate_reversal_potential(T=T, K_in=K_in, K_out=K_out)

    model = model_class(voltage_func, times)

    if default_parameters is not None:
        model.default_parameters = default_parameters

    model.Erev = Erev

    initial_gkr_guess = np.max(data)/1000

    initial_params = model.get_default_parameters()
    initial_params[model.GKr_index] = initial_gkr_guess

    # First fit only Gkr

    def gkr_opt_func(gkr):
        p = model.get_default_parameters()
        p[8] = gkr
        return ((model.SimulateForwardModel() - data)**2).sum()

    initial_gkr = scipy.optimize.minimize_scalar(gkr_opt_func).x
    initial_params[8] = initial_gkr

    # plot initial guess
    # plt.plot(times, model.SimulateForwardModel(initial_params))
    # plt.plot(times, data)
    # plt.show()

    fitted_params, score = fit_model(model, data, starting_parameters=initial_params, max_iterations=max_iterations)

    fig = plt.figure(figsize=(14,12))
    fig.plot(times, data)
    fig.plot(times, model.SimulateForwardModel(fitted_params))

    if output_dir is not None:
           df = pd.DataFrame(np.column_stack((fitted_params[None,:], [score])), columns=model.parameter_labels + ['SSE'])
           df.to_csv(os.path.join(output_dir, f"{well}_{protocol}_fitted_params.csv"))
           fig.savefig(f"{well}_{protocol}_fit")

    return fitted_params
