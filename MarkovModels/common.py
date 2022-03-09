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
import uuid
from numba import njit
import subprocess
import sys
import datetime
import numpy.polynomial.polynomial as poly


def get_protocol_directory():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "protocols")


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
            "data_directory",
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
    parser.add_argument("-o", "--output", type=str, help="The directory to output figures and data to")
    return parser


def calculate_reversal_potential(T=293.15, K_in=120, K_out=5):
    """
    Compute the Nernst potential of a potassium channel.

    """
    # E is the Nernst potential for potassium ions across the membrane
    # Gas constant R, temperature T, Faradays constat F
    R = 8.31455
    F = 96485

    # valency of ions (1 in the case of K^+)
    z = 1

    # Nernst potential
    E = R * T / (z * F) * np.log(K_out / K_in)

    # Convert to mV
    return E * 1e3


def cov_ellipse(cov, offset=[0, 0], q=None,
                nsig=None, ax: plt.axes = None,
                label: str = None,
                rotate: float = None,
                resize_axes: bool = False,
                color: str = None):
    """
    copied from stackoverflow
    Parameters
    ----------


    cov : (2, 2) array
        Covariance matrix.
    q : array of floats, optional
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
        if label is None:
            label = "{}% confidence region".format(int(q * 100))
        else:
            label = label

        e = matplotlib.patches.Ellipse(offset,
                                       width[0],
                                       height[0],
                                       math.degrees(rotate),
                                       fill=False,
                                       label=label,
                                       color=color)
        ax.add_patch(e)
        e.set_clip_box(ax.bbox)

        window_width = max(np.abs(width[0] * math.cos(rotate)), np.abs(height[0] * math.sin(rotate)))
        window_height = max(np.abs(width[0] * math.sin(rotate)), np.abs(height[0] * math.cos(rotate)))
        # window_height = np.abs(height[0] * np.sin(rotate))
        # max_dim = max(window_width, window_height)

    # Change plot extents
    if resize_axes:
        if rotate != 0:
            ax.set_xlim(offset[0] - window_width, offset[0] + window_width)
            ax.set_ylim(offset[1] - window_height, offset[1] + window_height)
        else:
            ax.set_xlim(offset[0] - width / 2, offset[0] + width / 2)
            ax.set_ylim(offset[1] - height / 2, offset[1] + height / 2)

    return ax


def remove_spikes(times, voltages, spike_times, time_to_remove):
    lst = np.column_stack((times, voltages))
    indices_to_remove = []

    for spike in spike_times:
        spike_iter = [(i, t) for i, t in enumerate(lst[:, 0]) if t > spike]
        end_iter = [(i, t) for i, t in enumerate(lst[:, 0]) if t > spike + time_to_remove]

        if len(spike_iter) == 0 or len(end_iter) == 0:
            break

        spike_index = spike_iter[0][0]
        end_index = end_iter[0][0]

        if spike_index > len(lst) or end_index > len(lst):
            break
        indices_to_remove.append((spike_index - 1, end_index))

    indices_remaining = np.array(remove_indices(list(range(len(times))), indices_to_remove))
    return lst[indices_remaining, 0], lst[indices_remaining, 1], indices_remaining


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
        lsts.append(lst[indices_to_remove[i - 1][1] : indices_to_remove[i][0] + 1])

    lsts.append(lst[indices_to_remove[-1][1]:-1])

    lst = list(first_lst) + [index for lst in lsts for index in lst]
    return np.unique(lst)


def detect_spikes(x, y, threshold=100, window_size=0, earliest=True):
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

    if window_size > 0:
        spike_indices = [index - window_size + np.argmax(
            np.abs(y[(index - window_size):(index + window_size)])) for index in spike_indices]
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


def get_protocol_from_csv(protocol_name: str, directory=None, holding_potential=-80):
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

    protocol = pd.read_csv(os.path.join(directory, protocol_name + ".csv"))

    times = protocol["time"].values.flatten()
    voltages = protocol["voltage"].values.flatten()

    def staircase_protocol_safe(t):
        return np.interp([t], times, voltages)[0] if t < times[-1] and t > times[0] else holding_potential

    return staircase_protocol_safe, times[0], times[-1], times[1] - times[0]


def get_ramp_protocol_from_csv(protocol_name: str, directory=None, holding_potential=-80, threshold=0.001):
    """Generate a function by interpolating
    time-series data.

    Params:
    Holding potential: the value to return for times outside of the
    range

    :returns: a Tuple containg float->float which returns the voltage (in mV) at any given time t (in ms), tstart, tend, tstep and a Tuple describing the protocol.

    """

    if directory is None:
        directory = get_protocol_directory()

    protocol = pd.read_csv(os.path.join(directory, protocol_name + ".csv"))

    times = protocol["time"].values.flatten()
    voltages = protocol["voltage"].values.flatten()

    # Find gradient changes
    diff2 = np.abs(np.diff(voltages, n=2))
    # diff1 = np.abs(np.diff(voltages, n=1))

    windows = np.argwhere(diff2 > threshold).flatten()
    window_locs = np.unique(windows)
    window_locs = np.array([val for val in window_locs if val + 1 not in window_locs]) + 1

    windows = zip([0] + list(window_locs), list(window_locs) + [len(voltages) - 1])

    lst = []
    t_diff = times[1] - times[0]
    for start, end in windows:
        start_t = start * t_diff
        end_t = end * t_diff
        lst.append((start_t, end_t, voltages[start + 1], voltages[end - 1]))

    lst.append((end_t, np.inf, voltages[-1], voltages[-1]))

    protocol = tuple(lst)

    @njit
    def protocol_func(t):
        if t < 0 or t >= protocol[-1][1]:
            return holding_potential

        for i in range(len(protocol)):
            if t < protocol[i][1]:
                if np.abs(protocol[i][3] - protocol[i][2]) > threshold:
                    return protocol[i][2] + (t - protocol[i][0]) * (protocol[i][3] - protocol[i][2]) / (protocol[i][1] - protocol[i][0])
                else:
                    return protocol[i][3]

        raise Exception()
    return protocol_func, times[0], times[-1], times[1] - times[0], protocol


def fit_model(mm, data, starting_parameters=None, fix_parameters=[],
              max_iterations=None, subset_indices=None, method=pints.CMAES,
              solver=None, log_transform=True):
    """
    Fit a MarkovModel to some dataset using pints.

    Params:

    mm: A MarkovModel

    data: The data set to fit to: a (1,n) numpy array
    consiting of observations corresponding to the times in mm.times.

    starting_parameters: An initial guess for the optimal parameters

    fix_parameters: Which parameters (if any) should be ignored and set to fixed values

    max_iterations: An optional upper bound on the number of evaluations PINTS should perform

    method: Which optimisation method should be used

    returns: A pair containing the optimal parameters and the corresponding sum of square errors.

    """

    if log_transform:
        transformation = pints.LogTransformation(mm.get_no_parameters())
    else:
        transformation = None

    if solver is None:
        solver = mm.make_forward_solver_current()

    if subset_indices is None:
        subset_indices = np.array(list(range(len(mm.times))))

    if starting_parameters is None:
        starting_parameters = mm.get_default_parameters()

    class Boundaries(pints.Boundaries):
        def __init__(self, parameters, fix_parameters=None):
            self.fix_parameters = fix_parameters
            self.parameters = parameters

        def check(self, parameters):
            '''Check that each rate constant lies in the range 1.67E-5 < A*exp(B*V) < 1E3
            '''
            return True

        def n_parameters(self):
            return mm.get_no_parameters() - \
                len(self.fix_parameters) if self.fix_parameters is not None\
                else mm.get_no_parameters()

    class PintsWrapper(pints.ForwardModelS1):
        def __init__(self, mm, parameters, fix_parameters=None):
            self.mm = mm
            self.parameters = np.array(parameters)

            self.fix_parameters = fix_parameters

            unfixed_parameters = tuple([i for i in range(len(parameters)) if i not in fix_parameters])
            if fix_parameters is None:
                fix_parameters = tuple()

            if len(fix_parameters) > 0:
                def simulate(p, times):
                    sim_parameters = np.copy(parameters)
                    for i, j in enumerate(unfixed_parameters):
                        sim_parameters[j] = p[i]
                    sol = solver(sim_parameters)[subset_indices]
                    return sol
            else:
                def simulate(p, times):
                    try:
                        return solver(p)[subset_indices]
                    except Exception:
                        return np.full(times.shape, np.inf)

            self.simulate = simulate

        def n_parameters(self):
            return len(self.parameters) - len(self.fix_parameters)

        def simulateS1(self, parameters, times):
            raise NotImplementedError()

    model = PintsWrapper(mm, starting_parameters,
                         fix_parameters=fix_parameters)

    problem = pints.SingleOutputProblem(model, mm.times[subset_indices], data[subset_indices])
    error = pints.SumOfSquaresError(problem)
    boundaries = Boundaries(starting_parameters, fix_parameters)

    if fix_parameters is not None:
        params_not_fixed = [starting_parameters[i] for i in range(
            len(starting_parameters)) if i not in fix_parameters]
    else:
        params_not_fixed = starting_parameters

    controller = pints.OptimisationController(
        error, params_not_fixed, boundaries=boundaries, method=method, transformation=transformation)
    if max_iterations is not None:
        print("Setting max iterations = {}".format(max_iterations))
        controller.set_max_iterations(max_iterations)

    found_parameters, found_value = controller.run()

    # Now run with Nelder-Mead
    controller = pints.OptimisationController(
        error, params_not_fixed, boundaries=boundaries, method=pints.NelderMead,
        transformation=transformation)

    if max_iterations:
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
        t_end = 15000
        t_step = 0.1
    else:
        # Check the protocol folders for a protocol with the same name
        protocol_dir = get_protocol_directory()
        possible_protocol_path = os.path.join(protocol_dir, protocol_name + ".csv")
        if os.path.exists(possible_protocol_path):
            try:
                v, t_start, t_end, t_step, _ = get_ramp_protocol_from_csv(protocol_name)
            except:
                # TODO
                raise
        else:
            # Protocol not found
            raise Exception("Protocol not found at " + possible_protocol_path)
    return v, t_start, t_end, t_step

def get_data(well, protocol, data_directory):
    # Find data
    regex = re.compile(f"^newtonrun4-{protocol}-{well}.csv$")
    fname = next(filter(regex.match, os.listdir(data_directory)))
    data = pd.read_csv(os.path.join(data_directory, fname))['current'].values
    return data


def fit_well_data(model_class, well, protocol, data_directory, max_iterations,
                  output_dir=None, T=298, K_in=120, K_out=5,
                  default_parameters: float = None, removal_duration=5,
                  repeats=1, infer_E_rev=False, fit_initial_conductance=True):

    # Ignore files that have been commented out
    voltage_func, t_start, t_end, t_step, protocol_desc = get_ramp_protocol_from_csv(protocol)

    data = get_data(well, protocol, data_directory)

    times = pd.read_csv(os.path.join(data_directory, f"newtonrun4-{protocol}-times.csv"))['time'].values

    if infer_E_rev:
        Erev = infer_reversal_potential(protocol, data, times)
    else:
        Erev = calculate_reversal_potential(T=T, K_in=K_in, K_out=K_out)

    model = model_class(voltage_func, times, parameters=default_parameters, Erev=Erev)
    model.protocol_description = protocol_desc

    # Try fitting G_Kr on its own first
    initial_gkr = np.max(data) / 10

    initial_params = model.get_default_parameters()
    initial_params[model.GKr_index] = initial_gkr

    initial_score = ((model.SimulateForwardModel() - data)**2).sum()
    print(f"initial score is {initial_score}")

    # First fit only Gkr
    if fit_initial_conductance:
        def gkr_opt_func(gkr):
            p = model.get_default_parameters()
            p[8] = gkr
            return ((model.SimulateForwardModel(p) - data)**2).sum()

        if initial_gkr <= 0:
            initial_gkr = 1

            initial_params[8] = initial_gkr

    print(f"initial_gkr is {initial_gkr}")

    fitted_params_list = []
    scores = []

    columns = model.parameter_labels

    if infer_E_rev:
        columns.append("E_rev")

    dfs = []
    for i in range(repeats):
        fitted_params, scores = fit_model(model, data, starting_parameters=initial_params,
                                          max_iterations=max_iterations)

        fig = plt.figure(figsize=(14, 12))
        ax = fig.subplots(1)
        ax.plot(times, data)
        ax.plot(times, model.SimulateForwardModel(fitted_params))

        if infer_E_rev:
            fitted_params = np.append(fitted_params, Erev)

        dfs.append(pd.DataFrame(np.column_stack((*fitted_params.T, scores.T)), columns=columns + ['score']))

    df = pd.concat(dfs, ignore_index=True)

    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

            df.to_csv(os.path.join(output_dir, f"{well}_{protocol}_fitted_params_{i}.csv"))
            fig.savefig(os.path.join(output_dir, f"{well}_{protocol}_fit_{i}"))
            ax.cla()
    else:
        plt.show()

    return df


def get_all_wells_in_directory(data_dir, regex="^newtonrun4-([a-z|A-Z|0-9]*)-([A-Z][0-9][0-9]).csv$", group=1):
    regex = re.compile(regex)
    wells = []

    for f in filter(regex.match, os.listdir(data_dir)):
        well = re.search(regex, f).groups()[group]
        wells.append(well)

    return wells


def infer_reversal_potential(protocol: str, current: np.array, times, ax=None, output_path=None, plot=False):

    orig_times = times
    orig_current = current

    if ax or output_path:
        plot = True

    protocol_func, _, _, _, protocol_desc = get_ramp_protocol_from_csv(protocol)

    tstart = times[0]
    tstep = times[1] - times[0]

    # First, find the reversal ramp. Search backwards along the protocol until we find a >= 40mV step
    step = next(filter(lambda x: x[2] >= -74, reversed(protocol_desc)))

    if step[1] - step[0] > 200 or step[1] - step[0] < 50:
        raise Exception("Failed to find reversal ramp in protocol")

    # Next extract steps
    istart = int((step[0] + tstart) / tstep)
    iend = int((step[1] + tstart) / tstep)

    times = times[istart:iend]
    current = current[istart:iend]

    voltages = np.array([protocol_func(t) for t in times])

    fitted_poly = poly.Polynomial.fit(voltages, current, 4)

    if plot:
        if ax is None:
            fig = plt.figure()
            ax = fig.subplots()

        ax.plot(*fitted_poly.linspace())
        ax.set_xlabel('voltage mV')
        ax.set_ylabel('current nA')
        # Now plot current vs voltage
        plt.plot(voltages, current, 'x', markersize=2, color='grey')

        if output_path is not None:
            fig = ax.figure
            fig.savefig(output_path)

    roots = np.unique([np.real(root) for root in fitted_poly.roots()
                       if root > np.min(voltages) and root < np.max(voltages)])

    # It makes sense to take the last root. This should be the first time that
    # the current crosses 0 and where the ion-channel kinetics are too slow to
    # play a role
    if len(roots) == 0:
        return np.nan

    deriv = fitted_poly.deriv()(roots[-1])

    if deriv < 0:
        return np.nan

    return roots[-1]


def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def setup_output_directory(dirname: str = None, subdir_name: str = None):

    if dirname is None:
        if subdir_name:
            dirname = os.path.join("output", f"{subdir_name}-{uuid.uuid4()}")
        else:
            dirname = os.path.join("output", f"output-{uuid.uuid4()}")

    if subdir_name is not None:
        dirname = os.path.join(dirname, subdir_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(os.path.join(dirname, 'info.txt'), 'w') as description_fout:
        git_hash = get_git_revision_hash()
        datetimestr = str(datetime.datetime.now())
        description_fout.write(f"Date: {datetimestr}\n")
        description_fout.write(f"Commit {git_hash}\n")
        command = " ".join(sys.argv)
        description_fout.write(f"Command: {command}\n")

    return dirname
