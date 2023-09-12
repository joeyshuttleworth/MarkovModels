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
import numba
import pints.plot
import regex as re
import uuid
from numba import njit
import subprocess
import sys
import datetime
import logging
import time
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


def calculate_reversal_potential(T=293, K_in=120, K_out=5):
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

    spike_indices = np.array([np.argmax(times > spike_time) for spike_time in
                              spike_times])
    intervals_to_remove = [(spike, spike + int(np.argmax(times[spike: ] > times[spike]\
                                                 + time_to_remove))) for spike in spike_indices]

    intervals_to_remove = np.vstack(intervals_to_remove)

    indices = remove_indices(list(range(len(times))), intervals_to_remove)
    return times[indices], voltages[indices], indices


def remove_indices(lst, indices_to_remove):
    """Remove a list of indices from some list-like object

    Params:

    lst: a list-like object

    indices_to_remove: A list of pairs of indices (a,b) with a<b such that we
    remove indices strictly between a and b


    returns a new list

    """
    indices_to_remove = np.vstack(indices_to_remove)

    # Ensure intervals don't overlap
    for interval1, interval2 in zip(indices_to_remove[:-1, :], indices_to_remove[1:, :]):
        if interval1[1] > interval2[0]:
            interval1[1] = interval2[1]
            interval2[0] = -1
            interval2[1] = -1

    indices_to_remove = [v for v in indices_to_remove if v[0] >= 0 and v[1] >= 0]

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
        spike_indices = [index - window_size + np.argmax( np.abs(y[(index -
                                                                    window_size):(index + window_size)])) for index in spike_indices]
        spike_indices = np.unique(spike_indices)

    if(len(spike_indices) == 0):
        return [], np.array([])

    spike_indices -= 1

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

    protocol = pd.read_csv(os.path.join(directory, protocol_name + ".csv"),
                           float_precision='round_trip')

    times = protocol["time"].values.flatten()
    voltages = protocol["voltage"].values.flatten()

    def protocol_safe(t):
        return np.interp([t], times, voltages)[0] if t < times[-1] and t > times[0] else holding_potential

    return protocol_safe, times


def get_ramp_protocol_from_csv(protocol_name: str, directory=None,
                               holding_potential=-80.0, threshold=0.001):
    """Generate a function by interpolating
    time-series data.

    Params:
    Holding potential: the value to return for times outside of the
    range

    :returns: a Tuple containg float->float which returns the voltage (in mV) at any given time t (in ms), tstart, tend, tstep and a Tuple describing the protocol.

    """

    if directory is None:
        directory = get_protocol_directory()

    protocol = pd.read_csv(os.path.join(directory, protocol_name + ".csv"),
                           float_precision='round_trip')

    times = protocol["time"].values.flatten().astype(np.float64)
    voltages = protocol["voltage"].values.flatten()

    # Find gradient changes
    diff2 = np.abs(np.diff(voltages, n=2))
    # diff1 = np.abs(np.diff(voltages, n=1))

    windows = np.argwhere(diff2 > threshold).flatten()
    window_locs = np.unique(windows)
    window_locs = np.array([val for val in window_locs if val + 1 not in window_locs]) + 1

    windows = zip([0] + list(window_locs), list(window_locs) + [len(voltages) - 1])

    lst = []
    for start, end in windows:
        start_t = times[start]
        end_t = times[end-1]

        v_start = voltages[start]
        v_end = voltages[end-1]

        lst.append((start_t, end_t, v_start, v_end))

    lst.append((end_t, np.inf, holding_potential, holding_potential))

    protocol = tuple(lst)

    @njit
    def protocol_func(t: np.float64, offset=0.0):
        t = t + offset
        if t <= 0 or t >= protocol[-1][1]:
            return holding_potential

        for i in range(len(protocol)):
            if t < protocol[i][1]:
                # ramp_start = protocol[i][0]
                if protocol[i][3] - protocol[i][2] != 0:
                    return protocol[i][2] + (t - protocol[i][0]) * (protocol[i][3] - protocol[i][2]) / (protocol[i][1] - protocol[i][0])
                else:
                    return protocol[i][3]

        raise Exception()

    return protocol_func, times, protocol


def fit_model(mm, data, times=None, starting_parameters=None,
              fix_parameters=[], max_iterations=None, subset_indices=None,
              method=pints.CMAES, solver=None, log_transform=True, repeats=1,
              return_fitting_df=False, parallel=False,
              randomise_initial_guess=True, output_dir=None, solver_type=None,
              no_conductance_boundary=False,
              rng=None):
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
    if not times:
        times = mm.times

    if rng is None:
        rng = np.random.default_rng()

    if log_transform:
        # Assume that the conductance is the last parameter and that the parameters are arranged included

        if mm.transformations:
            transformations = [t for i, t in enumerate(mm.transformations) if i not in fix_parameters]
            transformation = pints.ComposedTransformation(*mm.transformations)

        else:
            # Use a-space transformation (Four Ways to Fit...)
            no_rates = int((mm.get_no_parameters() - 1)/2)
            log_transformations = [pints.LogTransformation(1) for i in range(no_rates)]
            identity_transformations = [pints.IdentityTransformation(1) for i in range(no_rates)]

            # Flatten and include conductance on the end
            transformations = [w for u, v
                               in zip(log_transformations, identity_transformations)
                               for w in (u, v)]\
                                   + [pints.IdentityTransformation(1)]

            transformations = [t for i, t in enumerate(transformations) if i not in fix_parameters]
            transformation = pints.ComposedTransformation(*transformations)

    else:
        transformation = None

    if starting_parameters is None:
        starting_parameters = mm.get_default_parameters()

    if max_iterations == 0:
        return starting_parameters, np.inf

    if solver is None:
        try:
            solver = mm.make_forward_solver_of_type(solver_type)
        except numba.core.errors.TypingError as exc:
            logging.warning(f"unable to make nopython forward solver {str(exc)}")
            solver = mm.make_forward_solver_of_type(solver_type, njitted=False)

    if subset_indices is None:
        subset_indices = np.array(list(range(len(mm.times))))

    fix_parameters = np.unique(fix_parameters)

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

    problem = pints.SingleOutputProblem(model, times[subset_indices],
                                        data[subset_indices])

    error = pints.SumOfSquaresError(problem)

    if fix_parameters is not None:
        unfixed_indices = [i for i in range(
            len(starting_parameters)) if i not in fix_parameters]
        params_not_fixed = starting_parameters[unfixed_indices]
    else:
        unfixed_indices = list(range(len(starting_parameters)))
        params_not_fixed = starting_parameters

    boundaries = fitting_boundaries(starting_parameters, mm,
                                    fix_parameters)

    if randomise_initial_guess:
        initial_guess_dist = fitting_boundaries(starting_parameters, mm,
                                                fix_parameters)
        starting_parameter_sets = []

    scores, parameter_sets, iterations, times_taken = [], [], [], []
    for i in range(repeats):
        if randomise_initial_guess:
            initial_guess = initial_guess_dist.sample(n=1).flatten()
            starting_parameter_sets.append(initial_guess)
            params_not_fixed = initial_guess

        if not boundaries.check(params_not_fixed):
            raise ValueError(f"starting parameter lie outside boundary: {params_not_fixed}")

        controller = pints.OptimisationController(error, params_not_fixed,
                                                  boundaries=boundaries,
                                                  method=method,
                                                  transformation=transformation)
        if not parallel:
            controller.set_parallel(False)

        try:
            if max_iterations is not None:
                controller.set_max_iterations(max_iterations)

        except Exception as e:
            print(str(e))
            found_value = np.inf
            found_parameters = starting_parameters

        timer_start = time.process_time()
        found_parameters, found_value = controller.run()
        timer_end = time.process_time()
        time_elapsed = timer_end - timer_start

        this_run_iterations = controller.iterations()
        parameter_sets.append(found_parameters)
        scores.append(found_value)
        iterations.append(this_run_iterations)
        times_taken.append(time_elapsed)

    best_score = min(scores)
    best_index = scores.index(best_score)
    best_parameters = parameter_sets[best_index]

    if not np.all(np.isfinite(model.simulate(found_parameters, mm.times))):
        best_parameters = mm.get_default_parameters()
        best_score = np.inf

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        point2 = [p for i, p in enumerate(mm.get_default_parameters()) if i not
                  in fix_parameters]
        fig, axes = pints.plot.function_between_points(error,
                                                       point_1=best_parameters,
                                                       point_2=point2,
                                                       padding=0.1,
                                                       evaluations=100)

        fig.savefig(os.path.join(output_dir, 'best_fitting_profile_from_default'))
        plt.close(fig)

        if randomise_initial_guess:
            point_2 = starting_parameter_sets[best_index % len(starting_parameter_sets)]
            fig, axes = pints.plot.function_between_points(error,
                                                           point_1=best_parameters,
                                                           point_2=point_2,
                                                           padding=0.1,
                                                           evaluations=100)
            fig.savefig(os.path.join(output_dir, 'best_fitting_profile_from_initial_guess'))
            plt.close(fig)

    if fix_parameters:
        for i in np.unique(fix_parameters):
            best_parameters = np.insert(best_parameters,
                                        i,
                                        starting_parameters[i])
    if return_fitting_df:
        if fix_parameters:
            new_rows = parameter_sets
            for i in np.unique(fix_parameters):
                for j, row in enumerate(parameter_sets):
                    new_rows[j] = np.insert(row, i, starting_parameters[i])
            parameter_sets = np.array(new_rows)
        else:
            parameter_sets = np.vstack(parameter_sets)
        fitting_df = pd.DataFrame(parameter_sets,
                                  columns=mm.get_parameter_labels()[:parameter_sets.shape[1]])
        fitting_df['RMSE'] = scores
        fitting_df['iterations'] = iterations
        fitting_df['CPU_time'] = times_taken

        # Append starting parameters also
        if randomise_initial_guess:
            columns = np.array(mm.get_parameter_labels())
            initial_guess_df = pd.DataFrame(starting_parameter_sets,
                                            columns=columns[unfixed_indices])
            initial_guess_df['iterations'] = iterations
            initial_guess_df['RMSE'] = np.NaN

            fitting_df = pd.concat([fitting_df, initial_guess_df],
                                   ignore_index=True)

        return best_parameters, best_score, fitting_df
    else:
        return best_parameters, best_score


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
                v, times, _ = get_ramp_protocol_from_csv(protocol_name)
            except:
                # TODO
                raise
        else:
            # Protocol not found
            raise Exception("Protocol not found at " + possible_protocol_path)
    return v, t_start, t_end, t_step


def get_data(well, protocol, data_directory, experiment_name='newtonrun4', sweep=None):
    # Find data
    if sweep:
        regex = re.compile(f"^{experiment_name}-{protocol}-{well}-sweep{sweep}.csv$")
    else:
        regex = re.compile(f"^{experiment_name}-{protocol}-{well}.csv$")
    fname = next(filter(regex.match, os.listdir(data_directory)))
    data = pd.read_csv(os.path.join(data_directory, fname),
                       float_precision='round_trip')['current'].values
    return data


def fit_well_data(model_class_name : str, well, protocol, data_directory, max_iterations,
                  output_dir=None, T=None, K_in=None, K_out=None,
                  default_parameters: float = None, removal_duration=5,
                  repeats=1, infer_E_rev=False, fit_initial_conductance=True,
                  experiment_name='newtonrun4', solver=None, E_rev=None,
                  randomise_initial_guess=True, parallel=False,
                  solver_type=None, sweep=None, scale_conductance=True,
                  no_conductance_boundary=False):

    if default_parameters is None or len(default_parameters) == 0:
        default_parameters = make_model_of_class(model_class_name).get_default_parameters()

    parameter_labels = make_model_of_class(model_class_name).get_parameter_labels()

    if max_iterations == 0:
        df = pd.DataFrame(default_parameters[None, :], columns=parameter_labels)
        df['score'] = np.inf
        return df

    # Ignore files that have been commented out
    voltage_func, times, protocol_desc = get_ramp_protocol_from_csv(protocol)

    data = get_data(well, protocol, data_directory, experiment_name, sweep=sweep)

    times = pd.read_csv(os.path.join(data_directory, f"{experiment_name}-{protocol}-times.csv"),
                        float_precision='round_trip')['time'].values

    voltages = np.array([voltage_func(t) for t in times])

    spike_times, _ = detect_spikes(times, voltages, window_size=0)

    _, _, indices = remove_spikes(times, voltages, spike_times,
                                  removal_duration)

    if infer_E_rev:
        try:
            if output_dir:
                plot = True
                output_path = os.path.join(output_dir, 'infer_reversal_potential.png')
            inferred_E_rev = infer_reversal_potential(protocol, data, times, plot=plot,
                                                      output_path=output_path)
            if inferred_E_rev < -50 or inferred_E_rev > -100:
                E_rev = inferred_E_rev
        except Exception:
            pass

    model = make_model_of_class(model_class_name, voltage=voltage_func,
                                times=times,
                                default_parameters=default_parameters,
                                E_rev=E_rev,
                                protocol_description=protocol_desc)

    if default_parameters is not None:
        initial_params = default_parameters.copy()

    columns = model.get_parameter_labels()

    if infer_E_rev:
        columns.append("E_rev")

    if solver is not None and solver_type is not None:
        raise Exception('solver and solver type provided')

    if solver is None:
        try:
            solver = model.make_forward_solver_of_type(solver_type)
            solver()
        except numba.core.errors.TypingError as exc:
            logging.warning(f"unable to make nopython forward solver {str(exc)}")
            solver = model.make_forward_solver_of_type(solver_type, njitted=False)

    if not np.all(np.isfinite(solver())):
        raise Exception('Default parameters gave non-finite output')

    fitted_params, score, fitting_df = fit_model(model, data, solver=solver,
                                                 starting_parameters=initial_params,
                                                 max_iterations=max_iterations,
                                                 subset_indices=indices,
                                                 parallel=parallel,
                                                 randomise_initial_guess=randomise_initial_guess,
                                                 return_fitting_df=True,
                                                 repeats=repeats,
                                                 output_dir=output_dir,
                                                 solver_type=solver_type,
                                                 no_conductance_boundary=no_conductance_boundary)

    fig = plt.figure(figsize=(14, 12))
    ax = fig.subplots()
    for i, row in fitting_df.iterrows():
        fitted_params = row[model.get_parameter_labels()].values.flatten()
        try:
            ax.plot(times, solver(fitted_params), label='fitted parameters')
            ax.plot(times, solver(initial_params), label='default parameters')
            ax.plot(times, data, color='grey', label='data', alpha=.5)
        except Exception:
            pass

        ax.legend()

        if infer_E_rev:
            fitted_params = np.append(fitted_params, E_rev)

        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            fname = f"{well}_{protocol}_fit_{i}" if i < repeats else f"{well}_{protocol}_initial_guess_{i}"

            fig.savefig(os.path.join(output_dir, fname))
            ax.cla()
    plt.close(fig)

    fitting_df['score'] = fitting_df['RMSE']
    fitting_df.to_csv(os.path.join(output_dir, f"{well}_{protocol}_fitted_params.csv"))
    return fitting_df


def get_all_wells_in_directory(data_dir, experiment_name='newtonrun4'):

    regex = f"^{experiment_name}-([a-z|A-Z|0-9]*)-([A-Z]|0-9]*)"
    regex = re.compile(regex)
    wells = []
    group = 1

    for f in filter(regex.match, os.listdir(data_dir)):
        well = re.search(regex, f).groups()[group]
        wells.append(well)

    wells = list(set(wells))
    return wells


def infer_reversal_potential(protocol: str, current: np.array, times, ax=None,
                             output_path=None, plot=None):

    if output_path:
        dirname = os.path.dirname(output_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    if (ax or output_path) and plot is not False:
        plot = True

    protocol_func, _, protocol_desc = get_ramp_protocol_from_csv(protocol)

    # First, find the reversal ramp. Search backwards along the protocol until we find a >= 40mV step
    step = next(filter(lambda x: x[2] >= -74, reversed(protocol_desc)))

    if step[1] - step[0] > 200 or step[1] - step[0] < 50:
        raise Exception("Failed to find reversal ramp in protocol")
    else:
        step = step[0:2]

    # Next extract steps
    istart = np.argmax(times >= step[0])
    iend = np.argmax(times > step[1])

    if istart == 0 or iend == 0 or istart == iend:
        raise Exception("Couldn't identify reversal ramp")

    times = times[istart:iend]
    current = current[istart:iend]

    voltages = np.array([protocol_func(t) for t in times])

    fitted_poly = poly.Polynomial.fit(voltages, current, 4)

    roots = np.unique([np.real(root) for root in fitted_poly.roots()
                       if root > np.min(voltages) and root < np.max(voltages)])

    # Take the last root (greatest voltage). This should be the first time that
    # the current crosses 0 and where the ion-channel kinetics are too slow to
    # play a role

    if len(roots) == 0:
        return np.nan

    if plot:
        created_fig = False
        if ax is None and output_path is not None:

            created_fig = True
            fig = plt.figure()
            ax = fig.subplots()

        ax.plot(*fitted_poly.linspace())
        ax.set_xlabel('voltage mV')
        ax.set_ylabel('current nA')
        # Now plot current vs voltage
        ax.plot(voltages, current, 'x', markersize=2, color='grey')
        ax.axvline(roots[-1], linestyle='--', color='grey', label="$E_{Kr}$")
        ax.axhline(0, linestyle='--', color='grey')
        ax.legend()

        if output_path is not None:
            fig = ax.figure
            fig.savefig(output_path)

        if created_fig:
            plt.close(fig)

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


def compute_mcmc_chains(model, times, indices, data, solver=None,
                        starting_parameters=None, sigma2=1, no_chains=1,
                        chain_length=1000, burn_in=None, log_likelihood_func=None,
                        log_transform=True):
    n = len(indices)

    if solver is None:
        solver = model.make_forward_solver_current()

    if starting_parameters is None:
        starting_parameters = model.get_default_parameters().flatten()

    if log_transform:
        # Assume that the conductance is the last parameter and that the parameters are arranged included

        # log all parameters
        no_rates = int((model.get_no_parameters() - 1)/2)
        # log_transformations = [pints.LogTransformation(1) for i in range(no_rates)]
        # identity_transformations = [pints.IdentityTransformation(1) for i in range(no_rates)]

        # Flatten and include conductance on the end
        transformations = [pints.LogTransformation(1) for i in range(len(starting_parameters))]
        transformation = pints.ComposedTransformation(*transformations)

    else:
        transformation = None

    if burn_in is None:
        burn_in = int(chain_length / 10)

    if starting_parameters is None:
        starting_parameters = model.get_default_parameters()

    if log_likelihood_func is None:
        @njit
        def log_likelihood_func(p):

            if np.any(p <= 0):
                ll = -np.inf
            else:
                try:
                    output = solver(p, times)[indices]
                    error = output - data[indices]
                    SSE = np.sum(error**2)
                    ll = -n * 0.5 * np.log(2 * np.pi * sigma2) - SSE / (2 * sigma2)

                except Exception:
                    ll = -np.inf

            return ll

    class pints_likelihood(pints.LogPDF):
        def __call__(self, p):
            return log_likelihood_func(p)

        def n_parameters(self):
            return len(starting_parameters)

    class pints_prior(pints.LogPrior):
        def __init__(self, parameters=starting_parameters):
            self.parameters = parameters

        def __call__(self, parameters=starting_parameters):
            # Make sure transition rates are not too big
            for i in range(int(len(parameters)/2)):
                a = parameters[2*i]
                b = parameters[2*i + 1]

                vs = np.array([-120, 40])

                extreme_rates = np.abs(a*np.exp(b*vs))
                max_rate = np.max(extreme_rates)
                min_rate = np.min(extreme_rates)

                if max_rate > 1e7:
                    return -np.inf

                # if min_rate < 1e-8:
                #     return -np.inf

            # Ensure that all parameters > 0
            return 0 if np.all(parameters > 0) else -np.inf

        def n_parameters(self):
            return model.get_no_parameters()

    posterior = pints.LogPosterior(pints_likelihood(), pints_prior())

    initial_likelihood = log_likelihood_func(starting_parameters)

    print('initial_parameters likelihood = ', initial_likelihood)
    if not np.isfinite(initial_likelihood):
        print("{model} MCMC failed, initial parameters had non-finite log likelihood")
        return np.full((no_chains, chain_length, len(starting_parameters)), np.nan)

    print(f"initial likelihood is {initial_likelihood}")

    mcmc = pints.MCMCController(posterior, no_chains,
                                np.tile(starting_parameters, [no_chains, 1]),
                                method=pints.HaarioBardenetACMC,
                                transformation=transformation)

    mcmc.set_max_iterations(chain_length)

    try:
        samples = mcmc.run()
    except ValueError as exc:
        print(str(exc))
        return np.full((no_chains, chain_length, len(starting_parameters)), np.nan)

    return samples[:, burn_in:, :]


def make_model_of_class(name: str, times=None, voltage=None, *args, **kwargs):
    from .BeattieModel import BeattieModel
    from .ClosedOpenModel import ClosedOpenModel
    from .WangModel import WangModel
    from .KempModel import KempModel

    from .model_generation import generate_markov_model_from_graph

    from markov_builder.models.thirty_models import (
        model_00,
        model_01,
        model_02,
        model_03,
        model_04,
        model_05,
        model_06,
        model_07,
        model_08,
        model_09,
        model_10,
        model_11,
        model_12,
        model_13,
        model_14,
        model_30,
    )

    thirty_models = [
        model_00, model_01, model_02, model_03, model_04,
        model_05, model_06, model_07, model_08, model_09, model_10,
        model_11, model_12, model_13, model_14, model_30
    ]

    thirty_models_regex = re.compile(r'^model([0-9]+)$')

    if name == 'Beattie' or name == 'BeattieModel':
        model = BeattieModel(times, voltage, *args, **kwargs)
    elif name == 'Kemp' or name == 'KempModel':
        model = KempModel(times, voltage, *args, **kwargs)
    elif name == 'CO' or name == 'ClosedOpenModel':
        model = ClosedOpenModel(times, voltage, *args, **kwargs)
    elif name == 'Wang' or name == 'WangModel':
        model = WangModel(times, voltage, *args, **kwargs)
    elif thirty_models_regex.match(name):
        model_no = int(thirty_models_regex.search(name).group(1))
        model = generate_markov_model_from_graph(thirty_models[model_no](), times, voltage,
                                                 *args, **kwargs)
    else:
        assert False, f"no model with name {name}"
    return model


class fitting_boundaries(pints.Boundaries):
    def __init__(self, full_parameters, mm, fix_parameters=None):
        self.fix_parameters = fix_parameters
        self.full_parameters = full_parameters
        self.mm = mm

    def check(self, parameters):
        parameters = parameters.copy()
        if self.fix_parameters:
            for i in np.unique(self.fix_parameters):
                parameters = np.insert(parameters, i, self.full_parameters[i])

        # rates function
        rates_func = self.mm.get_rates_func(njitted=False)

        Vs = [-120, 60]
        rates_1 = rates_func(parameters, Vs[0])
        rates_2 = rates_func(parameters, Vs[1])

        max_transition_rates = np.max(np.vstack([rates_1, rates_2]), axis=0)

        if not np.all(max_transition_rates < 1e3):
            return False

        if not np.all(max_transition_rates > 1.67e-5):
            return False

        if max([p for i, p in enumerate(parameters) if i != self.mm.GKr_index]) > 1e5:
            return False

        if min([p for i, p in enumerate(parameters) if i != self.mm.GKr_index]) < 1e-7:
            return False

        # Ensure that all parameters > 0
        return np.all(parameters > 0)

    def n_parameters(self):
        return self.mm.get_no_parameters() - \
            len(self.fix_parameters) if self.fix_parameters is not None\
            else self.mm.get_no_parameters()

    def _sample_once(self, min_log_p, max_log_p):
        for i in range(1000):
            p = np.empty(self.full_parameters.shape)
            p[-1] = self.full_parameters[-1]
            p[:-1] = 10**np.random.uniform(min_log_p, max_log_p,
                                           self.full_parameters[:-1].shape)

            if self.fix_parameters:
                p = p[[i for i in range(len(self.full_parameters)) if i not in
                       self.fix_parameters]]

            # Check this lies in boundaries
            if self.check(p):
                return p
        logging.warning("Couldn't sample from boundaries")
        return np.NaN

    def sample(self, n=1):
        min_log_p, max_log_p = [-7, 1]

        no_parameters = len(self.full_parameters) if not self.fix_parameters\
            else len(self.full_parameters) - len(self.fix_parameters)

        ret_vec = np.full((n, no_parameters), np.nan)
        for i in range(n):
            ret_vec[i, :] = self._sample_once(min_log_p, max_log_p)

        return ret_vec


def make_myokit_model(model_name: str):

    from markov_builder.models.thirty_models import (
        model_00,
        model_01,
        model_02,
        model_03,
        model_04,
        model_05,
        model_06,
        model_07,
        model_08,
        model_09,
        model_10,
        model_11,
        model_12,
        model_13,
        model_14,
        model_30,
    )

    thirty_models_regex = re.compile(r'^model([0-9]*)$')

    thirty_models = [
        model_00, model_01, model_02, model_03, model_04,
        model_05, model_06, model_07, model_08, model_09, model_10,
        model_11, model_12, model_13, model_14, model_30
    ]
    if thirty_models_regex.match(model_name):
        model_no = int(thirty_models_regex.search(model_name).group(1))
        mk_model = thirty_models[model_no]().generate_myokit_model()

    return mk_model
