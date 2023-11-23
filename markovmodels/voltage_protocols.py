import os

import numpy as np
import pandas as pd
import regex as re
from numba import njit


def get_protocol_list():
    directory = get_protocol_directory()

    files = os.listdir(directory)
    regex = re.compile(r'^([a-z|0-9]*)\.csv$')

    protocols = []
    for fname in filter(regex.match, files):
        name = re.search(regex, fname).group(1)
        protocols.append(name)
    return protocols


def get_protocol_directory():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "protocols")


def remove_spikes(times, voltages, spike_times, time_to_remove):

    spike_indices = np.array([np.argmax(times > spike_time) for spike_time in
                              spike_times])
    intervals_to_remove = [(spike,
                            spike + int(np.argmax(times[spike:] > times[spike] \
                                                  + time_to_remove)))
                           for spike in spike_indices]

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

    return np.unique(lst).astype(int)


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

        lst.append(np.array([start_t, end_t, v_start, v_end]))

    lst.append(np.array([end_t, np.inf, holding_potential, holding_potential]))
    protocol = np.vstack(lst).astype(np.float64)

    protocol_func = make_voltage_function_from_description(protocol, holding_potential)

    return protocol_func, times, protocol


def make_voltage_function_from_description(desc, holding_potential=-80):

    @njit
    def protocol_func(t: np.float64, offset=0.0,
                      protocol_description=desc):
        desc = protocol_description.reshape(-1, 4)
        t = t + offset

        if t <= 0 or t >= desc[-1, 1]:
            return holding_potential

        for row in desc:
            if t < row[1]:
                # ramp_start = desc[i][0]
                if row[3] - row[2] != 0:
                    return row[2] + (t - row[0])\
                        * (row[3] - row[2]) / \
                        (row[1] - row[0])
                else:
                    return row[3]

    return protocol_func


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
        spike_indices = [index - window_size \
                         + np.argmax(
                             np.abs(y[(index - window_size):(index + window_size)]))
                         for index in spike_indices]
        spike_indices = np.unique(spike_indices)

    if len(spike_indices) == 0:
        return [], np.array([])

    spike_indices -= 1

    return x[spike_indices], np.array(spike_indices)


def design_space_to_desc(d, t_step=.1):
    durations = d[1::2]
    voltages = d[::2]

    # Add leak ramp
    leak_ramp_steps = [[0.0, 249.9, -80.0, -80.0],
                       [250.0, 299.90000000000003, -120.0, -120.0],
                       [300.0, 699.9000000000001, -119.99999999999991, -80.00999999999992],
                       [700.0, 899.9000000000001, -80.0, -80.0],
                       [900.0, 1899.9, 40.0, 40.0],
                       [1900.0, 2399.9, -120.0, -120.0],
                       [2400.0, 3399.9, -80.0, -80.0]]

    reversal_steps = [[13900.0, 14399.900000000001, 40.0, 40.0],
                      [14400.0, 14409.900000000001, -70.0, -70.0],
                      [14410.0, 14509.900000000001, -70.00000000000364, -109.96000000000276],
                      [14510.0, 14899.900000000001, -120.0, -120.0],
                      [14900.0, 15399.800000000001, -80.0, -80.0],
                      [15399.800000000001, np.inf, -80.0, -80.0]]

    lines = leak_ramp_steps

    t_cur = lines[-1][1]
    # Add steps from the design
    for dur, v in zip(durations, voltages):
        if dur <= 0:
            print("WARNING: negative duration")
            dur = 0

        lines.append([t_cur, t_cur + dur, v, v])
        t_cur += dur + t_step

    for row in reversal_steps:
        dur = row[1] - row[0]
        tstart = t_cur
        tend = t_cur + dur
        vstart, vend = row[2], row[3]
        t_cur += dur + t_step
        lines.append([tstart, tend, vstart, vend])

    return np.vstack([np.array(line) for line in lines]).astype(np.float64)


def get_design_space_representation(desc):
    return np.array([(line[2], line[1] - line[0]) for line in desc[7:-6, :]]).flatten()


def desc_to_table(desc):
    output_lines = ['Type \t Voltage \t Duration']
    for (tstart, tend, vstart, vend) in desc:
        dur = tend - tstart

        if vstart == vend:
            _type = 'Set'
        else:
            _type = 'Ramp'

        output_lines.append(f"{_type}\t{vend}\t{dur}")

    return output_lines
