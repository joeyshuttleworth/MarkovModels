import numpy as np
import pandas as pd
import markovmodels.voltage_protocols
from markovmodels.voltage_protocols import detect_spikes, remove_spikes


def D_opt_utility(desc, params, s_model, hybrid=False, crhs=None, indices=None):
    """ Evaluate the D-optimality of design, d for a certain parameter vector"""
    s_model.protocol_description = desc
    s_model.voltage = markovmodels.voltage_protocols.make_voltage_function_from_description(desc)
    # output = model.make_hybrid_solver_current(njitted=False, hybrid=hybrid)

    sens = s_model.make_hybrid_solver_states(njitted=False, hybrid=False, crhs=crhs)()
    voltages = np.array([s_model.voltage(t) for t in s_model.times])
    I_Kr_sens = s_model.auxiliary_function(sens.T, params, voltages)[:, 0, :].T

    if indices is not None:
        I_Kr_sens = I_Kr_sens[indices]

    return np.log(np.linalg.det(I_Kr_sens.T @ I_Kr_sens))


def entropy_utility(desc, params, model, hybrid=False, removal_duration=5,
                    include_vars=None, cfunc=None, n_skip=10,
                    n_voxels_per_variable=10):
    """ Evaluate the D-optimality of design, d for a certain parameter vector"""
    model.protocol_description = desc
    model.voltage = markovmodels.voltage_protocols.make_voltage_function_from_description(desc)
    # output = model.make_hybrid_solver_current(njitted=False, hybrid=hybrid)
    times = np.arange(0, desc[-1][0], .5)
    model.times = times
    voltages = np.array([model.voltage(t) for t in times])

    # Remove observations within 5ms of a spike
    removal_duration = removal_duration
    spike_times, _ = detect_spikes(times, voltages, window_size=0)
    _, _, indices = remove_spikes(times, voltages, spike_times, removal_duration)

    params = params.astype(np.float64)
    states = model.make_hybrid_solver_states(njitted=False, hybrid=False, cfunc=cfunc)(params)
    n_voxels_per_variable = n_voxels_per_variable

    times_in_each_voxel = count_voxel_visitations(states[:, include_vars],
                                                  n_voxels_per_variable, times,
                                                  indices, n_skip=n_skip)
    log_prob = np.full(times_in_each_voxel.shape, 0)
    visited_voxel_indices = times_in_each_voxel != 0
    log_prob[visited_voxel_indices] = -np.log(times_in_each_voxel[visited_voxel_indices] / times[indices].shape[0]).flatten()
    return np.sum(log_prob * (times_in_each_voxel / times[indices].shape[0]))


def count_voxel_visitations(states, n_voxels_per_variable, times, indices, n_skip, return_voxels_visited=False):
    print(states)
    voxels = np.full((times[::n_skip].shape[0], states.shape[1]), 0, dtype=int)

    print(voxels)

    for i, (t, x) in enumerate(zip(times[indices][::n_skip], states[indices, ][::n_skip, :])):
        print(x)
        voxels[i, :] = np.floor(x.flatten() * n_voxels_per_variable)

    no_states = states.shape[1]
    times_in_each_voxel = np.zeros([n_voxels_per_variable for i in range(no_states)]).astype(int)

    for voxel in voxels:
        times_in_each_voxel[*voxel.astype(int)] += 1

    if return_voxels_visited:
        return times_in_each_voxel, voxels
    else:
        return times_in_each_voxel


def prediction_spread_utility(desc, params, model, indices=None, hybrid=False,
                              removal_duration=0):

    model.protocol_description = desc
    model.voltage = markovmodels.voltage_protocols.make_voltage_function_from_description(desc)

    times = model.times
    voltages = np.array([model.voltage(t) for t in times])
    spike_times, _ = detect_spikes(times, voltages, window_size=0)
    _, _, indices = remove_spikes(times, voltages, spike_times, removal_duration)

    solver = model.make_hybrid_solver_current(hybrid=hybrid, njitted=False)

    predictions = np.array([solver(p)[indices] for p in params])
    prediction_band = np.hstack([np.argmin(predictions, axis=1),
                                 np.argmax(predictions, axis=1)])[indices, :]

    return np.mean(prediction_band[:, 1] - prediction_band[:, 0])


def entropy_weighted_D_opt_utility(desc, params, s_model,
                                   n_voxels_per_variable=10,
                                   removal_duration=None, n_skip=10, crhs=None,
                                   hybrid=False, include_vars=None):

    s_model.protocol_description = desc
    s_model.voltage = markovmodels.voltage_protocols.make_voltage_function_from_description(desc)

    states = s_model.make_hybrid_solver_states(hybrid=hybrid, crhs=crhs)()
    no_states = s_model.markov_model.get_no_state_vars()
    voltages = [s_model.voltage(t) for t in s_model.times]
    sens = s_model.auxiliary_function(states.T, params, voltages).flatten()
    times = s_model.times

    spike_times, _ = detect_spikes(times, voltages, window_size=0)
    _, _, indices = remove_spikes(times, voltages, spike_times, removal_duration)

    times_in_each_voxel, voxels_visited = count_voxel_visitations(
        states[:, include_vars],
        n_voxels_per_variable, times,
        indices, n_skip=n_skip)

    log_prob = np.full(times_in_each_voxel.shape, 0)
    visited_voxel_indices = times_in_each_voxel != 0
    log_prob[visited_voxel_indices] = -np.log(times_in_each_voxel[visited_voxel_indices] / times[indices].shape[0]).flatten()

    w_sens = sens * log_prob[voxels_visited]
    weighted_D_opt = w_sens.T @ w_sens

    return weighted_D_opt

