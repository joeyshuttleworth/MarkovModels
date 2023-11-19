import numpy as np
import os
import markovmodels.voltage_protocols
from markovmodels.voltage_protocols import detect_spikes, remove_spikes


def D_opt_utility(desc, params, s_model, hybrid=False, crhs=None, removal_duration=0):
    """ Evaluate the D-optimality of design, d for a certain parameter vector"""
    s_model.protocol_description = desc
    s_model.voltage = markovmodels.voltage_protocols.make_voltage_function_from_description(desc)
    s_model.times = np.arange(0, desc[-1][0], .5)

    times = s_model.times
    voltages = np.array([s_model.voltage(t) for t in times])

    spike_times, _ = detect_spikes(times, voltages, window_size=0)
    _, _, indices = remove_spikes(times, voltages, spike_times, removal_duration)

    res = s_model.make_hybrid_solver_states(njitted=False, hybrid=False, crhs=crhs)(params)

    I_Kr_sens = s_model.auxiliary_function(res.T, params, voltages)[:, 0, :].T

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
    states = model.make_hybrid_solver_states(njitted=False, hybrid=False, crhs=cfunc)(params)
    n_voxels_per_variable = n_voxels_per_variable

    if include_vars is not None:
        states = states[:, include_vars]

    times_in_each_voxel = count_voxel_visitations(states,
                                                  n_voxels_per_variable, times,
                                                  indices, n_skip=n_skip)
    log_prob = np.full(times_in_each_voxel.shape, 0)
    visited_voxel_indices = times_in_each_voxel != 0
    log_prob[visited_voxel_indices] = -np.log(times_in_each_voxel[visited_voxel_indices] / times[indices].shape[0]).flatten()
    return np.sum(log_prob * (times_in_each_voxel / times[indices].shape[0]))


def count_voxel_visitations(states, n_voxels_per_variable, times, indices, n_skip, return_voxels_visited=False):
    voxels = np.full((times[indices][::n_skip].shape[0], states.shape[1]), 0, dtype=int)

    for i, (t, x) in enumerate(zip(times[indices][::n_skip], states[indices, :][::n_skip, :])):
        voxels[i, :] = np.floor(x.flatten() * n_voxels_per_variable)

    no_states = states.shape[1]
    times_in_each_voxel = np.zeros([n_voxels_per_variable for i in range(no_states)]).astype(int)

    for voxel in voxels:
        times_in_each_voxel[tuple(voxel.astype(int))] += 1

    if return_voxels_visited:
        return times_in_each_voxel, voxels
    else:
        return times_in_each_voxel


def prediction_spread_utility(desc, params, model, indices=None, hybrid=False,
                              removal_duration=0, ax=None):

    model.protocol_description = desc
    model.voltage = markovmodels.voltage_protocols.make_voltage_function_from_description(desc)

    times = np.arange(0, desc[-1][0], .5)
    model.times = times
    voltages = np.array([model.voltage(t) for t in times])
    spike_times, _ = detect_spikes(times, voltages, window_size=0)
    _, _, indices = remove_spikes(times, voltages, spike_times, removal_duration)
    solver = model.make_hybrid_solver_current(hybrid=hybrid, njitted=False)

    predictions = np.vstack([solver(p).flatten() for p in params])
    min_pred = np.min(predictions, axis=0).flatten()[indices]
    max_pred = np.max(predictions, axis=0).flatten()[indices]

    if ax is not None:
        ax.plot(times, predictions.T)

    return np.mean(max_pred - min_pred)


def entropy_weighted_A_opt_utility(desc, params, s_model,
                                   n_voxels_per_variable=10,
                                   removal_duration=5, n_skip=1,
                                   hybrid=False, include_vars=None,
                                   include_params=None, ax=None):

    s_model.protocol_description = desc
    s_model.voltage = markovmodels.voltage_protocols.make_voltage_function_from_description(desc)
    times = np.arange(0, desc[-1][0], .5)
    s_model.times = times
    voltages = np.array([s_model.voltage(t) for t in s_model.times])
    spike_times, _ = detect_spikes(times, voltages, window_size=0)
    _, _, indices = remove_spikes(times, voltages, spike_times, removal_duration)

    # no_states = s_model.markov_model.get_no_state_vars()
    states = s_model.make_hybrid_solver_states(hybrid=hybrid)(params)

    if include_params is None:
        no_params = len(s_model.get_default_parameters())
        include_params = [i for i in range(no_params)]

    aux_func = s_model.define_auxiliary_function()
    sens = aux_func(states.T, params,
                    voltages).reshape([times.shape[0], -1])
    sens = sens / s_model.get_default_parameters()[None, :]
    sens = sens[:, include_params]

    if include_params is not None:
        sens = sens[:, include_params]

    if include_vars is None:
        include_vars = [i for i in range(s_model.markov_model.get_no_state_vars())]

    times_in_each_voxel, voxels_visited = count_voxel_visitations(
        states[:, include_vars].reshape(times.shape[0], -1),
        n_voxels_per_variable, times,
        indices, n_skip=n_skip,
        return_voxels_visited=True)

    log_prob = np.full(times_in_each_voxel.shape, 0)
    visited_voxel_indices = times_in_each_voxel != 0

    log_prob[visited_voxel_indices] = -np.log(times_in_each_voxel[visited_voxel_indices] / times[indices].shape[0])

    # This is some confusing numpy indexing manipulation.Get the indices (of
    # the flattened log_prob array) which describes the voxel that the model is
    # in at each time-step
    raveled_visited_indices = np.ravel_multi_index(voxels_visited.T, log_prob.shape)
    w_sens = sens[indices, :][::n_skip, :] * log_prob.flatten()[raveled_visited_indices, None]
    weighted_A_opt = np.trace(w_sens.T @ w_sens)

    if ax is not None:
        ax.plot(times, states[:, include_vars])

    return np.log(weighted_A_opt)


def discriminate_spread_of_predictions_utility(desc, params1, params2, model1,
                                               model2, removal_duration=0,
                                               sigma2=100, hybrid=False,
                                               ax=None):

    means = [0, 0]
    varis = [0, 0]
    predictions = [0, 0]
    for i, (model, params) in enumerate(zip([model1, model2], [params1, params2])):
        model.protocol_description = desc
        model.voltage = markovmodels.voltage_protocols.make_voltage_function_from_description(desc)

        times = np.arange(0, desc[-1][0], .5)
        model.times = times
        voltages = np.array([model.voltage(t) for t in times])

        spike_times, _ = detect_spikes(times, voltages, window_size=0)
        _, _, indices = remove_spikes(times, voltages, spike_times, removal_duration)

        solver = model.make_hybrid_solver_current(hybrid=hybrid, njitted=False)

        params = params.astype(np.float64)

        predictions[i] = np.vstack([solver(p).flatten()[indices] for p in params])

        mean_pred = predictions[i].mean(axis=1)
        var_pred = predictions[i].std(axis=1)**2

        means[i] = mean_pred
        varis[i] = var_pred

    if ax is not None:
        ax.plot(times, predictions[0].T, color='blue', label='model1')
        ax.plot(times, predictions[1].T, color='blue', label='model1')

    return np.sum(((means[1] - means[0])**2) / (varis[1] + varis[0] + sigma2))


def save_es(es, output_dir, fname):
    # Pickle and save optimisation results
    with open(os.path.join(output_dir, fname), 'wb') as fout:
        fout.write(es.pickle_dumps())

