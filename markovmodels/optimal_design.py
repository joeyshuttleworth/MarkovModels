import numpy as np
import os
import markovmodels.voltage_protocols
from markovmodels.SensitivitiesMarkovModel import SensitivitiesMarkovModel
from markovmodels.voltage_protocols import detect_spikes, remove_spikes


def D_opt_utility(desc, params, s_model, hybrid=False, solver=None,
                  removal_duration=0, ax=None, t_range=None, rescale=True,
                  use_parameters=None):
    """ Evaluate the D-optimality of design, d for a certain parameter vector"""
    times = np.arange(0, desc[-1, 0], .5)
    s_model.times = times

    if solver is None:
        s_model.protocol_description = desc
        s_model.voltage = markovmodels.voltage_protocols.make_voltage_function_from_description(desc)
        s_model.times = times
        solver = s_model.make_hybrid_solver_states(njitted=False, hybrid=False)

    voltages = np.array([s_model.voltage(t, protocol_description=desc) for t in times])

    spike_times, _ = detect_spikes(times, voltages, window_size=0)
    _, _, indices = remove_spikes(times, voltages, spike_times, removal_duration)

    if use_parameters:
        labels = s_model.markov_model.get_parameter_labels()
        params = np.array([p for lab, p in zip(labels, params)
                           if lab in use_parameters])
    res = solver(params, times=times, protocol_description=desc)

    I_Kr_sens = s_model.auxiliary_function(res.T, params, voltages)[:, 0, :].T

    if rescale:
        I_Kr_sens = I_Kr_sens * params[None, :]

    if ax:
        ax.plot(times, I_Kr_sens)

    if indices is not None:
        indices = np.arange(0, times.shape[0]).astype(int)
        I_Kr_sens = I_Kr_sens[indices, :]

    if t_range is not None:
        tstart, tend = t_range
        istart = np.argmax(times[indices] >= tstart)

        if tend > 0:
            iend = np.argmax(times[indices] > tend)
        else:
            iend = None

        I_Kr_sens = I_Kr_sens[istart:iend, :]

    ret_val = np.log(np.linalg.det(I_Kr_sens.T @ I_Kr_sens))

    if not np.isfinite(ret_val):
        return -np.inf

    else:
        return ret_val


def entropy_utility(desc, params, model, hybrid=False, removal_duration=5,
                    include_vars=None, cfunc=None, n_skip=10, t_range=(0, 0),
                    n_voxels_per_variable=10):
    """ Evaluate the D-optimality of design, d for a certain parameter vector"""
    model.protocol_description = desc
    model.voltage = markovmodels.voltage_protocols.make_voltage_function_from_description(desc)
    # output = model.make_hybrid_solver_current(njitted=False, hybrid=hybrid)
    times = np.arange(0, desc[-1, 0], .5)
    model.times = times
    voltages = np.array([model.voltage(t) for t in times])

    # Remove observations within 5ms of a spike
    removal_duration = removal_duration
    spike_times, _ = detect_spikes(times, voltages, window_size=0)
    _, _, indices = remove_spikes(times, voltages, spike_times, removal_duration)

    istart = np.argmax(times[indices] >= t_range[0])
    iend = np.argmax(times[indices] > t_range[1])

    if iend == 0:
        iend = None

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
    return np.sum((log_prob * (times_in_each_voxel / times[indices].shape[0])[istart:iend]))


def count_voxel_visitations(states, n_voxels_per_variable, times, indices,
                            n_skip, return_voxels_visited=False):
    voxels = np.full((times[indices][::n_skip].shape[0], states.shape[1]), 0, dtype=int)

    for i, (t, x) in enumerate(zip(times[indices][::n_skip], states[indices, :][::n_skip, :])):
        voxels[i, :] = np.floor(x.flatten() *
                                n_voxels_per_variable).astype(int)
    voxels = np.clip(voxels, 0, n_voxels_per_variable - 1).astype(int)

    no_states = states.shape[1]
    times_in_each_voxel = np.zeros([n_voxels_per_variable for i in range(no_states)]).astype(int)

    for voxel in voxels:
        times_in_each_voxel[tuple(voxel)] += 1

    if return_voxels_visited:
        return times_in_each_voxel, voxels
    else:
        return times_in_each_voxel


def prediction_spread_utility(desc, params, model, indices=None, hybrid=False,
                              removal_duration=0, ax=None, mode='spread',
                              solver=None, t_range=None):

    times = np.arange(0, desc[-1, 0], .5)

    if solver is None:
        model.protocol_description = desc
        model.times = times
        solver = model.make_hybrid_solver_current(hybrid=hybrid, njitted=False)

    if indices is None:
        voltages = np.array([model.voltage(t, protocol_description=desc) for t
                             in model.times])
        spike_times, _ = detect_spikes(model.times, voltages, window_size=0)
        _, _, indices = remove_spikes(model.times, voltages, spike_times, removal_duration)

    predictions = np.vstack([solver(p, times,
                                    protocol_description=desc).flatten() for p in params])
    min_pred = np.min(predictions, axis=0).flatten()[indices]
    max_pred = np.max(predictions, axis=0).flatten()[indices]

    if t_range is not None:
        tstart, tend = t_range
        istart = np.argmax(times[indices] >= tstart)

        if tend > 0:
            iend = np.argmax(times[indices] > tend)
        else:
            iend = -1

        min_pred = min_pred[istart:iend]

    times = model.times

    if ax is not None:
        ax.plot(times, predictions.T)

    if mode == 'spread':
        return np.mean(max_pred - min_pred)
    elif mode == 'std':
        return np.mean(np.std(predictions, axis=0))
    elif mode == 'quantiles':
        upper_qs = np.quantile(predictions, .9, axis=0)
        lower_qs = np.quantile(predictions, .1, axis=0)
        return np.abs(np.sum(upper_qs - lower_qs))
    else:
        raise ValueError()


def entropy_weighted_A_opt_utility(desc, params, s_model,
                                   n_voxels_per_variable=10,
                                   removal_duration=5, n_skip=1, hybrid=False,
                                   include_vars=None, solver=None,
                                   include_params=None, ax=None, t_range=(0, 0)):

    times = np.arange(0, desc[-1, 0], .5)

    voltages = np.array([s_model.voltage(t, protocol_description=desc) for t in
                         times])
    spike_times, _ = detect_spikes(times, voltages, window_size=0)
    _, _, indices = remove_spikes(times, voltages, spike_times, removal_duration)

    if solver is None:
        solver = s_model.make_hybrid_solver_states(njitted=False, hybrid=False,
                                                   protocol_description=desc)
    states = solver(params, times=times, protocol_description=desc)

    istart = np.argmax(times[indices] >= t_range[0])
    iend = np.argmax(times[indices] >= t_range[1])

    if iend == 0:
        iend = None

    if include_vars is None:
        include_vars = [i for i in range(s_model.markov_model.get_no_state_vars())]

    if np.any((states[:, include_vars] < 0) | (states[:, include_vars] > 1)):
        return -np.inf

    if ax is not None:
        ax.plot(times, states[:, include_vars])

    if include_params is None:
        no_params = len(s_model.get_default_parameters())
        include_params = [i for i in range(no_params)]

    aux_func = s_model.define_auxiliary_function()
    sens = aux_func(states.T, params,
                    voltages).reshape([times.shape[0], -1])
    sens = sens * s_model.get_default_parameters()[None, :]
    sens = sens[:, include_params]

    if include_params is not None:
        sens = sens[:, include_params]

    times_in_each_voxel, voxels_visited = count_voxel_visitations(
        states[:, include_vars].reshape(times.shape[0], -1).copy(),
        n_voxels_per_variable, times,
        indices, n_skip=n_skip,
        return_voxels_visited=True)

    log_prob = np.full(times_in_each_voxel.shape, 0)
    visited_voxel_indices = (times_in_each_voxel != 0)

    log_prob[visited_voxel_indices] = -np.log(times_in_each_voxel[visited_voxel_indices] / times[indices].shape[0])

    # This is some confusing numpy indexing manipulation.Get the indices (of
    # the flattened log_prob array) which describes the voxel that the model is
    # in at each time-step
    raveled_visited_indices = np.ravel_multi_index(voxels_visited.T, log_prob.shape)
    w_sens = sens[indices, :][::n_skip, :] \
        * log_prob.flatten()[raveled_visited_indices, None]

    w_sens = w_sens[istart:iend, :]
    weighted_A_opt = np.trace(w_sens.T @ w_sens)

    return np.log(weighted_A_opt)


def discriminate_spread_of_predictions_utility(desc, params1, params2, model1,
                                               model2, removal_duration=0,
                                               sigma2=100, hybrid=False, solver1=None,
                                               solver2=None, ax=None, t_range=(0, 0)):

    means = [0, 0]
    varis = [0, 0]
    predictions = [0, 0]

    if solver1 is None:
        solver1 = model1.make_hybrid_solver_current()
    if solver2 is None:
        solver2 = model2.make_hybrid_solver_current()

    solvers = [solver1, solver2]

    times = np.arange(0, desc[-1, 0], .5)
    voltages = np.array([model1.voltage(t, protocol_description=desc) for t in times])
    spike_times, _ = detect_spikes(times, voltages, window_size=0)
    _, _, indices = remove_spikes(times, voltages, spike_times, removal_duration)

    istart = np.argmax(times[indices] >= t_range[0])
    iend = np.argmax(times[indices] >= t_range[1])

    if iend == 0:
        iend = None

    model1.times = times
    model2.times = times

    for i, (model, params) in enumerate(zip([model1, model2], [params1, params2])):
        params = params.astype(np.float64)

        solver = solvers[i]

        predictions[i] = np.vstack([solver(p.flatten(), times=times,
                                           protocol_description=desc).flatten()
                                    for p in params])[:, indices][:, istart:iend]

        mean_pred = predictions[i].mean(axis=1)
        var_pred = predictions[i].std(axis=1)**2

        means[i] = mean_pred
        varis[i] = var_pred

    if ax is not None:
        ax.plot(times[indices], predictions[0].T, color='blue', label='model1')
        ax.plot(times[indices], predictions[1].T, color='orange', label='model1')

    return np.sum(((means[1] - means[0])**2) / (varis[1] + varis[0] + sigma2))


def save_es(es, output_dir, fname):
    # Pickle and save optimisation results
    with open(os.path.join(output_dir, fname), 'wb') as fout:
        fout.write(es.pickle_dumps())

