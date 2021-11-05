#!/usr/bin/env python3

import common
import logging
import os
import numpy as np
import pandas as pd
from MarkovModel import MarkovModel
from BeattieModel import BeattieModel

import matplotlib.pyplot as plt

criteria = ['D', 'A']
sigma2 = 0.006

def get_sensitivities(model: MarkovModel, spike_removal_duration: float, parameters: np.array):

    # Use staircase protocol
    staircase_protocol = common.get_protocol("staircase")

    times = model.times
    voltages = model.GetVoltage()

    spikes, _ = common.detect_spikes(times, voltages, 1000)
    times, voltages = common.remove_spikes(times, voltages, spikes,
                                           spike_removal_duration)

    # Get synthetic data
    model = BeattieModel(times=times,
                         protocol=staircase_protocol)

    _, S1 = model.SimulateForwardModelSensitivities(parameters)
    return S1


def main():

    plt.style.use('classic')
    args = common.get_args(description="Plot various optimality criteria")
    output_dir = os.path.join(args.output, "plot_criteria")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    spike_removal_durations = np.linspace(0, 50, 10)

    params = np.array([2.07E-3, 7.17E-2, 3.44E-5, 6.18E-2, 4.18E-1, 2.58E-2,
                       4.75E-2, 2.51E-2, 3.33E-2])
    # params = BeattieModel.get_default_parameters()

    t_end = 15000
    t_step = 0.1
    no_steps = int(t_end/t_step)
    times = np.linspace(0, t_end, no_steps)

    model = BeattieModel(times=times,
                         protocol=common.get_protocol('staircase'),
                         Erev=common.calculate_reversal_potential(37))

    voltages = model.GetVoltage()

    D_optimalities = []
    A_optimalities = []

    current, S1 = model.SimulateForwardModelSensitivities(params)
    S1n = S1 * np.array(params)[None, :]
    spikes_times, spike_indices = common.detect_spikes(times, voltages)

    covs = []

    for time_to_remove in spike_removal_durations:

        indices = common.remove_indices(list(range(len(times))),
                                        [(spike,
                                          int(spike + time_to_remove/t_step))
                                         for spike in spike_indices])

        H = np.dot(S1[indices, :].T, S1[indices, :])
        cov = np.linalg.inv(H)

        D_optimalities.append(np.linalg.det(H))
        A_optimalities.append(np.trace(H))

        cov = sigma2 * cov
        covs.append(cov)

    D_optimalities = np.array(D_optimalities)
    A_optimalities = np.array(A_optimalities)

    D_optimalities = D_optimalities / D_optimalities.max()
    A_optimalities = A_optimalities / A_optimalities.max()

    df = pd.DataFrame(np.column_stack((spike_removal_durations, np.log(D_optimalities), np.log(A_optimalities))),
                      columns=('time removed after spikes /ms', "normalised log D-optimality", "normalised log A-optimality"))

    fig = plt.figure(figsize=(24, 20))
    axs = fig.subplots(3)
    df.set_index('time removed after spikes /ms', inplace=True)
    df.plot(legend=True, subplots=True)
    plt.savefig(os.path.join(output_dir, "criteria.pdf"))

    axs[0].plot(times, model.SimulateForwardModel(params), label='Current /nA')
    axs[1].plot(times, voltages, label='voltage / mV')
    axs[1].set_ylim(-150, 50)
    axs[0].legend()
    axs[1].legend()

    param_labels = [f"S(p{i})" for i in range(model.n_params)]

    for i in range(model.n_params):
        axs[2].plot(times, S1n[:, i], label=param_labels[i])
    axs[2].legend()

    fig.savefig(os.path.join(output_dir, 'sensitivities_plot.pdf'))

    p_of_interest = (4, 6)
    plt.clf()
    plt.figure()

    fig = plt.figure(figsize=(19, 14))
    axs = fig.subplots(2)

    offset = [params[p_of_interest[0]], params[p_of_interest[1]]]

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    cov = covs[0][p_of_interest, :]
    cov = cov[:, p_of_interest]
    eigvals, eigvecs = np.linalg.eigh(cov)
    rotation = np.arctan2(*eigvecs[::-1, 0])

    # Plot biggest region on top plot
    common.cov_ellipse(cov, nsig=1,
                       ax=axs[0],
                       resize_axes=True,
                       color=colors[len(covs) % len(colors)],
                       offset=offset,
                       label_arg="{:.2f}ms".format(
                           spike_removal_durations[i]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    first_rotation = rotation

    # Plot confidence regions starting with the largest (most observations removed)
    for i, cov in list(enumerate(covs))[::-10] + [(len(covs), covs[0])]:
        sub_cov = cov[p_of_interest, :]
        sub_cov = sub_cov[:, p_of_interest]
        print("covariance is ", cov)
        eigvals, eigvecs = np.linalg.eigh(sub_cov)
        print('eigvals are ', eigvals)
        rotation = np.arctan2(*eigvecs[::-1, 0])

        common.cov_ellipse(sub_cov, q=[0.95],
                           ax=axs[1],
                           offset=offset,
                           color=colors[i % len(colors)],
                           rotate=rotation - first_rotation,
                           resize_axes=(i == 0),
                           label_arg="{:.2f}ms".format(
                               spike_removal_durations[i % len(spike_removal_durations)]))

    axs[0].set_title(f"95% confidence regions after spike removal")
    axs[0].plot(*offset, 'x', color='red', label=f"p{p_of_interest[0]+1} = {offset[0]}, p{p_of_interest[1]+1} = {offset[1]}")
    axs[1].plot(*offset, 'x', color='red', label=f"p{p_of_interest[0]+1} = {offset[0]}, p{p_of_interest[1]+1} = {offset[1]}")
    axs[0].set_xlabel(f"p{p_of_interest[0]+1} / ms^-1")
    axs[0].set_ylabel(f"p{p_of_interest[1]+1} / ms^-1")
    axs[1].xaxis.set_ticks([])
    axs[1].yaxis.set_ticks([])
    axs[1].set_xlabel('rotated and scaled view')

    plt.legend()
    fig.savefig(os.path.join(output_dir,
                             f"p{p_of_interest[0]+1} and p{p_of_interest[1]+1} rotated confidence regions.pdf"))


    # Now plot predictions
    # We can use less timesteps now -- only interested in plotting
    pred_times = np.linspace(0, 15000, 1000)
    n_samples = 1000

    pred_model = BeattieModel(times=pred_times,
                              protocol=common.get_protocol('staircase'),
                              Erev=common.calculate_reversal_potential(37))

    def get_trajectory(p):
        try:
            soln = pred_model.SimulateForwardModel(p)
        except:
            logging.warning(f"Failed to simulate model with p={p}")
            return None

        if np.all(np.isfinite(soln)):
            return soln
        else:
            return None

    fig = plt.figure()
    plt.style.use('classic')
    mean_param_trajectory = get_trajectory(params)
    plt.plot(pred_times, mean_param_trajectory)
    ax = plt.gca()
    ax.set_ylim(np.min(mean_param_trajectory), np.max(mean_param_trajectory))
    ax.set_xlim(np.min(pred_times), np.max(pred_times))

    for i, cov in enumerate(covs):
        samples = np.random.multivariate_normal(params, cov, n_samples)

        # Filter out invalid samples
        samples = [s for s in samples if np.all(s) > 0]

        sample_trajectories = np.column_stack([traj for traj
                                               in map(get_trajectory,
                                                      samples)
                                               if traj is not None])

        plt.plot(pred_times, sample_trajectories, color='grey', alpha=0.1)
        plt.savefig(os.path.join(output_dir, "{:.2f}ms_sample_trajectories.pdf".format(spike_removal_durations[i])))


if __name__ == "__main__":
    main()
