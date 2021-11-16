#!/usr/bin/env python3

import common
import logging
import os
import numpy as np
import pandas as pd
import seaborn as sns
from MarkovModel import MarkovModel
from BeattieModel import BeattieModel

import matplotlib.pyplot as plt

criteria = ['D', 'A', 'G']
sigma2 = 0.0001


def main():

    plt.style.use('classic')
    args = common.get_args(description="Plot various optimality criteria")
    output_dir = os.path.join(args.output, "plot_criteria")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    spike_removal_durations = np.linspace(0, 50, 20)

    params = np.array([2.07E-3, 7.17E-2, 3.44E-5, 6.18E-2, 4.18E-1, 2.58E-2,
                       4.75E-2, 2.51E-2, 3.33E-2])

    t_end = 15000
    t_step = 0.1
    no_steps = int(t_end/t_step)
    times = np.linspace(0, t_end, no_steps)

    model = BeattieModel(times=times,
                         protocol=common.get_protocol('staircase'),
                         Erev=common.calculate_reversal_potential(37),
                         parameters=params)

    voltages = model.GetVoltage()

    D_optimalities = []
    A_optimalities = []
    G_optimalities = []

    current, S1 = model.SimulateForwardModelSensitivities(params)
    # S1n = S1 * np.array(params)[None, :]
    spike_times, spike_indices = common.detect_spikes(times, voltages,
                                                      window_size=1)

    covs = []
    for time_to_remove in spike_removal_durations:
        indices = common.remove_indices(list(range(len(times))),
                                        [(spike,
                                          int(spike + time_to_remove/t_step))
                                         for spike in spike_indices])

        H = np.dot(S1[indices, :].T, S1[indices, :])

        D_optimalities.append(np.linalg.det(H))
        A_optimalities.append(np.trace(H))

        cov = np.linalg.inv(H)
        G_optimalities.append(np.sum(
            np.apply_along_axis(
                lambda row: row @ cov @ row.T,
                1, S1)))

        cov = sigma2 * cov
        covs.append(cov)

    D_optimalities = np.array(D_optimalities)
    A_optimalities = np.array(A_optimalities)
    G_optimalities = np.array(G_optimalities)

    D_optimalities = D_optimalities / D_optimalities.max()
    A_optimalities = A_optimalities / A_optimalities.max()
    G_optimalities = G_optimalities / G_optimalities.max()

    df = pd.DataFrame(np.column_stack((spike_removal_durations, np.log(D_optimalities), np.log(A_optimalities), np.log(A_optimalities))),
                      columns=('time removed after spikes /ms', "normalised log D-optimality", "normalised log A-optimality", "normalised log G-optimality"))
    df.set_index('time removed after spikes /ms', inplace=True)
    df.plot(legend=True, subplots=True)
    plt.savefig(os.path.join(output_dir, "criteria.pdf"))

    plot_regions(times, model, spike_times, spike_indices, output_dir,
                 spike_removal_durations, (4, 6))
    plot_regions(times, model, spike_times, spike_indices, output_dir,
                 spike_removal_durations, (5, 7))

    plot_regions(times, model, spike_times, spike_indices, output_dir,
                 spike_removal_durations, (4, 5))

    plot_regions(times, model, spike_times, spike_indices, output_dir,
                 spike_removal_durations, (6, 7))

    # Sample steady states and timescales
    for i, cov in enumerate(covs):
        plt.clf()
        fig = plt.gcf()
        axs = fig.subplots(2)

        try:
            a_inf, tau_a, r_inf, tau_r = monte_carlo_tau_r(
                params, cov, voltage=40)
            # axs[0].scatter(a_inf, tau_a, marker='x')
            sns.kdeplot(data=pd.DataFrame(zip(a_inf, tau_a), columns=[
                        'a_inf', 'tau_a']), shade=True, fill=True, ax=axs[0], x='a_inf', y='tau_a')
            axs[0].set_ylabel('tau_a')
            axs[0].set_xlabel('a_inf')
            axs[0].set_title(
                f"+40mV with {spike_removal_durations[i]:.2f}ms removed")
            # axs[1].scatter(r_inf, tau_r, marker='x')
            axs[1].set_ylabel('tau_r')
            axs[1].set_xlabel('r_inf')
            sns.kdeplot(data=pd.DataFrame(zip(r_inf, tau_r), columns=[
                        'r_inf', 'tau_r']), shade=True, ax=axs[1], x='r_inf', y='tau_r')

            fig.savefig(os.path.join(output_dir, f"{i}.png"))
        except Exception as e:
            logging.warning(e)
            break

    # Now plot predictions
    # We can use less timesteps now -- only interested in plotting
    pred_times = np.linspace(0, 15000, 1000)
    n_samples = 100

    pred_model = BeattieModel(times=pred_times,
                              protocol=common.get_protocol('staircase'),
                              Erev=common.calculate_reversal_potential(37),
                              parameters=params)

    def get_trajectory(p):
        try:
            soln = pred_model.SimulateForwardModel(p)
        except Exception as e:
            logging.warning(f"Failed to simulate model with p={p}")
            return None

        if np.all(np.isfinite(soln)):
            soln += np.random.normal(0, np.sqrt(sigma2), soln.shape)
            return soln
        else:
            return None

    fig = plt.figure(figsize=(14, 12))
    plt.style.use('classic')

    mean_param_trajectory = current

    ax = plt.gca()
    ax.set_ylim(np.min(mean_param_trajectory), np.max(mean_param_trajectory))
    ax.set_xlim(np.min(pred_times), np.max(pred_times))

    for i, cov in list(enumerate(covs)):
        samples = np.random.multivariate_normal(params, cov, n_samples)

        # Filter out invalid samples
        samples = [s for s in samples if np.all(s) > 0]

        print(f"spike_indices are {spike_indices}")

        mean_estimate_uncertainty = np.apply_along_axis(lambda row:
                                                        row @ cov @ row.T, 1,
                                                        S1)

        upper_bound = mean_param_trajectory + 1.96*mean_estimate_uncertainty
        lower_bound = mean_param_trajectory - 1.96*mean_estimate_uncertainty

        plt.clf()
        fig = plt.figure(figsize=(14,12))
        axs = fig.subplots(2)

        axs[0].fill_between(times, lower_bound, upper_bound, color='grey',
                         alpha=0.25)
        axs[0].plot(times, mean_param_trajectory, 'red')
        axs[0].set_ylim(np.min(current)*1.5, np.max(current)*1.5)
        for sample in samples:
            trajectory = get_trajectory(sample)
            if trajectory is not None:
                axs[0].plot(pred_times, trajectory, color='grey', alpha=0.05)
        axs[1].plot(times, model.GetVoltage())
        axs[1].set_xlabel("time /ms")
        axs[1].set_ylabel("membrane voltage /mV")

        fig.savefig(os.path.join(output_dir, "{:.2f}ms_sample_trajectories.png".format(
            spike_removal_durations[i])))


def monte_carlo_tau_r(mean, cov, n_samples=10000, voltage=40):
    samples = np.random.multivariate_normal(mean, cov, n_samples)
    k1 = samples[:, 0] * np.exp(samples[:, 1] * voltage)
    k2 = samples[:, 2] * np.exp(-samples[:, 3] * voltage)
    k3 = samples[:, 4] * np.exp(samples[:, 5] * voltage)
    k4 = samples[:, 6] * np.exp(-samples[:, 7] * voltage)

    a_inf = 1 / (k1 + k2)
    tau_a = k1 / (k1 + k2)

    r_inf = 1 / (k3 + k4)
    tau_r = k4 / (k3 + k4)

    return a_inf, tau_a, r_inf, tau_r


def plot_regions(times, model, spike_times, spike_indices, output_dir, spike_removal_durations, p_of_interest=(4, 6)):
    voltages = model.GetVoltage()
    fig = plt.figure(figsize=(24, 20))
    axs = fig.subplots(3)
    current, S1 = model.SimulateForwardModelSensitivities()
    params = model.get_default_parameters()

    t_step = (model.times[-1] - model.times[0])/len(model.times)

    axs[0].plot(times, current, label='Current /nA')
    axs[0].plot(spike_times, current[spike_indices], 'x', color='red')
    axs[1].plot(times, voltages, label='voltage / mV')
    axs[1].set_ylim(-150, 50)
    axs[0].legend()
    axs[1].legend()

    param_labels = [f"S(p{i})" for i in range(model.n_params)]

    for i in range(model.n_params):
        axs[2].plot(times, S1[:, i]*params[i], label=param_labels[i])
    axs[2].legend()

    fig.savefig(os.path.join(output_dir, 'sensitivities_plot.pdf'))

    plt.clf()
    plt.figure()

    fig = plt.figure(figsize=(19, 14))
    axs = fig.subplots(2)

    offset = [params[p_of_interest[0]], params[p_of_interest[1]]]

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    covs = []
    for time_to_remove in spike_removal_durations:
        indices = common.remove_indices(list(range(len(times))),
                                        [(spike,
                                          int(spike + time_to_remove/t_step))
                                         for spike in spike_indices])

        H = np.dot(S1[indices, :].T, S1[indices, :])
        covs.append(np.linalg.inv(H))

    cov = covs[0][p_of_interest, :]
    cov = cov[:, p_of_interest]
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Plot biggest region on top plot
    common.cov_ellipse(cov, nsig=1,
                       ax=axs[0],
                       resize_axes=True,
                       color=colors[0 % len(colors)],
                       offset=offset,
                       label_arg="{:.2f}ms".format(
                           spike_removal_durations[-1]))
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Plot confidence regions starting with the largest (most observations
    # removed)
    first_rotation = np.arctan2(*eigvecs[::-1, -1])

    for i, cov in reversed(list(enumerate(covs[0:5]))):
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
                           resize_axes=(i == len(covs)-1),
                           label_arg="{:.2f}ms".format(
                               spike_removal_durations[i % len(spike_removal_durations)]))

    axs[0].set_title(f"95% confidence regions after spike removal")
    axs[0].plot(*offset, 'x', color='red',
                label=f"p{p_of_interest[0]+1} = {offset[0]}, p{p_of_interest[1]+1} = {offset[1]}")
    axs[1].plot(*offset, 'x', color='red',
                label=f"p{p_of_interest[0]+1} = {offset[0]}, p{p_of_interest[1]+1} = {offset[1]}")
    axs[0].set_xlabel(f"p{p_of_interest[0]+1} / ms^-1")
    axs[0].set_ylabel(f"p{p_of_interest[1]+1} / ms^-1")
    axs[1].xaxis.set_ticks([])
    axs[1].yaxis.set_ticks([])
    axs[1].set_xlabel('rotated and scaled view')

    plt.legend()
    fig.savefig(os.path.join(output_dir,
                             f"p{p_of_interest[0]+1} and p{p_of_interest[1]+1} rotated confidence regions.pdf"))


if __name__ == "__main__":
    main()
