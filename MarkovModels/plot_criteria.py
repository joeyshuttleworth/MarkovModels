#!/usr/bin/env python3

import common
import os
import numpy as np
import pandas as pd
from MarkovModel import MarkovModel
from BeattieModel import BeattieModel

import matplotlib.pyplot as plt

criteria = ['D', 'A']

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

    spike_removal_durations = np.linspace(0, 25, 10)

    params = BeattieModel.get_default_parameters()

    t_end = 15000
    t_step = 0.1
    no_steps = int(t_end/t_step)
    times = np.linspace(0, t_end, no_steps)

    model = BeattieModel(times=times,
                         protocol=common.get_protocol('staircase'),
                         Erev=common.calculate_reversal_potential(37))
    times = model.times
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

        H = np.dot(S1n[indices, :].T, S1n[indices, :])
        cov = np.linalg.inv(H)

        D_optimalities.append(np.linalg.det(H))
        A_optimalities.append(np.trace(H))
        covs.append(cov)

    D_optimalities = np.array(D_optimalities)
    A_optimalities = np.array(A_optimalities)

    D_optimalities = D_optimalities / D_optimalities.max()
    A_optimalities = A_optimalities / A_optimalities.max()

    df = pd.DataFrame(np.column_stack((spike_removal_durations, np.log(D_optimalities), np.log(A_optimalities))),
                      columns=('time removed', "normalised log D-optimality", "normalised log A-optimality"))

    fig, axs = plt.subplots(3)
    df.set_index('time removed', inplace=True)
    df.plot(legend=True, subplots=True)
    plt.savefig(os.path.join(output_dir, "criteria.pdf"))

    axs[0].plot(times, model.SimulateForwardModel(), label='Current /nA')
    axs[1].plot(times, voltages, label='voltage / mV')
    axs[0].legend()
    axs[1].legend()

    param_labels = [f"S(p{i})" for i in range(model.n_params)]

    for i in range(model.n_params):
        axs[2].plot(times, S1n[:, i], label=param_labels[i])

    fig.savefig(os.path.join(output_dir, 'sensitivities_plot.pdf'))

    p_of_interest = (4, 6)
    plt.clf()
    plt.figure()

    # Plot ellipses
    for i, cov in enumerate(reversed(covs)):
        cov = cov[p_of_interest, :]
        cov = cov[:, p_of_interest]
        sigma2 = 0.006
        cov = sigma2 * cov
        if i == 0:
            new_figure = True
        else:
            new_figure = False
        fig, ax = common.cov_ellipse(cov, nsig=1,
                                     new_figure=new_figure,
                                     label_arg="{:.2f}ms".format(
                                        spike_removal_durations[-i-1]))
        ax.title = f"params {p_of_interest} rotated confidence regions"
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
    plt.legend()
    plt.savefig(os.path.join(output_dir,
                             f"covariance_ellipses_params_{p_of_interest}.pdf"))

if __name__ == "__main__":
    main()
