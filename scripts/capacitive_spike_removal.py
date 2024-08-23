#!/usr/bin/env python3

import argparse
import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import pints
import pints.plot
from multiprocessing import Pool

from markovmodels.utilities import setup_output_directory
from markovmodels.model_generation import make_model_of_class
from markovmodels.voltage_protocols import get_ramp_protocol_from_csv
from markovmodels.ArtefactModel import ArtefactModel
from markovmodels.SensitivitiesMarkovModel import SensitivitiesMarkovModel
from markovmodels.plotting import cov_ellipse

from markovmodels.voltage_protocols import remove_spikes, detect_spikes,\
    make_voltage_function_from_description, get_ramp_protocol_from_json

from numba import njit

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import cycler

# Don't use scientific notation offsets on plots (it's confusing)
mpl.rcParams["axes.formatter.useoffset"] = False

sigma = 0.01


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_class", default='model3')
    parser.add_argument("--figsize", nargs=2, type=float, default=[5.3, 6])
    parser.add_argument("-n", "--n_samples", type=int, default=50)
    parser.add_argument("-N", "--no_chains", type=int, default=4)
    parser.add_argument("-l", "--chain_length", type=int, default=1000)
    parser.add_argument("-b", "--burn-in", type=int, default=None)
    parser.add_argument("-H", "--heatmap_size", type=int, default=0)
    parser.add_argument("-c", "--cpus", type=int, default=1)
    parser.add_argument("-i", "--max_iterations", type=int, default=None)
    parser.add_argument("-s", "--short", help="run with a reduced set of removal times", action='store_true')
    parser.add_argument("-L", "--linear_model", help="Run with a simple linear model\
    instead (debugging)", action='store_true')
    parser.add_argument('-o', '--output')
    parser.add_argument('--reversal_potential', type=float, default=-91.71)
    parser.add_argument('--parameters_file')

    global args
    args = parser.parse_args()

    global optimiser
    optimiser = pints.CMAES

    if args.linear_model:
        optimiser = pints.NelderMead

    # Setup a pool for parallel computation
    pool = Pool(args.cpus)

    output_dir = setup_output_directory(args.output, "plot_criteria")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    spike_removal_durations = np.unique(np.concatenate((np.linspace(0, 11, 25),
                                        np.linspace(10, 100, 46),
                                        # np.linspace(100, 250, 10)
                                                        )))

    # Write durations to file
    pd.DataFrame(spike_removal_durations[:, None],
                 columns=('removal_duration',)).to_csv(os.path.join(output_dir, "removal_durations.csv"))

    print(spike_removal_durations)

    if args.short:
        spike_removal_durations = np.array([0, 1])

    protocol_func, times, desc = get_ramp_protocol_from_csv('staircaseramp')

    desc = np.vstack((desc, [[desc[-1, 1], np.inf, -80.0, -80.0]]))
    print(desc)

    tend = desc[np.isfinite(desc[:, 1]), 1].max()
    tstart = 0
    tstep = .5
    times = np.linspace(tstart, tend, int((tend - tstart) / tstep))
    voltages = np.array([protocol_func(t) for t in times])

    full_times = times

    Erev = args.reversal_potential

    print(f"Reversal potential is {Erev}")

    if args.parameters_file:
        default_parameters = np.loadtxt(args.parameters_file, delimiter=',').flatten()

    model = make_model_of_class(args.model_class, times=times,
                                voltage=protocol_func, E_rev=Erev,
                                protocol_description=desc)

    params = model.get_default_parameters()
    solver = model.make_forward_solver_current()

    # Plot representative sample from DGP
    sample_mean = solver()
    data = generate_synthetic_data(args.model_class, solver, model.times,
                                   sigma, n_repeats=args.n_samples).T

    columns = ["times"] + [f"current_{i}" for i in range(args.n_samples)]

    pd.DataFrame(np.vstack((times, *data.T)).T, columns=columns).to_csv(
        os.path.join(output_dir, "synthetic_data.csv"))

    D_optimalities = []
    A_optimalities = []
    mles = []

    logging.info("Getting model sensitivities")
    s_model = SensitivitiesMarkovModel(model)

    v = 0.0
    p = s_model.get_default_parameters()
    y = np.full(s_model.get_no_state_vars(), 1.0)

    print('S1 func', s_model.func_S1(y, p, v).flatten())

    s_solver = s_model.make_hybrid_solver_states(hybrid=False, njitted=True)
    states = s_solver()

    print(states)
    S1 = s_model.auxiliary_function(states.T, params, voltages)[:, 0, :].T

    s_S1 = S1 * params[None, :]

    spike_times, spike_indices = detect_spikes(times, voltages, threshold=10)

    print(f"Spike locations are: {spike_times}")

    covs = []
    indices_used = []

    sample_fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    sample_axs = sample_fig.subplots(2, height_ratios=[.33, 1])

    sample_axs[0].plot(times, voltages)

    r_states = model.y
    no_states = len(r_states)

    print(r_states)
    print(list(s_model.y))
    print(model.get_no_state_vars())

    labels= [r'$\dfrac{\mathrm{d}}{\mathrm{d}' f"{p}" r'} I_\mathrm{Kr}(t)$' for p in model.p]
    print(S1.shape)
    print(labels)
    sample_axs[1].plot(times, s_S1,
                       label=labels)

    sample_axs[1].legend()
    sample_fig.savefig(os.path.join(output_dir, 'normalised_sensitivities'))

    # TODO Compute MLEs for each repeat and each removal duration

    for time_to_remove in spike_removal_durations:
        print(time_to_remove)
        _, _, indices = remove_spikes(times, voltages, spike_times,
                                      time_to_remove)

        indices_used.append(indices)
        # Plot the observations being removed
        sample_fig.clf()
        axs = sample_fig.subplots(2)

        for ax in axs:
            ax.spines[['top', 'right']].set_visible(False)

        axs[0].plot(times*1e-3, voltages)
        for t in spike_times:
            axs[0].axvspan(1e-3 * t, (t + time_to_remove)*1e-3, alpha=0.4, color='red', lw=0)

        # Plot the observations under consideration
        axs[1].plot(times*1e-3, data[:, 0].flatten(), color='grey',
                    alpha=.5)
        axs[1].plot(times*1e-3, solver(params))
        axs[1].set_xlabel(r'$t$ (s)')
        axs[1].set_xticklabels([])
        axs[0].set_ylabel(r'$V$ (mV)')
        axs[0].set_xticklabels([])
        axs[1].set_ylabel(r'$I_\mathrm{Kr}$ (pA)')
        axs[1].set_xlim([0, times[-1]])

        sample_fig.savefig(os.path.join(output_dir, f"spike_removal_{time_to_remove:.0f}.png"))
        for ax in axs:
            ax.cla()

        H = s_S1[indices, :].T @ s_S1[indices, :]
        H_inv = np.linalg.inv(H)
        D_optimalities.append(np.linalg.det(H_inv))
        A_optimalities.append(np.trace(H_inv))

        print(D_optimalities[-1])
        print(A_optimalities[-1])

        # G_optimalities.append(np.max(np.diag(S1[indices, :] @ H_inv @ S1[indices, :].T)))

        cov = sigma**2 * np.linalg.inv(S1[indices, :].T @ S1[indices, :])
        covs.append(cov)

        # SSE = np.sum((solver()[indices] - data[indices])**2)
        # print(f"{time_to_remove:.2f}ms removed: SSE is {SSE}")

    # if args.heatmap_size > 0:

    #     logging.info(f"Drawing {args.heatmap_size} x {args.heatmap_size} likelihood heatmap")

    #     args_list = [(args.model_class, times, data, output_dir,
    #                   time_to_remove, params, indices) for time_to_remove,
    #                  indices in zip(spike_removal_durations, indices_used)]

    #     args_list = args_list if args.short else args_list[0:20]
    #     pool.map(draw_heatmaps, *zip(*args_list))

    #     logging.info("Finished drawing heatmaps")

    D_optimalities = np.array(D_optimalities)
    A_optimalities = np.array(A_optimalities)

    # Normalise with respect to first score i.e 0 ms removed
    D_optimalities = D_optimalities / D_optimalities[0]
    A_optimalities = A_optimalities / A_optimalities[0]

    df = pd.DataFrame(np.column_stack((spike_removal_durations*1e-3,
                                       np.log(D_optimalities),
                                       np.log(A_optimalities))),
                      columns=('time removed after spikes /s',
                               "normalised log D-optimality",
                               "normalised log A-optimality"))

    df.set_index('time removed after spikes /s', inplace=True)

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    ax = fig.subplots()
    ax.spines[['top', 'right']].set_visible(False)

    df.plot(legend=True, subplots=True, ax=ax)

    print("plotting criteria")
    fig.savefig(os.path.join(output_dir, "criteria.pdf"))

    # Now plot it zoomed in on the first 25ms
    fig.savefig(os.path.join(output_dir, "criteria_zoomed_in.pdf"))

    plt.close(fig)

    fig = plt.figure(figsize=[args.figsize[0], args.figsize[1]/2],
                               constrained_layout=True)
    ax = fig.subplots()
    df[df.index <= 25.0].plot(legend=True, subplots=False, ax=ax)
    ax.set_xlabel(r'$t$ (s)')
    ax.set_ylabel('')

    xticks = ax.get_xticks()
    xticks = list(xticks) + [5.0e-3, 1e-2]
    ax.set_xticks(np.unique(xticks))
    fig.savefig(os.path.join(output_dir, "criteria_shared_zoomed.pdf"))

    conf_fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    conf_axs = conf_fig.subplots(2)

    sigma2 = sigma ** 2

    labels = [f"{duration:.2f}ms removed" for duration in spike_removal_durations[::4]]
    fig = plt.figure(figsize=args.figsize)
    axs = fig.subplots(3)

    param_fig = plt.figure(figsize=args.figsize)
    param_axs = param_fig.subplots(model.get_no_parameters())

    std_fig = plt.figure(figsize=args.figsize)
    std_axs = std_fig.subplots(5)

    conf_fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    sample_fig.clf()
    sample_axs = sample_fig.subplots(2, height_ratios=[0.33, 1])

    indices_to_plot = [0, 2, 10, 20, 40]
    covs_to_plot = [covs[i] for i in indices_to_plot]
    durations_to_plot = [covs[i] for i in indices_to_plot]

    plot_regions(covs_to_plot, labels, params, output_dir,
                 durations_to_plot, conf_fig, sigma2, p_of_interest=(4, 6))
    plot_regions(covs_to_plot, labels, params, output_dir,
                 durations_to_plot, conf_fig, sigma2, (5, 7))

    plot_regions(covs_to_plot, labels, params, output_dir,
                 durations_to_plot, conf_fig, sigma2, (4, 5))

    plot_regions(covs_to_plot, labels, params, output_dir,
                 durations_to_plot, conf_fig, sigma2, (6, 7))

    plot_regions(covs_to_plot, labels, params, output_dir,
                 durations_to_plot, conf_fig, sigma2, (0, 1))

    for time_to_remove, cov in list(zip(spike_removal_durations, covs)):
        for ax in sample_axs:
            ax.cla()
        plot_sample_trajectories(solver, full_times, voltages, time_to_remove,
                                 params, cov, sample_axs, args.n_samples,
                                 spike_indices)
        try:
            sample_fig.savefig(os.path.join(output_dir, f"sample_trajectories_{time_to_remove:.2f}.png"))
        except Exception:
            logging.warning(f"Failed to plot trajectories for {time_to_remove}")
            pass



def plot_sample_trajectories(solver, times, voltages, removal_duration, params, cov, axs, n_samples, spike_indices):

    mean_param_trajectory = solver(params)
    axs[1].plot(times*1e-3, mean_param_trajectory, 'blue')
    axs[1].set_ylim(np.min(mean_param_trajectory) * 1.5, np.max(mean_param_trajectory) * 1.5)

    axs[0].plot(times * 1e-3, voltages)

    for spike in spike_indices:
        axs[0].axvspan(times[spike]*1e-3, (times[spike] + removal_duration) * 1e-3, alpha=0.2, color='red', lw=0)

    def get_trajectory(p):
        try:
            soln = solver(p, times)
            return soln
        except Exception as e:
            print(str(e))
            return np.full(times.shape, np.nan)

    samples = np.random.multivariate_normal(params, cov, n_samples)
    count = 0

    indices = np.unique(np.array(list(range(len(times)))[::50] + list(spike_indices)))
    for sample in samples:
        trajectory = get_trajectory(sample)
        if np.all(np.isfinite(trajectory)):
            count += 1
        axs[1].plot(times*1e-3, trajectory, color='grey', alpha=0.3)

    print(f"{removal_duration:.2f}: Successfully ran {count} out of {n_samples} simulations")

    axs[1].set_xlabel(r'$t$ (s)')
    axs[0].set_ylabel(r'$V_\mathrm{m}$ (mV)')

    for ax in axs:
        ax.spines[['top', 'right']].set_visible(False)


def monte_carlo_tau_inf(mean, cov, n_samples=10000, voltage=40):
    samples = np.random.multivariate_normal(mean, cov, n_samples)
    k1 = (samples[:, 0] * np.exp(samples[:, 1] * voltage)).flatten()
    k2 = (samples[:, 2] * np.exp(-samples[:, 3] * voltage)).flatten()
    k3 = (samples[:, 4] * np.exp(samples[:, 5] * voltage)).flatten()
    k4 = (samples[:, 6] * np.exp(-samples[:, 7] * voltage)).flatten()

    a_inf = k1 / (k1 + k2)
    tau_a = 1 / (k1 + k2)

    r_inf = k4 / (k3 + k4)
    tau_r = 1 / (k3 + k4)

    return a_inf, tau_a, r_inf, tau_r, samples[:, -1].flatten()


def plot_regions(covs, labels, params, output_dir, spike_removal_durations,
                 fig, sigma2, p_of_interest=(4, 6)):
    offset = [params[p_of_interest[0]], params[p_of_interest[1]]]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    cov = covs[0][p_of_interest, :]
    cov = cov[:, p_of_interest]
    eigvals, eigvecs = np.linalg.eigh(cov)

    fig.clf()
    axs = fig.subplots(2)

    cmap = sns.cubehelix_palette(start=2, as_cmap=True)
    n = len(covs)
    colors = [cmap(i/n) for i in range(n)]

    axs[0].set_prop_cycle(cycler('color', colors))
    axs[1].set_prop_cycle(cycler('color', colors))

    # Plot smallest region on top plot
    cov_ellipse(cov, nsig=1,
                ax=axs[0],
                resize_axes=True,
                color=colors[0 % len(colors)],
                offset=offset,
                label=labels[0])

    eigvals, eigvecs = np.linalg.eigh(cov)

    # Plot confidence regions starting with the largest (most observations
    # removed)
    first_rotation = np.arctan2(*eigvecs[::-1, 0])

    no_regions = len(list(covs))

    for i, cov in reversed(list(enumerate(covs))):
        sub_cov = cov[p_of_interest, :]
        sub_cov = sub_cov[:, p_of_interest]
        eigvals, eigvecs = np.linalg.eigh(sub_cov)

        rotation = np.arctan2(*eigvecs[::-1, 0])

        cov_ellipse(sub_cov, q=[0.95],
                    ax=axs[1],
                    offset=offset,
                    color=colors[i % len(colors)],
                    label=labels[i])

    axs[0].set_title(f"95% confidence regions after spike removal")
    axs[0].plot(*offset, 'x', color='red')
    axs[1].plot(*offset, 'x', color='red')

    for ax in axs:
        ax.spines[['top', 'right']].set_visible(False)

    ax.legend()
    fig.savefig(os.path.join(output_dir,
                             f"p{p_of_interest[0]+1} and p{p_of_interest[1]+1} confidence regions.png"))


def draw_likelihood_heatmap(model, solver, params, mle, cov, mle_cov, data, sigma2,
                            ranges, no_points, p_index, output_dir,
                            subset_indices=None, filename=None, title=None):

    if filename is None:
        filename = f"log_likelihood_heatmap_{p_index[0]}_{p_index[1]}"

    if subset_indices is None:
        subset_indices = list(range(len(model.times)))

    x_index = p_index[0]
    y_index = p_index[1]

    xs = np.linspace(ranges[0][0], ranges[0][1], no_points)
    ys = np.linspace(ranges[1][0], ranges[1][1], no_points)

    print(f"Modifying variables {x_index} and {y_index}")

    times = model.times[subset_indices]

    n = len(subset_indices)

    @njit
    def log_likelihood(x, y):
        solver_input = np.copy(params)
        solver_input[x_index] = x
        solver_input[y_index] = y
        try:
            output = solver(solver_input)[subset_indices]
        except:
            output = np.full(times.shape, np.nan)
        error = output - data[subset_indices]
        SSE = np.sum(error**2)
        return - n * 0.5 * np.log(2 * np.pi * sigma2) - SSE / (2 * sigma2)

    fix_parameters = [i for i in range(9) if i not in p_index]

    print(f"Fixing parameters {fix_parameters}")

    xs, ys = np.meshgrid(xs, ys)
    zs = []
    for x, y in zip(xs.flatten(), ys.flatten()):
        zs.append(log_likelihood(x, y))

    zs = np.array(zs).reshape(xs.shape)

    fig = plt.figure(figsize=args.figsize)
    ax = fig.subplots()

    c = ax.pcolormesh(
        xs,
        ys,
        zs,
        vmax=np.max(zs),
        vmin=np.max(zs) - 10,
        label="log likelihood",
        shading="gouraud",
        cmap="viridis",
        rasterized=True
    )

    # Draw confidence region over the heatmap
    subcov = cov[(x_index, y_index), :][:, (x_index, y_index)]
    cov_ellipse(subcov, offset=(params[x_index], params[y_index]), q=[0.95], ax=ax,
                       color='red', label='Approximated sampling distribution of MLE (95%)')
    ax.plot(params[x_index], params[y_index], marker='x', color='red', linestyle='None', label='true_params')

    ax.plot(mle[p_index[0]], mle[p_index[1]], marker='o', linestyle='None', color='pink', label='mle')

    # Draw normal approximation of 95% confidence region (marginal)
    subcov = mle_cov[(x_index, y_index), :][:, (x_index, y_index)]
    cov_ellipse(subcov, offset=(mle[x_index], mle[y_index]), q=[0.95], ax=ax,
                       color='pink', label='95% confidence region (normal approximation)')
    # Draw 2 param versions
    mle_2param, _ = common.fit_model(model, data, params, fix_parameters=fix_parameters,
                                     subset_indices=subset_indices, solver=solver,
                                     max_iterations=args.max_iterations,
                                     method=optimiser)
    mle_params = np.copy(params)
    mle_params[x_index] = mle_2param[0]
    mle_params[y_index] = mle_2param[1]
    S1 = model.SimulateForwardModelSensitivities(mle_params)[1][subset_indices]
    S1 = S1[:, [x_index, y_index]]
    try:
        mle_2param_cov = np.linalg.inv(S1.T @ S1) * sigma2
        cov_ellipse(mle_2param_cov, offset=mle_2param, q=[0.95], ax=ax,
                           color='purple', label='Conditional 95% confidence region (normal approximation)')
    except np.linalg.LinAlgError:
        print("Failed to invert Hessian matrix")
        print(S1)

    ax.spines[['top', 'right']].set_visible(False)
    ax.plot(*mle_2param, marker='+', linestyle='None', color='purple', label='conditional mle')

    ax.set_xlabel(f"p_{p_index[0]+1}")
    ax.set_ylabel(f"p_{p_index[1]+1}")
    ax.axis([ranges[0][0], ranges[0][1], ranges[1][0], ranges[1][1]])

    ax.legend()

    if title is not None:
        ax.set_title(title)

    fig.colorbar(c, label="log likelihood of data")
    fig.savefig(os.path.join(output_dir, filename))

    fig.clf()

    # Plot MLE trajectories
    ax = fig.subplots()
    ax.spines[['top', 'right']].set_visible(False)

    ax.plot(model.times, solver(mle, model.times), label='MLE trajectory')
    ax.plot(model.times, data, label='data', alpha=0.1)
    ax.plot(model.times, solver(mle_params, model.times), label='conditional MLE trajectory (conditional)')
    ax.plot(model.times, solver(mle, model.times), label='conditional MLE trajectory')
    ax.legend()
    fig.savefig(os.path.join(output_dir, f"{filename}_traces.png"))
    ax.cla()

    print(f"RMSE of true params = {((solver(params) - data)**2)[subset_indices].mean()}")
    print(f"RMSE of MLE params = {((solver(mle) - data)**2)[subset_indices].mean()}")
    print(f"RMSE of conditional MLE params = {((solver(mle_params) - data)**2)[subset_indices].mean()}")

    return


def compute_tau_inf_from_samples(samples, voltage=40):
    k1 = samples[:, :, 0] * np.exp(samples[:, :, 1] * voltage)
    k2 = samples[:, :, 2] * np.exp(-samples[:, :, 3] * voltage)
    k3 = samples[:, :, 4] * np.exp(samples[:, :, 5] * voltage)
    k4 = samples[:, :, 6] * np.exp(-samples[:, :, 7] * voltage)

    a_inf = k1 / (k1 + k2)
    tau_a = 1 / (k1 + k2)

    r_inf = k4 / (k3 + k4)
    tau_r = 1 / (k3 + k4)

    return a_inf, tau_a, r_inf, tau_r


def compute_tau_inf_from_params(params, voltage=40):
    k1 = params[0] * np.exp(params[1] * voltage)
    k2 = params[2] * np.exp(-params[3] * voltage)
    k3 = params[4] * np.exp(params[5] * voltage)
    k4 = params[6] * np.exp(-params[7] * voltage)

    a_inf = k1 / (k1 + k2)
    tau_a = 1 / (k1 + k2)

    r_inf = k4 / (k3 + k4)
    tau_r = 1 / (k3 + k4)

    return a_inf, tau_a, r_inf, tau_r


# def draw_heatmaps(model_class, times, data, output_dir, time_to_remove, params, indices=None):

#     if indices is None:
#         indices = list(range(len(times)))

#     print(f"using {len(indices)} indices")

#     protocol_func, tstart, tend, tstep, protocol_desc = common.get_ramp_protocol_from_csv('staircase')

#     Erev = common.calculate_reversal_potential(310.15)
#     model = model_class(times=times, voltage=protocol_func, Erev=Erev, parameters=params)
#     model.window_locs = [t for t, _, _, _ in protocol_desc]
#     model.protocol_description = protocol_desc
#     solver = model.make_forward_solver_current()

#     S1 = model.SimulateForwardModelSensitivities(params)[1][indices]
#     cov = sigma2 * np.linalg.inv(S1.T @ S1)

#     mle, _ = common.fit_model(model, data, params, subset_indices=indices,
#                               solver=solver, max_iterations=args.max_iterations,
#                               method=optimiser)

#     S1_tmp = model.SimulateForwardModelSensitivities(mle)[1][indices]
#     mle_cov = sigma2 * np.linalg.inv(S1_tmp.T @ S1_tmp)

    # for x_index, y_index in [(4, 6), (5, 7), (4, 7)]:
    #     width = np.sqrt(cov[x_index, x_index]) * 3
    #     height = np.sqrt(cov[y_index, y_index]) * 3

    #     x = params[x_index]
    #     y = params[y_index]

    #     ranges = [[x - width, x + width], [y - height, y + height]]
    #     draw_likelihood_heatmap(model, solver, params, mle, cov, mle_cov, data,
    #                             sigma2, ranges, args.heatmap_size, p_index=(x_index, y_index),
    #                             subset_indices=indices, output_dir=output_dir,
    #                             filename=f"heatmap_{x_index+1}_{y_index+1}_{int(time_to_remove):d}ms_removed.png",
    #                             title=f"log likelihood heatmap with {time_to_remove:.2f}ms removed")


def generate_synthetic_data(model_class, solver, times, noise, n_repeats=1,
                            parameters=None, rng=None, output_path=None):

    if parameters is None:
        parameters = make_model_of_class(args.model_class).get_default_parameters()
    if times is None:
        times = model.times

    mean = solver(parameters, times=times)

    if rng is None:
        rng = np.random.default_rng()

    if output_path:
        fig = plt.figure(figsize=args.figsize)
        axs = fig.subplots(3)
        axs[0].plot(times, data, label='data')
        axs[0].plot(times, sample_mean, label='mean')
        axs[0].legend()
        states = model.GetStateVariables()
        axs[1].plot(times, states[:, 0] + states[:, 1], label='r')
        axs[1].plot(times, states[:, 2] + states[:, 1], label='a')
        axs[1].legend()
        axs[2].plot(times, voltages)
        fig.savefig(os.path.join(output_dir, "synthetic_data.png"))
        fig.clf()

    return (mean + rng.normal(0, noise, (n_repeats, mean.shape[0])))



if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
