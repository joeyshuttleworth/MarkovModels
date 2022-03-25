#!/usr/bin/env python3

from MarkovModels import common
import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import pints
import pints.plot
from pathos.multiprocessing import ProcessPool as Pool

from MarkovModels.BeattieModel import BeattieModel
from MarkovModels.LinearModel import LinearModel

from numba import njit

import matplotlib.pyplot as plt
import matplotlib as mpl

# Don't use scientific notation offsets on plots (it's confusing)
mpl.rcParams["axes.formatter.useoffset"] = False

sigma2 = 0.01**2


def main():
    plt.style.use('classic')

    parser = common.get_parser(description="Plot various optimality criteria")
    parser.add_argument("-n", "--no_samples", type=int, default=1000)
    parser.add_argument("-N", "--no_chains", type=int, default=4)
    parser.add_argument("-l", "--chain_length", type=int, default=1000)
    parser.add_argument("-b", "--burn-in", type=int, default=None)
    parser.add_argument("-H", "--heatmap_size", type=int, default=0)
    parser.add_argument("-c", "--cpus", type=int, default=1)
    parser.add_argument("-i", "--max_iterations", type=int, default=None)
    parser.add_argument("-s", "--short", help="run with a reduced set of removal times", action='store_true')
    parser.add_argument("-L", "--linear_model", help="Run with a simple linear model\
    instead (debugging)", action='store_true')

    global args
    args = parser.parse_args()

    global optimiser
    optimiser = pints.CMAES

    if args.linear_model:
        optimiser = pints.NelderMead

    # Setup a pool for parallel computation
    pool = Pool(args.cpus)

    output_dir = common.setup_output_directory(args.output, "plot_criteria")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    spike_removal_durations = np.unique(np.concatenate((np.linspace(0, 10, 25),
                                        np.linspace(10, 100, 45),
                                        np.linspace(100, 250, 10))))

    # Write durations to file
    pd.DataFrame(spike_removal_durations[:, None],
                 columns=('removal_duration',)).to_csv(os.path.join(output_dir, "removal_durations.csv"))

    print(spike_removal_durations)

    if args.short:
        spike_removal_durations = np.array([0, 1])

    params = np.array([2.07E-3, 7.17E-2, 3.44E-5, 6.18E-2, 4.18E-1, 2.58E-2,
                       4.75E-2, 2.51E-2, 3.33E-2])

    protocol_func, tstart, tend, tstep, protocol_desc = common.get_ramp_protocol_from_csv('staircase')

    times = np.linspace(tstart, tend, int((tend - tstart) / tstep))

    full_times = times
    Erev = common.calculate_reversal_potential(310.15)
    print(f"Reversal potential is {Erev}")

    if args.linear_model:
        model = LinearModel(times=times, voltage=protocol_func, Erev=Erev, parameters=params)
        model_class = LinearModel
    else:
        model = BeattieModel(times=times, voltage=protocol_func, Erev=Erev, parameters=params)
        model_class = BeattieModel

    model.protocol_description = protocol_desc
    model.window_locs = [t for t, _, _, _ in protocol_desc]

    solver = model.make_hybrid_solver_current()

    voltages = np.array([protocol_func(t) for t in times])

    # Plot representative sample from DGP
    sample_mean = solver()

    noise = np.random.normal(0, np.sqrt(sigma2), times.shape)
    data = sample_mean + noise

    pd.DataFrame(np.column_stack((data, times)),
                 columns=('time', 'current',)).to_csv(
                     os.path.join(output_dir, "synthetic_data.csv"))

    fig = plt.figure(figsize=(20, 18))
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

    D_optimalities = []
    A_optimalities = []

    logging.info("Getting model sensitivities")
    _, S1 = model.SimulateForwardModelSensitivities(params)
    current = solver(params)
    spike_times, spike_indices = common.detect_spikes(times, voltages,
                                                      window_size=0)

    print(f"Spike locations are: {spike_times}")

    covs = []
    indices_used = []

    sample_fig = plt.figure(figsize=(14, 12))
    sample_axs = sample_fig.subplots(2)

    for time_to_remove in spike_removal_durations:
        indices = common.remove_indices(list(range(len(times))),
                                        [(spike,
                                          int(spike + time_to_remove / tstep))
                                         for spike in spike_indices])
        indices_used.append(indices)
        # Plot the observations being removed
        fig.clf()
        axs = fig.subplots(2)
        axs[0].plot(times, voltages)
        for t in spike_times:
            axs[0].axvspan(t, t + time_to_remove, alpha=0.4, color='red', lw=0)

        # Plot the observations under consideration
        axs[1].scatter(times[indices], data[indices], marker='x')
        axs[1].plot(times, solver(params))
        axs[1].set_xlabel('time / ms')
        axs[0].set_ylabel('current / nA')
        axs[1].set_ylabel('current / nA')
        axs[1].set_xlim([0, times[-1]])

        fig.savefig(os.path.join(output_dir, f"spike_removal_{time_to_remove:.0f}.png"))
        for ax in axs:
            ax.cla()

        H = S1[indices, :].T @ S1[indices, :]
        H_inv = np.linalg.inv(H)

        D_optimalities.append(np.linalg.det(H_inv))
        A_optimalities.append(np.trace(H_inv))
        # G_optimalities.append(np.max(np.diag(S1[indices, :] @ H_inv @ S1[indices, :].T)))

        cov = sigma2 * H_inv
        covs.append(cov)

        # SSE = np.sum((solver()[indices] - data[indices])**2)
        # print(f"{time_to_remove:.2f}ms removed: SSE is {SSE}")

    if args.heatmap_size > 0:

        logging.info(f"Drawing {args.heatmap_size} x {args.heatmap_size} likelihood heatmap")

        args_list = [(model_class, times, data, output_dir, time_to_remove, params, indices)
                     for time_to_remove, indices in zip(spike_removal_durations, indices_used)]

        args_list = args_list if args.short else args_list[0:20]
        pool.map(draw_heatmaps, *zip(*args_list))

        logging.info("Finished drawing heatmaps")

    for time_to_remove, cov in list(zip(spike_removal_durations, covs)):
        plot_sample_trajectories(solver, full_times, voltages, time_to_remove, params, cov, sample_axs,
                                 args.no_samples, spike_indices)
        sample_fig.savefig(os.path.join(output_dir, f"sample_trajectories_{time_to_remove:.2f}.png"))
        for ax in sample_axs:
            ax.cla()

    D_optimalities = np.array(D_optimalities)
    A_optimalities = np.array(A_optimalities)

    # Normalise with respect to first score i.e 0 ms removed
    D_optimalities = D_optimalities / D_optimalities[0]
    A_optimalities = A_optimalities / A_optimalities[0]

    df = pd.DataFrame(np.column_stack((spike_removal_durations,
                                       # Bayesian_D_optimalities,
                                       np.log(D_optimalities),
                                       np.log(A_optimalities))),
                      columns=('time removed after spikes /ms',
                               "normalised log D-optimality",
                               "normalised log A-optimality"))

    df.set_index('time removed after spikes /ms', inplace=True)

    fig = plt.figure(figsize=(16, 12))
    ax = fig.subplots()
    df.plot(legend=True, subplots=True, ax=ax)

    print("plotting criteria")
    fig.savefig(os.path.join(output_dir, "criteria.pdf"))

    # Now plot it zoomed in on the first 25ms
    ax.set_xlim([0, 25])
    fig.savefig(os.path.join(output_dir, "criteria_zoomed_in.pdf"))

    conf_fig = plt.figure(figsize=(16, 12))
    conf_axs = conf_fig.subplots(2)

    labels = [f"{duration:.2f}ms removed" for duration in spike_removal_durations[::4]]
    plot_regions(covs[::4], labels, params, output_dir,
                 spike_removal_durations, conf_fig, conf_axs, sigma2, p_of_interest=(4, 6))
    plot_regions(covs[::4], labels, params, output_dir,
                 spike_removal_durations, conf_fig, conf_axs, sigma2, (5, 7))

    plot_regions(covs[::4], labels, params, output_dir,
                 spike_removal_durations, conf_fig, conf_axs, sigma2, (4, 5))

    plot_regions(covs[::4], labels, params, output_dir,
                 spike_removal_durations, conf_fig, conf_axs, sigma2, (6, 7))

    fig = plt.figure(figsize=(18, 14))
    axs = fig.subplots(3)

    param_fig = plt.figure(figsize=(18, 14))
    param_axs = param_fig.subplots(model.get_no_parameters())

    std_fig = plt.figure(figsize=(22, 20))
    std_axs = std_fig.subplots(5)

    print("Start MCMC sampling")
    args_list = [(model_class, "staircase", times, data, params, index_set) for index_set in indices_used]
    mcmc_samples = pool.map(mcmc_chain_func, *zip(*args_list))

    rhats = np.array([rhat for _, rhat in mcmc_samples])
    mcmc_samples = np.stack([chain for chain, _ in mcmc_samples])

    np.save(os.path.join(output_dir, "mcmc_samples_all.npy"), mcmc_samples)

    # output rhats
    pd.DataFrame(np.column_stack((spike_removal_durations, rhats)),
                 columns=['removal_duration'] + ['rhat_%i' % i
                                                 for i in range(rhats.shape[1])]).to_csv(os.path.join(output_dir, "rhats.csv"))

    # Now, for each mcmc run plot some sample trajectories
    traj_fig = plt.figure(figsize=(18, 12))
    traj_ax = traj_fig.subplots()
    indices = np.unique(np.array(list(range(len(times)))[::50] + list(spike_indices)))

    # Sample with replacement from the chain.
    no_subsamples = 5000
    for j, mcmc_sample in enumerate(mcmc_samples):
        for sample in np.random.choice(mcmc_sample.shape[1], no_subsamples):
            trajectory = solver(mcmc_sample[0, sample, :], full_times)
            traj_ax.plot(times, trajectory, color='grey', alpha=.3)
        traj_ax.set_title(f"{no_subsamples} MCMC sampled trajectories {spike_removal_durations[j]:.2f}ms removed")
        traj_ax.plot(times, current, color='red')
        traj_ax.set_ylim([1.5 * np.min(current), 1.5 * np.max(current)])
        traj_fig.savefig(os.path.join(output_dir, f"{j}_mcmc_trajectories.png"))
        traj_ax.cla()

    voltage_list = [-120, -80, -60, -50, -40, -30, -20, -10, 0, 10, 20, 40]

    for i, samples in enumerate(mcmc_samples):
        for j in range(samples.shape[0]):
            df = pd.DataFrame(samples[j], columns=model.parameter_labels)
            df.to_csv(os.path.join(output_dir, f"mcmc_samples_[{i}]_chain_{j}.csv"))

        for j, p in [(j, "p%i" % (j + 1)) for j in range(model.get_no_parameters())]:
            for row in samples:
                try:
                    sns.kdeplot(data=pd.DataFrame(row[:, j], columns=[p]),
                                shade=False, ax=param_axs[j], common_norm=True,
                                color='blue')
                except Exception as e:
                    print(str(e))
        param_axs[0].set_title(f"{spike_removal_durations[i]:.2f}ms removed after each spike")

        for j, ax in enumerate(param_axs):
            ax.axvline(params[j], color='grey', linestyle='--')

        param_fig.tight_layout()
        param_fig.savefig(os.path.join(output_dir, f"mcmc_params_{i}.png"))

        for ax in param_axs:
            ax.cla()

        # Concatenate chains together using Fortran ordering i.e first index moves fastest
        samples = samples.reshape(samples.shape[0]*samples.shape[1], -1, order='F')

        if i < 20:
            try:
                pairwise_fig, pairwise_ax = pints.plot.pairwise(samples, kde=True,
                                                            parameter_names=['p%i' % i for i in range(1, 9)]
                                                            + ['g_kr'])
            except np.linalg.LinAlgError as ex:
                print("failed to produce pairwise plot")
                print(str(ex))

                pairwise_fig.tight_layout()
                pairwise_fig.savefig(os.path.join(output_dir,
                                                  f"pairwise_plot_{spike_removal_durations[i]:.2f}ms_removed.png"))
                pairwise_fig.clf()

    fig.clf()
    axs = fig.subplots(5)
    for voltage in voltage_list:
        steady_state_samples = []
        voi_samples = []
        sub_output_dir = os.path.join(output_dir, f"{voltage}mV")
        if not os.path.exists(sub_output_dir):
            os.makedirs(sub_output_dir)
        for i, cov, in enumerate(covs):
            # Normal approximation first
            a_inf, tau_a, r_inf, tau_r, gkr = monte_carlo_tau_inf(
                params, cov, voltage=voltage, n_samples=args.no_samples)

            voi_samples.append([a_inf, tau_a, r_inf, tau_r])

            I_Kr_inf = (gkr*r_inf*a_inf*(voltage - Erev)).flatten()
            steady_state_samples.append(I_Kr_inf)
            colnames = ['a_inf', 'tau_a', 'r_inf', 'tau_r', 'I_Kr_inf']

            vals_df = pd.DataFrame(data=np.column_stack((a_inf, tau_a, r_inf, tau_r, I_Kr_inf)), columns=colnames)
            try:
                axs[0].set_title(
                    f"{voltage}mV with {spike_removal_durations[i]:.2f}ms removed")
                for k, var in enumerate(colnames):
                    sns.kdeplot(data=pd.DataFrame(vals_df, columns=[var]),
                                shade=True, fill=True, ax=axs[k], x=var,
                                common_norm=True,
                                color='blue')

                fig.tight_layout()
                fig.savefig(os.path.join(sub_output_dir, f"{i}.png"))
            except Exception as e:
                print(str(e))

            for ax in axs:
                ax.cla()

            samples = mcmc_samples[i]
            res = compute_tau_inf_from_samples(samples, voltage=voltage)
            for j, (a_inf, tau_a, r_inf, tau_r) in enumerate(zip(*res)):
                gkrs = samples[j, :, -1]
                I_Kr_inf = (gkrs*r_inf*a_inf*(voltage - Erev)).flatten()
                colnames = ('a_inf', 'tau_a', 'r_inf', 'tau_r', 'I_Kr_inf')
                vals_df = pd.DataFrame(np.column_stack((a_inf, tau_a, r_inf, tau_r, I_Kr_inf.T)), columns=colnames)

                axs[0].set_title(f"{voltage:.2f}mV with {spike_removal_durations[i]:.2f}ms removed after each spike")
                try:
                    axs[0].set_title(
                        f"{voltage}mV with {spike_removal_durations[i]:.2f}ms removed (MCMC)")
                    for k, var in enumerate(colnames):
                        sns.kdeplot(data=pd.DataFrame(vals_df, columns=[var]),
                                    shade=False, ax=axs[k], x=var, common_norm=True,
                                    color='blue')
                    # Plot true values
                    for k, val in enumerate(compute_tau_inf_from_params(params, voltage=voltage)):
                        axs[k].axvline(val, color='grey', linestyle='--')

                except Exception as e:
                    print(str(e))

            fig.tight_layout()
            fig.savefig(os.path.join(sub_output_dir, f"mcmc_{i}.png"))
            for ax in axs:
                ax.cla()

        # Use the same axes for the Monte Carlo estimations of these variables
        steady_states_df = pd.DataFrame(columns=('IKr', 'a_inf', 'tau_a', 'r_inf',
                                                 'tau_r', 'removal_duration'))
        for i in range(len(steady_state_samples)):
            sample = np.column_stack((steady_state_samples[i].T, *voi_samples[i])).T
            df = pd.DataFrame(sample.T, columns=('IKr', 'a_inf', 'tau_a', 'r_inf', 'tau_r'))
            df['removal_duration'] = round(spike_removal_durations[i], 2)
            steady_states_df = steady_states_df.append(df, ignore_index=True)

        print("shape", steady_states_df.values.shape)
        print(f"dataframe is {steady_states_df}")

        for i, var in enumerate(('IKr', 'a_inf', 'tau_a', 'r_inf', 'tau_r')):
            try:
                df = df[np.isfinite(df).all(1)]
                sns.kdeplot(data=steady_states_df, x=var, ax=axs[i],
                            shade=False, common_norm=True,
                            hue='removal_duration', palette='viridis',
                            legend=(i == 0), color='blue')

            except Exception as ex:
                print(f"Failed to plot densities {str(ex)}")

        # Plot true values
        for k, val in enumerate(compute_tau_inf_from_params(params, voltage=voltage)):
            axs[k].axvline(val, color='grey', linestyle='--')

        fig.savefig(os.path.join(output_dir, f"steady_state_prediction_comparison_{voltage}mV.png"))
        for ax in axs:
            ax.cla()

        stds = [np.std(sample) for sample in steady_state_samples]

        std_axs[0].plot(spike_removal_durations, stds)
        std_axs[-1].set_xlabel('time removed after each spike /ms')
        std_axs[0].set_ylabel('std log error in IKr')
        std_axs[0].set_title(f"{voltage}mV")

        for i, var in enumerate(('a_inf', 'tau_a', 'r_inf', 'tau_r')):
            stds = [np.std(steady_states_df[steady_states_df['removal_duration'] == time_removed][var].values)
                    for time_removed in np.unique(steady_states_df['removal_duration'])]
            # print(stds)
            std_axs[i + 1].plot(spike_removal_durations, np.log(stds))
            std_axs[i + 1].set_ylabel(f"log std of {var} estimate")

        std_fig.savefig(os.path.join(sub_output_dir, 'standard_deviations'))

        for ax in std_axs:
            ax.cla()


def plot_sample_trajectories(solver, times, voltages, removal_duration, params, cov, axs, n_samples, spike_indices):

    mean_param_trajectory = solver(params)
    # axs[0].plot(times, mean_param_trajectory, 'red')
    axs[0].set_ylim(np.min(mean_param_trajectory) * 1.5, np.max(mean_param_trajectory) * 1.5)

    for spike in spike_indices:
        axs[1].axvspan(times[spike], times[spike] + removal_duration, alpha=0.2, color='red', lw=0)

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
        axs[0].plot(times[indices], trajectory[indices], color='grey', alpha=0.3)

    print(f"{removal_duration:.2f}: Successfully ran {count} out of {n_samples} simulations")

    axs[1].plot(times, voltages)
    axs[1].set_xlabel("time /ms")
    axs[1].set_ylabel("membrane voltage /mV")


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
                 fig, axs, sigma2, p_of_interest=(4, 6)):
    offset = [params[p_of_interest[0]], params[p_of_interest[1]]]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    cov = covs[0][p_of_interest, :]
    cov = cov[:, p_of_interest]
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Plot smallest region on top plot
    common.cov_ellipse(cov, nsig=1,
                       ax=axs[0],
                       resize_axes=True,
                       color=colors[0 % len(colors)],
                       offset=offset,
                       label=labels[0])

    eigvals, eigvecs = np.linalg.eigh(cov)

    # Plot confidence regions starting with the largest (most observations
    # removed)
    first_rotation = np.arctan2(*eigvecs[::-1, 0])

    for i, cov in reversed(list(enumerate(covs[0:10]))):
        sub_cov = cov[p_of_interest, :]
        sub_cov = sub_cov[:, p_of_interest]
        eigvals, eigvecs = np.linalg.eigh(sub_cov)

        rotation = np.arctan2(*eigvecs[::-1, 0])

        common.cov_ellipse(sub_cov, q=[0.95],
                           ax=axs[1],
                           offset=offset,
                           color=colors[i % len(colors)],
                           # rotate=rotation - first_rotation,
                           # resize_axes=(i == len(covs)-1),
                           label=labels[i])

    axs[0].set_title(f"95% confidence regions after spike removal")
    axs[0].plot(*offset, 'x', color='red',
                label=f"p{p_of_interest[0]+1} = {offset[0]}, p{p_of_interest[1]+1} = {offset[1]}")
    axs[1].plot(*offset, 'x', color='red',
                label=f"p{p_of_interest[0]+1} = {offset[0]}, p{p_of_interest[1]+1} = {offset[1]}")
    axs[0].set_xlabel(f"p{p_of_interest[0]+1} / ms^-1")
    axs[0].set_ylabel(f"p{p_of_interest[1]+1} / ms^-1")
    # axs[1].xaxis.set_ticks([])
    # axs[1].yaxis.set_ticks([])
    # axs[1].set_xlabel('rotated and scaled view')

    axs[1].legend()
    fig.savefig(os.path.join(output_dir,
                             f"p{p_of_interest[0]+1} and p{p_of_interest[1]+1} confidence regions.png"))
    for ax in axs:
        ax.cla()


def get_mcmc_chains(solver, times, indices, data, chain_length, starting_parameters, sigma2, burn_in=None):
    n = len(indices)
    print(f"number of timesteps is {n}")

    if burn_in is None:
        burn_in = int(chain_length / 10)

    @njit
    def log_likelihood_func(p):
        output = solver(p, times)[indices]
        error = output - data[indices]
        SSE = np.sum(error**2)
        ll = -n * 0.5 * np.log(2 * np.pi * sigma2) - SSE / (2 * sigma2)
        return ll

    class pints_likelihood(pints.LogPDF):
        def __call__(self, p):
            return log_likelihood_func(p)

        def n_parameters(self):
            return len(starting_parameters)

    prior = pints.UniformLogPrior([0] * pints_likelihood().n_parameters(),
                                  [1] * pints_likelihood().n_parameters())

    posterior = pints.LogPosterior(pints_likelihood(), prior)

    mcmc = pints.MCMCController(posterior, args.no_chains,
                                np.tile(starting_parameters, [args.no_chains, 1]),
                                method=pints.HaarioBardenetACMC)

    mcmc.set_max_iterations(args.chain_length)

    samples = mcmc.run()
    return samples[:, burn_in:, :]


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

    fig = plt.figure(figsize=(18, 14))
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
    common.cov_ellipse(subcov, offset=(params[x_index], params[y_index]), q=[0.95], ax=ax,
                       color='red', label='Approximated sampling distribution of MLE (95%)')
    ax.plot(params[x_index], params[y_index], marker='x', color='red', linestyle='None', label='true_params')

    ax.plot(mle[p_index[0]], mle[p_index[1]], marker='o', linestyle='None', color='pink', label='mle')

    # Draw normal approximation of 95% confidence region (marginal)
    subcov = mle_cov[(x_index, y_index), :][:, (x_index, y_index)]
    common.cov_ellipse(subcov, offset=(mle[x_index], mle[y_index]), q=[0.95], ax=ax,
                       color='pink', label='95% confidence region (normal approximation)')
    # Draw 2 param versions
    # Use NelderMead for 2d problem
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
        common.cov_ellipse(mle_2param_cov, offset=mle_2param, q=[0.95], ax=ax,
                           color='purple', label='Conditional 95% confidence region (normal approximation)')
    except np.linalg.LinAlgError:
        print("Failed to invert Hessian matrix")
        print(S1)

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


# Next, the MCMC version. Can be time consuming so perform this in parallel
def mcmc_chain_func(model_class, protocol, times, data, params, index_set):
    protocol_func, tstart, tend, tstep, protocol_desc = common.get_ramp_protocol_from_csv('staircase')

    times = np.linspace(tstart, tend, int((tend - tstart) / tstep))
    Erev = common.calculate_reversal_potential(310.15)

    model = model_class(times=times, voltage=protocol_func, Erev=Erev, parameters=params)
    model.protocol_description = protocol_desc
    model.window_locs = [t for t, _, _, _ in protocol_desc]

    chains = get_mcmc_chains(model.make_hybrid_solver_current(), times, index_set, data, args.chain_length, params,
                             sigma2, burn_in=args.burn_in)
    rhat = pints.rhat(chains)
    return chains, rhat


def draw_heatmaps(model_class, times, data, output_dir, time_to_remove, params, indices=None):

    if indices is None:
        indices = list(range(len(times)))

    print(f"using {len(indices)} indices")

    protocol_func, tstart, tend, tstep, protocol_desc = common.get_ramp_protocol_from_csv('staircase')

    Erev = common.calculate_reversal_potential(310.15)
    model = model_class(times=times, voltage=protocol_func, Erev=Erev, parameters=params)
    model.window_locs = [t for t, _, _, _ in protocol_desc]
    model.protocol_description = protocol_desc
    solver = model.make_hybrid_solver_current()

    S1 = model.SimulateForwardModelSensitivities(params)[1][indices]
    cov = sigma2 * np.linalg.inv(S1.T @ S1)

    mle, _ = common.fit_model(model, data, params, subset_indices=indices,
                              solver=solver, max_iterations=args.max_iterations,
                              method=optimiser)

    S1_tmp = model.SimulateForwardModelSensitivities(mle)[1][indices]
    mle_cov = sigma2 * np.linalg.inv(S1_tmp.T @ S1_tmp)

    for x_index, y_index in [(4, 6), (5, 7), (4, 7)]:
        width = np.sqrt(cov[x_index, x_index]) * 3
        height = np.sqrt(cov[y_index, y_index]) * 3

        x = params[x_index]
        y = params[y_index]

        ranges = [[x - width, x + width], [y - height, y + height]]
        draw_likelihood_heatmap(model, solver, params, mle, cov, mle_cov, data,
                                sigma2, ranges, args.heatmap_size, p_index=(x_index, y_index),
                                subset_indices=indices, output_dir=output_dir,
                                filename=f"heatmap_{x_index+1}_{y_index+1}_{int(time_to_remove):d}ms_removed.png",
                                title=f"log likelihood heatmap with {time_to_remove:.2f}ms removed")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
