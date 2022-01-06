#!/usr/bin/env python3

from MarkovModels import common
import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import pints

from MarkovModels.BeattieModel import BeattieModel

from numba import njit

import matplotlib.pyplot as plt

sigma2 = 0.0001


def main():

    plt.style.use('classic')
    parser = common.get_parser(description="Plot various optimality criteria")
    parser.add_argument("-n", "--no_samples", type=int, default=1000)
    parser.add_argument("-N", "--no_chains", type=int, default=8)
    parser.add_argument("-l", "--chain_length", type=int, default=1000)
    parser.add_argument("-b", "--burn-in", type=int, default=None)

    global args
    args = parser.parse_args()
    output_dir = os.path.join(args.output, "plot_criteria")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    spike_removal_durations = np.linspace(0, 50, 10)

    params = np.array([2.07E-3, 7.17E-2, 3.44E-5, 6.18E-2, 4.18E-1, 2.58E-2,
                       4.75E-2, 2.51E-2, 3.33E-2])

    protocol_func, tstart, tend, tstep, protocol_desc = common.get_ramp_protocol_from_csv('staircase')
    times = np.linspace(tstart, tend, int((tend - tstart) / tstep))
    Erev = common.calculate_reversal_potential(310.15)
    model = BeattieModel(times=times,
                         voltage=protocol_func,
                         Erev=Erev,
                         parameters=params)

    model.protocol_description = protocol_desc
    model.window_locs = [t for t, _, _, _ in protocol_desc]

    voltages = model.GetVoltage()

    D_optimalities = []
    A_optimalities = []
    G_optimalities = []

    logging.info("Getting model sensitivities")
    current, S1 = model.SimulateForwardModelSensitivities(params)
    spike_times, spike_indices = common.detect_spikes(times, voltages,
                                                      window_size=500)

    current_spikes, _ = common.detect_spikes(times, current, threshold=max(current) / 100,
                                             window_size=100)
    print(f"spike locations are{current_spikes}")

    covs = []
    indices_used = []

    # Setup for Bayesian D optimality
    nb_samples = 10

    for time_to_remove in spike_removal_durations:
        indices = common.remove_indices(list(range(len(times))),
                                        [(spike,
                                          int(spike + time_to_remove / tstep))
                                         for spike in spike_indices])
        indices_used.append(np.unique(indices))

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

    df = pd.DataFrame(np.column_stack((spike_removal_durations,
                                       # Bayesian_D_optimalities,
                                       np.log(D_optimalities),
                                       np.log(A_optimalities),
                                       np.log(A_optimalities))),
                      columns=('time removed after spikes /ms',
                               # "Bayesian D-optimality",
                               "normalised log D-optimality",
                               "normalised log A-optimality",
                               "normalised log G-optimality"))

    df.set_index('time removed after spikes /ms', inplace=True)

    df.plot(legend=True, subplots=True)

    fig = plt.gcf()
    fig.savefig(os.path.join(output_dir, "criteria.pdf"))

    conf_fig = plt.figure(figsize=(16, 12))
    conf_axs = conf_fig.subplots(2)

    plot_regions(times, model, spike_times, spike_indices, output_dir,
                 spike_removal_durations, conf_fig, conf_axs, p_of_interest=(4, 6))
    plot_regions(times, model, spike_times, spike_indices, output_dir,
                 spike_removal_durations, conf_fig, conf_axs, (5, 7))

    plot_regions(times, model, spike_times, spike_indices, output_dir,
                 spike_removal_durations, conf_fig, conf_axs, (4, 5))

    plot_regions(times, model, spike_times, spike_indices, output_dir,
                 spike_removal_durations, conf_fig, conf_axs, (6, 7))

    fig = plt.figure(figsize=(18, 14))
    axs = fig.subplots(4)

    # Sample steady states and timescales
    print("Sampling steady states and timescales")
    param_fig = plt.figure(figsize=(18, 14))
    param_axs = param_fig.subplots(model.get_no_parameters())

    forward_solver = model.make_hybrid_solver_current()
    data = forward_solver(model.get_default_parameters(), times, voltages) + \
        np.random.normal(0, np.sqrt(sigma2), (len(times),))

    # Next, the MCMC version
    mcmc_samples = [get_mcmc_chains(forward_solver, times, voltages,
                                   index_set, data, args.chain_length,
                                   model.get_default_parameters(), burn_in=args.burn_in) for index_set in indices_used]
    voltage_list = [-100, -80, -40, -30, -20, -10, 0, 10, 20, 30, 40]

    for voltage in voltage_list:
        steady_state_samples = []
        sub_output_dir = os.path.join(output_dir, f"{voltage}mV")
        if not os.path.exists(sub_output_dir):
            os.makedirs(sub_output_dir)
        for i, cov, in enumerate(covs):
            samples = mcmc_samples[i]

            # Normal approximation first
            a_inf, tau_a, r_inf, tau_r, gkr = monte_carlo_tau_inf(
                params, cov, voltage=voltage, n_samples=args.no_samples)

            I_Kr_inf = (gkr*r_inf*a_inf*(voltage - Erev)).flatten()
            steady_state_samples.append(I_Kr_inf)

            try:
                # axs[0].scatter(a_inf, tau_a, marker='x')
                sns.kdeplot(data=pd.DataFrame(zip(a_inf, tau_a), columns=[
                            'a_inf', 'tau_a']), shade=True, fill=True, ax=axs[0], x='a_inf', y='tau_a', common_norm=True)
                axs[0].set_title(
                    f"{voltage}mV with {spike_removal_durations[i]:.2f}ms removed")
                # axs[1].scatter(r_inf, tau_r, marker='x')
                sns.kdeplot(data=pd.DataFrame(zip(r_inf, tau_r), columns=[
                            'r_inf', 'tau_r']), shade=True, ax=axs[1], x='r_inf', y='tau_r', common_norm=True)
                # r_inf vs a_inf
                sns.kdeplot(data=pd.DataFrame(zip(r_inf, a_inf), columns=[
                    'r_inf', 'a_inf']), shade=True, ax=axs[2], x='r_inf', y='a_inf', common_norm=True)
                sns.kdeplot(data=pd.DataFrame(I_Kr_inf, columns=[
                    'I_Kr_inf']), shade=True, ax=axs[3], common_norm=True)

                axs[0].set_title(f"{voltage:.2f}mV with {spike_removal_durations[i]}ms removed after each spike")
                fig.savefig(os.path.join(sub_output_dir, f"{i}.png"))
            except Exception as e:
                print(str(e))

            for ax in axs:
                ax.cla()

            fig = plt.figure(figsize=(18, 14))
            axs = fig.subplots(4)

            res = compute_tau_inf_from_samples(samples, voltage=voltage)
            for j, (a_inf, tau_a, r_inf, tau_r) in enumerate(zip(*res)):
                gkrs = samples[j, :, -1]
                I_Kr_inf = (gkrs*r_inf*a_inf*(voltage - Erev)).flatten()

                axs[0].set_title(f"{voltage:.2f}mV with {spike_removal_durations[i]:.2f}ms removed after each spike")
                try:
                    # axs[0].scatter(a_inf, tau_a, marker='x')
                    sns.kdeplot(data=pd.DataFrame(zip(a_inf, tau_a), columns=[
                        'a_inf', 'tau_a']), shade=True, fill=True, ax=axs[0], x='a_inf', y='tau_a')
                    axs[0].set_title(
                        f"+40mV with {spike_removal_durations[i]:.2f}ms removed (MCMC)")
                    sns.kdeplot(data=pd.DataFrame(zip(r_inf, tau_r), columns=[
                        'r_inf', 'tau_r']), shade=True, ax=axs[1], x='r_inf', y='tau_r')
                    # r_inf vs a_inf
                    sns.kdeplot(data=pd.DataFrame(zip(r_inf, a_inf), columns=[
                        'r_inf', 'a_inf']), shade=True, ax=axs[2], x='r_inf', y='a_inf')
                    sns.kdeplot(data=pd.DataFrame(I_Kr_inf, columns=[
                        'I_Kr_inf']), shade=True, ax=axs[3])
                except Exception as e:
                    print(str(e))

            fig.savefig(os.path.join(sub_output_dir, f"mcmc_{i}.png"))
            for ax in axs:
                ax.cla()

            for j in range(samples.shape[0]):
                df = pd.DataFrame(samples[j], columns=model.parameter_labels)
                df.to_csv(os.path.join(sub_output_dir, f"mcmc_samples[i]_{i}_chain_{j}.csv"))

            for j, p in [(j, "p%i" % (j + 1)) for j in range(model.get_no_parameters())]:
                for row in samples:
                    try:
                        sns.kdeplot(data=pd.DataFrame(row[:, j], columns=[p]), shade=True, ax=param_axs[j])
                    except Exception as e:
                        print(str(e))
            param_axs[0].set_title(f"{voltage}mV with {spike_removal_durations[i]:.2f}ms removed after each spike")
            param_fig.savefig(os.path.join(sub_output_dir, f"mcmc_params_{i}.png"))

            for ax in param_axs:
                ax.cla()

        # Plot steady states on one axis for comparison
        fig = plt.figure(figsize=(16, 14))
        ax = fig.subplots()

        steady_states_df = pd.DataFrame(columns=('IKr', 'removal_duration'))
        for i, sample in enumerate(steady_state_samples):
            df = pd.DataFrame(sample.T, columns=('IKr',))
            df['removal_duration'] = spike_removal_durations[i]
            steady_states_df = steady_states_df.append(df, ignore_index=True)

        print("shape", steady_states_df.values.shape)

        print(f"dataframe is {steady_states_df}")

        sns.kdeplot(data=steady_states_df, x='IKr', shade=False, ax=ax, common_norm=True,
                    hue='removal_duration', palette='viridis')

        plot_x_lims = np.quantile(steady_state_samples[-1], (.05, .95))
        x_window_size = plot_x_lims[1] - plot_x_lims[0]

        plot_x_lims = np.mean(steady_state_samples[0]) + np.array([-x_window_size, x_window_size]) * .5

        ax.set_xlim(*plot_x_lims)

        fig.savefig(os.path.join(output_dir, f"steady_state_prediction_comparison_{voltage}mV.png"))
        ax.cla()

    # Now plot predictions
    # We can use less timesteps now -- only interested in plotting
    pred_times = times
    pred_voltages = np.array([model.voltage(t) for t in pred_times])
    pred_times = np.unique(list(pred_times) + list(current_spikes))

    n_samples = args.no_samples

    forward_solve = model.make_hybrid_solver_current()

    def get_trajectory(p):
        try:
            soln = forward_solve(p, times=pred_times, voltages=pred_voltages)
        except Exception:
            return np.full(pred_times.shape, np.nan)

        if np.all(np.isfinite(soln)):
            return soln
        else:
            return np.full(pred_times.shape, np.nan)

    fig = plt.figure(figsize=(14, 12))
    axs = fig.subplots(2)

    mean_param_trajectory = model.SimulateForwardModel()

    for i, cov in list(enumerate(covs)):
        samples = np.random.multivariate_normal(params, cov, n_samples)

        # Filter out invalid samples
        samples = [s for s in samples if np.all(s) > 0]

        mean_estimate_uncertainty = np.apply_along_axis(lambda row:
                                                        row @ cov @ row.T, 1,
                                                        S1)

        upper_bound = mean_param_trajectory + 1.96 * mean_estimate_uncertainty
        lower_bound = mean_param_trajectory - 1.96 * mean_estimate_uncertainty

        axs[0].fill_between(times, lower_bound, upper_bound, color='blue',
                            alpha=0.25)
        axs[0].plot(times, mean_param_trajectory, 'red')
        axs[0].set_ylim(np.min(current) * 1.5, np.max(current) * 1.5)

        count = 0
        for sample in samples:
            trajectory = get_trajectory(sample)
            if trajectory is not None:
                count += 1
                axs[0].plot(pred_times, trajectory, color='grey', alpha=0.1)

        print(f"{i}: Successfully ran {count} out of {n_samples} simulations")

        axs[1].plot(times, model.GetVoltage())
        axs[1].set_xlabel("time /ms")
        axs[1].set_ylabel("membrane voltage /mV")

        fig.savefig(os.path.join(output_dir, "{:.2f}ms_sample_trajectories.png".format(
            spike_removal_durations[i])))

        for ax in axs:
            ax.cla()


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


def plot_regions(times, model, spike_times, spike_indices, output_dir, spike_removal_durations,
                 fig, axs, p_of_interest=(4, 6)):
    params = model.get_default_parameters()
    tstep = times[1] - times[0]

    offset = [params[p_of_interest[0]], params[p_of_interest[1]]]

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    covs = []

    current, S1 = model.SimulateForwardModelSensitivities()

    for time_to_remove in spike_removal_durations:
        indices = common.remove_indices(list(range(len(times))),
                                        [(spike,
                                          int(spike + time_to_remove / tstep))
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
    first_rotation = np.arctan2(*eigvecs[::-1, 0])

    for i, cov in reversed(list(enumerate(covs[0:5]))):
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
                           label_arg="{:.2f}ms".format(
                               spike_removal_durations[i % len(spike_removal_durations)]))

    axs[0].set_title(f"95% confidence regions after spike removal")
    axs[0].plot(*offset, 'x', color='red',
                label=f"p{p_of_interest[0]+1} = {offset[0]}, p{p_of_interest[1]+1} = {offset[1]}")
    axs[1].plot(*offset, 'x', color='red',
                label=f"p{p_of_interest[0]+1} = {offset[0]}, p{p_of_interest[1]+1} = {offset[1]}")
    axs[0].set_xlabel(f"p{p_of_interest[0]+1} / ms^-1")
    axs[0].set_ylabel(f"p{p_of_interest[1]+1} / ms^-1")
    # axs[1].xaxis.set_ticks([])
    # axs[1].yaxis.set_ticks([])
    axs[1].set_xlabel('rotated and scaled view')

    axs[1].legend()
    fig.savefig(os.path.join(output_dir,
                             f"p{p_of_interest[0]+1} and p{p_of_interest[1]+1} rotated confidence regions.pdf"))
    for ax in axs:
        ax.cla()


def get_mcmc_chains(solver, times, voltages, indices, data, chain_length, default_parameters, burn_in=None):
    # Do the same as above but using mcmc on synthetic data
    print('doing mcmc')
    times = times
    voltages = voltages
    data = data[indices]

    if burn_in is None:
        burn_in = int(chain_length / 10)

    @njit
    def log_likelihood_func(p):
        sol = solver(p, times, voltages)[indices]
        return -0. * np.sum((sol - data)**2 / sigma2) - np.log(np.sqrt(sigma2 * 2 * np.pi))

    class pints_likelihood(pints.LogPDF):
        def __call__(self, p):
            return log_likelihood_func(p)

        def n_parameters(self):
            return len(default_parameters)

    prior = pints.UniformLogPrior([0] * pints_likelihood().n_parameters(),
                                  [1] * pints_likelihood().n_parameters())

    posterior = pints.LogPosterior(pints_likelihood(), prior)

    mcmc = pints.MCMCController(posterior, args.no_chains,
                                [default_parameters] * args.no_chains,
                                method=pints.HaarioBardenetACMC)

    mcmc.set_max_iterations(args.chain_length)

    samples = mcmc.run()
    return samples[:, burn_in:, :]


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


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
