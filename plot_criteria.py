#!/usr/bin/env python3

from MarkovModels import common
import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import pints
import uuid

from MarkovModels.BeattieModel import BeattieModel

from numba import njit

import matplotlib.pyplot as plt
import matplotlib as mpl

# Don't use scientific notation offsets on plots (it's confusing)
mpl.rcParams["axes.formatter.useoffset"] = False

sigma2 = 0.005**2


def main():
    plt.style.use('classic')
    parser = common.get_parser(description="Plot various optimality criteria")
    parser.add_argument("-n", "--no_samples", type=int, default=1000)
    parser.add_argument("-N", "--no_chains", type=int, default=8)
    parser.add_argument("-l", "--chain_length", type=int, default=1000)
    parser.add_argument("-b", "--burn-in", type=int, default=None)
    parser.add_argument("-H", "--heatmap_size", type=int, default=0)

    global args
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join('output', f"output_{uuid.uuid4()}")

    output_dir = os.path.join(args.output, "plot_criteria")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    spike_removal_durations = np.linspace(0, 149, 20)

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

    # Plot representative sample from DGP
    sample_mean = model.make_hybrid_solver_current(njitted=False)(params, times)
    noise = np.random.normal(0, np.sqrt(sigma2), times.shape)
    print(f"Sample std of noise is {noise.std()}")
    data = sample_mean + noise
    print(f"Sample std of noise is {np.std(data - sample_mean)}")

    plt.plot(times, data, label='data')
    plt.plot(times, sample_mean, color='red', label='mean')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "sample_of_DGP"))

    solver = model.make_hybrid_solver_current()
    # Plot heatmaps
    if args.heatmap_size > 0:
        logging.info(f"Drawing {args.heatmap_size} x {args.heatmap_size} likelihood heatmap")
        ranges = [[0.001, 0.6], [0.04, 0.065]]
        draw_likelihood_heatmap(model, solver, params, times, data,
                                sigma2, ranges, args.heatmap_size,
                                (4, 6), output_dir)
        ranges = [[0.01, 0.06], [0.01, 0.04]]
        draw_likelihood_heatmap(model, solver, params, times, data,
                                sigma2, ranges, args.heatmap_size,
                                (5, 7), output_dir)
        logging.info("Finished drawing heatmap")

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

    std_fig = plt.figure(figsize=(22, 20))
    std_axs = std_fig.subplots(5)

    forward_solver = model.make_hybrid_solver_current()
    # Next, the MCMC version
    mcmc_samples = [get_mcmc_chains(forward_solver, times, voltages,
                                    index_set, data, args.chain_length,
                                    model.get_default_parameters(), burn_in=args.burn_in) for index_set in indices_used]
    voltage_list = [-100, -80, -40, -30, -20, -10, 0, 10, 20, 30, 40]

    for voltage in voltage_list:
        steady_state_samples = []
        voi_samples = []
        sub_output_dir = os.path.join(output_dir, f"{voltage}mV")
        if not os.path.exists(sub_output_dir):
            os.makedirs(sub_output_dir)
        for i, cov, in enumerate(covs):
            samples = mcmc_samples[i]

            # Normal approximation first
            a_inf, tau_a, r_inf, tau_r, gkr = monte_carlo_tau_inf(
                params, cov, voltage=voltage, n_samples=args.no_samples)

            voi_samples.append([a_inf, tau_a, r_inf, tau_r])

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

                axs[0].set_title(f"{voltage}mV with {spike_removal_durations[i]:.2f}ms removed after each spike")
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
                        f"{voltage} with {spike_removal_durations[i]:.2f}ms removed (MCMC)")
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
                df.to_csv(os.path.join(sub_output_dir, f"mcmc_samples_[{i}]_chain_{j}.csv"))

            for j, p in [(j, "p%i" % (j + 1)) for j in range(model.get_no_parameters())]:
                for row in samples:
                    try:
                        sns.kdeplot(data=pd.DataFrame(row[:, j], columns=[p]), shade=False, ax=param_axs[j])
                    except Exception as e:
                        print(str(e))
            param_axs[0].set_title(f"{voltage}mV with {spike_removal_durations[i]:.2f}ms removed after each spike")
            param_fig.savefig(os.path.join(sub_output_dir, f"mcmc_params_{i}.png"))

            for ax in param_axs:
                ax.cla()

        # Plot steady states on one axis for comparison
        fig = plt.figure(figsize=(24, 22))
        axs = fig.subplots(5)

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
            sns.kdeplot(data=steady_states_df, x=var, ax=axs[i], shade=False, common_norm=True,
                        hue='removal_duration', palette='viridis', legend=(i==0))


        plot_x_lims = np.quantile(steady_state_samples[-1], (.05, .95))
        x_window_size = plot_x_lims[1] - plot_x_lims[0]

        plot_x_lims = np.mean(steady_state_samples[0]) + np.array([-x_window_size, x_window_size]) * .5

        ax.set_xlim(*plot_x_lims)

        fig.savefig(os.path.join(output_dir, f"steady_state_prediction_comparison_{voltage}mV.png"))
        for ax in axs:
            ax.cla()

        mcmc_steady_state_samples = [compute_tau_inf_from_samples(samples, voltage=voltage) for samples in mcmc_samples]
        varlist = ['a_inf', 'tau_a', 'r_inf', 'tau_r']
        mcmc_steady_states_df = pd.DataFrame(columns=varlist+['removal_duration'])

        for i in range(len(mcmc_steady_state_samples)):
            sample = mcmc_steady_state_samples[i, 0]
            df = pd.DataFrame(sample, columns=('a_inf', 'tau_a', 'r_inf', 'tau_r'))
            df['removal_duration'] = round(spike_removal_durations[i], 2)
            mcmc_steady_states_df = steady_states_df.append(df, ignore_index=True)

        for i, var in enumerate():
            sns.kdeplot(data=mcmc_steady_states_df, x=var, ax=axs[i], shade=False, common_norm=True,
                        hue='removal_duration', palette='viridis', legend=(i == 0))
        fig.savefig(os.path.join(output_dir, f"mcmc_steady_state_prediction_comparison{voltage}mV.png"))

        for ax in axs:
            ax.cla()

        stds = [np.std(sample) for sample in steady_state_samples]

        std_axs[0].plot(spike_removal_durations, stds)
        std_axs[-1].set_xlabel('time removed after each spike /ms')
        std_axs[0].set_title(f"{voltage}mV")

        for i, var in enumerate(('a_inf', 'tau_a', 'r_inf', 'tau_r')):
            stds = [np.std(steady_states_df[steady_states_df['removal_duration'] == time_removed][var].values)
                    for time_removed in np.unique(steady_states_df['removal_duration'])]
            print(stds)
            std_axs[i + 1].plot(spike_removal_durations, stds)
            std_axs[i + 1].set_ylabel(f"std of {var} estimate")

        std_fig.savefig(os.path.join(sub_output_dir, 'standard_deviations'))

        for ax in std_axs:
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

    # Plot smallest region on top plot
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

    for i, cov in reversed(list(enumerate(covs[0::2]))):
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
    # axs[1].set_xlabel('rotated and scaled view')

    axs[1].legend()
    fig.savefig(os.path.join(output_dir,
                             f"p{p_of_interest[0]+1} and p{p_of_interest[1]+1} confidence regions.png"))
    for ax in axs:
        ax.cla()


def get_mcmc_chains(solver, times, voltages, indices, data, chain_length, default_parameters, burn_in=None):
    # Do the same as above but using mcmc on synthetic data
    print('doing mcmc')
    times = times
    voltages = voltages
    data = data[indices]
    n = len(indices)

    if burn_in is None:
        burn_in = int(chain_length / 10)

    @njit
    def log_likelihood_func(p):
        sol = solver(p, times, voltages)[indices]
        SSE = ((sol - data)**2).sum()
        return -n * 0.5 * np.log(2 * np.pi) - n * 0.5 * np.log(sigma2) - SSE / (2 * sigma2)

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


def draw_likelihood_heatmap(model, solver, params, times, data, sigma2, ranges, no_points, p_index, output_dir):
    xs = np.linspace(ranges[0][0], ranges[0][1], no_points)
    ys = np.linspace(ranges[1][0], ranges[1][1], no_points)

    x_index = p_index[0]
    y_index = p_index[1]

    print(f"Modifying variables {x_index} and {y_index}")

    @njit
    def log_likelihood(x, y):
        n = len(times)
        solver_input = np.copy(params)
        solver_input[x_index] = x
        solver_input[y_index] = y
        output = solver(solver_input, times)
        error = output - data
        SSE = (error**2).sum()
        return -n * 0.5 * np.log(2 * np.pi * sigma2) - SSE / (2 * sigma2)

    fix_parameters = [i for i in range(9) if i not in p_index]

    print(f"Fixing parameters {fix_parameters}")

    mle, _ = common.fit_model(model, data, params, fix_parameters=fix_parameters,
                              max_iterations=1000)

    plt.plot(times, data, color='grey', label='data')
    plt.plot(times, solver(params, times), label='true_model')
    mle_params = np.copy(params)
    mle_params[p_index[0]] = mle[0]
    mle_params[p_index[1]] = mle[1]
    plt.plot(times, solver(mle_params, times), label='mle')
    plt.legend()

    xs, ys = np.meshgrid(xs, ys)
    zs = []
    for x, y in zip(xs.flatten(), ys.flatten()):
        zs.append(log_likelihood(x,y))

    zs = np.array(zs).reshape(xs.shape)

    print(zs, zs.shape)

    fig = plt.figure(figsize=(18, 14))
    ax = fig.subplots()

    # limits for colormap
    ll_of_true_params = log_likelihood(*params[[p_index[0], p_index[1]]])

    c = ax.pcolormesh(
        xs,
        ys,
        zs,
        vmax = np.max(zs),
        vmin = 0,
        label="log likelihood",
        shading="gouraud",
        rasterized=True
    )

    ax.set_xlabel(f"p_{p_index[0]+1}")
    ax.set_ylabel(f"p_{p_index[1]+1}")
    ax.axis([ranges[0][0], ranges[0][1], ranges[1][0], ranges[1][1]])

    ax.plot(params[p_index[0]], params[p_index[1]], 'x', color='purple')
    ax.plot(mle[0], mle[1], 'o', color='pink')

    print(f"ll of mle {log_likelihood(*mle)}")
    print(f"ll of true values {ll_of_true_params}")
    max_z = np.argmax(zs)
    print(max_z)
    print(f"max ll on heatmap {np.max(zs)} at {xs.flatten()[np.argmax(zs)], ys.flatten()[np.argmax(zs)]}")

    print(f"std of mle error {(solver(mle_params, times)-data).std()}")
    print(f"std of true_params error {(solver(params, times)-data).std()}")

    fig.colorbar(c)
    fig.savefig(os.path.join(output_dir, f"heatmap_{p_index[0]+1}_{p_index[1]+1}"))
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


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
