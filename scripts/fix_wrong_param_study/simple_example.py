#!/usr/bin/env python3

import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import scipy
import pints

from MarkovModels import common
from matplotlib.gridspec import GridSpec

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 8})
rc('text', usetex=True)
rc('figure', dpi=500)


def create_axes(fig):
    gs = GridSpec(5, 4, figure=fig,
                  height_ratios=[.5, .5, 1, 1, 1])

    # Setup plots of observation times
    observation_time_axes = [[
         fig.add_subplot(gs[0, 0]),
         fig.add_subplot(gs[0, 1]),
         fig.add_subplot(gs[1, 0]),
         fig.add_subplot(gs[1, 1]),
    ], [
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[1, 3]),
    ]]



    # observation_time_axes[2].set_title(r'\textbf a')

    prediction_plot_axs = [fig.add_subplot(gs[3, 0:2]),
                           fig.add_subplot(gs[3, 2:4])]

    mcmc_axs = [fig.add_subplot(gs[4, 0:2]),
                           fig.add_subplot(gs[4, 2:4])]

    scatter_axs = [fig.add_subplot(gs[2, 0:2]),
                   fig.add_subplot(gs[2, 2:4])
                   ]

    # prediction_plot_ax.set_title(r'\textbf b', loc='left')
    # scatter_ax.set_title(r'\textbf c', loc='left')

    # for ax in observation_time_axes + prediction_plot_axs, scatter_axs:
    #     for side in ['top', 'right']:
    # ax.spines[side].set_visible(False)

    return observation_time_axes, scatter_axs, mcmc_axs, prediction_plot_axs


def true_dgp(theta, T):
    theta1, theta2 = theta
    T = np.array(T)

    trajectory = np.exp(-T / theta1) + np.exp(-T / theta2)
    return trajectory


def discrepant_forward_model(theta, T):
    T = np.array(T)
    return theta[1] * np.exp(-T / theta[0])


def generate_data_set(T, theta=[10, 1], sigma=0.01):
    true_trajectory = true_dgp(theta, T)
    obs = np.random.normal(true_trajectory, sigma, T.shape)
    data = np.vstack((T, obs,)).T

    return data


def fit_model(dataset, T, ax=None, label=''):

    observed_dataset = np.vstack(list({tuple(row) for row in dataset if row[0]
                                       in T}))
    observed_dataset = observed_dataset[observed_dataset[:, 0].argsort()]

    # use scipy optimise
    def min_func(theta):
        return np.sum((observed_dataset[:, 1] - discrepant_forward_model(theta,
                                                                         T))**2)
    x0 = [1, 1]

    bounds = [[0, 1e5], [0, 1e5]]

    n_repeats = 15
    result = None
    for i in range(n_repeats):
        x0 = [1, 1]
        if x0[1] > x0[0]:
            x0 = x0[[1, 0]]
        new_result = scipy.optimize.dual_annealing(min_func, x0=x0,
                                                   bounds=bounds)
        if result:
            if new_result.fun < result.fun:
                result = new_result
        else:
            result = new_result

    if ax:
        if len(T) < 15:
            ax.scatter(*observed_dataset.T, color='grey',
                       zorder=1, marker='x')
        else:
            ax.plot(*observed_dataset.T, lw=0.5, color='grey',
                    zorder=1, alpha=.75)

        all_T = np.linspace(0, max(*T, 1.2), 100)
        ax.plot(all_T, discrepant_forward_model(result.x, all_T), '--',
                lw=.75, color='red', label='fitted_model')
        ax.plot(all_T, true_dgp(true_theta, all_T), label='true DGP', lw=.75)

        ax.set_xlim(0, 1.3)
        ax.set_ylim(0, 2.25)

        # ax.set_xlabel('$t$')
        ax.set_ylabel('$y$', rotation=0)

        # ax.legend()
        # fig.savefig(os.path.join(output_dir, f"fitting_{label}"))
        # plt.close(fig)

    return result.x


def main():

    argument_parser = argparse.ArgumentParser()

    argument_parser.add_argument('-o', '--output')
    argument_parser.add_argument('--figsize', default=[4.685, 7.67], type=int,
                                 nargs=2)
    argument_parser.add_argument('--results_dir')
    argument_parser.add_argument('--no_datasets', default=10, type=int)
    argument_parser.add_argument('--sigma', default=0.01, type=float)
    argument_parser.add_argument('--file_format', default='pdf')
    argument_parser.add_argument('--no_chains', default=1, type=int)
    argument_parser.add_argument('--chain_length', default=10000, type=int)
    argument_parser.add_argument('--burn_in', default=0, type=int)
    argument_parser.add_argument('--sampling_frequency', default=10, type=int)

    global args
    args = argument_parser.parse_args()
    global output_dir
    output_dir = common.setup_output_directory(args.output, subdir_name='simple_example')

    global true_theta
    true_theta = np.array([1, 0.1])

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    observation_axes, scatter_axes, mcmc_axes, prediction_axes = create_axes(fig)

    if not args.results_dir:
        generate_data_and_fit(observation_axes[0], scatter_axes[0], mcmc_axes[0],
                              prediction_axes[0], sampling_frequency=10,
                              sigma=args.sigma)

        generate_data_and_fit(observation_axes[1], scatter_axes[1], mcmc_axes[1],
                              prediction_axes[1], sampling_frequency=100,
                              sigma=args.sigma, dash=True)

    fig.savefig(os.path.join(output_dir, f"Fig1.{args.file_format}"))


def generate_data_and_fit(observation_axes, scatter_ax, mcmc_ax, prediction_ax,
                          sampling_frequency, sigma, dash=False):
    # Generate data sets
    N_datasets = args.no_datasets

    T1 = np.linspace(0, 0.1, sampling_frequency + 1)
    T2 = np.linspace(0, 1, sampling_frequency + 1)
    T3 = np.linspace(.2, 1.2, sampling_frequency + 1)
    T4 = np.linspace(0.5, 1, sampling_frequency + 1)

    Ts = [T1, T2, T3, T4]

    all_T = np.unique(sorted(np.concatenate(Ts)))

    datasets = [generate_data_set(all_T, true_theta, sigma=sigma) for i in
                range(N_datasets)]

    def fit_datasets_using_times(datasets, T, ax, label=''):
        thetas = []
        for i, dataset in enumerate(datasets):
            if i != 0:
                ax = None
            theta_1, theta_2 = fit_model(dataset, T, ax=ax, label=f"{label}_{i}")
            thetas.append([theta_1, theta_2])
        return np.vstack(thetas)

    print(datasets)
    estimates = [fit_datasets_using_times(datasets, T,
                                          observation_axes[i],
                                          label=f"{i}") for i, T in enumerate((Ts))]
    estimates.append(fit_datasets_using_times(datasets, all_T, None, r'{\textrm{all}}'))

    observation_axes[0].set_title(r'\textbf a', loc='left')
    # observation_axes[0].set_xlabel(r'$t$')
    # observation_axes[1].set_xlabel(r'$t$')
    # observation_axes[2].set_xlabel(r'$t$')
    # observation_axes[3].set_xlabel(r'$t$')

    observation_axes[0].set_xlabel(r'$t$')
    observation_axes[1].set_xlabel(r'$t$')
    observation_axes[2].set_xlabel(r'$t$')
    observation_axes[3].set_xlabel(r'$t$')

    observation_axes[0].set_title(r"$T_1\'$" if dash else r'$T_1$')
    observation_axes[1].set_title(r"$T_2\'$" if dash else r'$T_2$')
    observation_axes[2].set_title(r"$T_3\'$" if dash else r'$T_3$')
    observation_axes[3].set_title(r"$T_4\'$" if dash else r'$T_4$')

    rows = []
    if dash:
        T_labels = [r'$\prime T_1$', r'$\prime T_2$', r'$\prime T_3$', r'$\prime T_4$', r'$\prime T_{\textrm{all}}']
    else:
        T_labels = ['$T_1$', '$T_2$', '$T_3$', '$T_4$', r'$T_{\textrm{all}}']

    for x, T in zip(estimates, T_labels):
        row = pd.DataFrame(x, columns=[r'$\hat\theta_1$', r'$\hat\theta_2$'])
        row['time_range'] = T
        row['dataset_index'] = list(range(row.values.shape[0]))
        rows.append(row)

    estimates_df = pd.concat(rows, ignore_index=True)

    make_scatter_plots(estimates_df, scatter_ax)
    make_prediction_plots(estimates_df, datasets, prediction_ax)

    estimates_df.to_csv(os.path.join(output_dir, f"fitting_results_{sampling_frequency}.csv"))

    # Now use PINTS MCMC on the same problem
    Ts.append(np.array(all_T))
    do_mcmc(datasets, Ts, mcmc_ax, sampling_frequency)


def do_mcmc(datasets, observation_times, mcmc_ax, sampling_frequency):
    # Use uninformative prior
    prior = pints.UniformLogPrior([0, 0], [1e1, 1e1])

    class pints_log_likelihood(pints.LogPDF):
        def __init__(self, observation_times, data, sigma2):
            self.observation_times = observation_times
            self.data = data
            self.sigma2 = sigma2

        def __call__(self, p):
            # Likelihood function

            observed_dataset = np.vstack(list({tuple(row) for row in self.data if row[0]
                                               in self.observation_times}))

            observed_dataset = observed_dataset[observed_dataset[:, 0].argsort()][:, 1]

            error = discrepant_forward_model(p, self.observation_times) - observed_dataset
            SSE = np.sum(error**2)

            n = len(self.observation_times)

            ll = -n * 0.5 * np.log(2 * np.pi * self.sigma2) - SSE / (2 * self.sigma2)
            return ll

        def n_parameters(self):
            return len(true_theta)

    starting_parameters = prior.sample(n=args.no_chains)
    # starting_parameters = np.tile(true_theta, reps=[args.no_chains])

    data_set = datasets[0]

    mcmc_figure = plt.figure(figsize=args.figsize)

    alpha = 0.5
    palette = sns.color_palette()
    palette = [(r, g, b, alpha) for r, g, b in palette[:len(observation_times)]]

    dfs = []
    for i, observation_times in enumerate(observation_times):
        print('performing mcmc on dataset %d' % i)
        print(data_set)
        posterior = pints.LogPosterior(pints_log_likelihood(observation_times,
                                                            data_set, args.sigma**2), prior)
        mcmc = pints.MCMCController(posterior, args.no_chains,
                                    starting_parameters,
                                    method=pints.HaarioBardenetACMC)

        mcmc.set_max_iterations(args.chain_length)
        samples = mcmc.run()[:, args.burn_in:, :]

        np.save(os.path.join(output_dir, f"mcmc_chains_chains_{sampling_frequency}.npy"),
                samples)

        sub_df = pd.DataFrame(samples.reshape([-1, 2]), columns=[r'$\theta_1$',
                                                                 r'$\theta_2$'])
        sub_df['observation times'] = r'$T_{' f"{i+1}" r'}$' if i < 4 else r'$T_{\textrm{all}}$'

        dfs.append(sub_df)

    df = pd.concat(dfs, ignore_index=True)

    plot_mcmc_kde(mcmc_ax, df, palette)


def plot_mcmc_kde(mcmc_ax, df, palette):
    sns.kdeplot(data=df, x=r'$\theta_1$', y=r'$\theta_2$',
                palette=palette, hue='observation times',
                levels=[.01, 0.1, .5], ax=mcmc_ax)


def make_scatter_plots(df, ax, label=''):
    df['observation times'] = df['time_range']
    g = sns.scatterplot(ax=ax, data=df, x=df.columns[0], y=df.columns[1],
                        hue='observation times', style='observation times',
                        s=25, legend=False)


def make_prediction_plots(estimates, datasets, ax):
    # Use only first dataset
    df = estimates[estimates.dataset_index == 0]

    linestyles = [(0, ()),
      (0, (1, 2)),
      (0, (1, 1)),
      (0, (5, 5)),
      (0, (3, 5, 1, 5)),
      (0, (3, 5, 1, 5, 1, 5))]

    predictions = []
    T = np.linspace(0, 2, 100)

    for time_range in df.time_range.unique():
        params = df[df.time_range == time_range][[r'$\hat\theta_1$', r'$\hat\theta_2$']].values[0, :].astype(np.float64)
        prediction = discrepant_forward_model(params, T)
        predictions.append(prediction)

        ax.plot(T, prediction, '--', color='red', lw=.5)

    predictions = np.vstack(predictions)
    # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # blue = colors[0]
    ax.plot(T, true_dgp(true_theta, T), label='true DGP', lw=1)
    max_predict = np.max(predictions, axis=0)
    min_predict = np.min(predictions, axis=0)

    # ax.plot(T, min_predict, '--', color='red')
    # ax.plot(T, max_predict, '--', color='red')

    ax.fill_between(T, min_predict, max_predict, color='orange', alpha=0.25)

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$y$', rotation=0)

    # fig.savefig(os.path.join(output_dir, 'prediction_plot'))


if __name__ == '__main__':
    main()
