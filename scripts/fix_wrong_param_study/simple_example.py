#!/usr/bin/env python3

import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import scipy

from markovmodels import common
from matplotlib.gridspec import GridSpec

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 8})
rc('text', usetex=True)
rc('figure', dpi=500)


def create_axes(fig):
    gs = GridSpec(2, 4, figure=fig,
                  height_ratios=[.4, 1],
                  bottom=.3)

    # caption_axes = [fig.add_subplot(gs[0, 0]),
    #                 fig.add_subplot(gs[0, 2:4]),
    #                 fig.add_subplot(gs[0, 4:])]

    # caption_axes[0].text(0, .5, r'\textbf a')
    # caption_axes[1].text(0, .5, r'\textbf b')
    # caption_axes[2].text(0, .5, r'\textbf c')

    # for ax in caption_axes:
    #     ax.axis('off')

    # Setup plots of observation times
    observation_time_axes = [
         fig.add_subplot(gs[0, 0]),
         fig.add_subplot(gs[0, 1]),
         fig.add_subplot(gs[0, 2]),
         fig.add_subplot(gs[0, 3]),
    ]

    # observation_time_axes[2].set_title(r'\textbf a')

    prediction_plot_ax = fig.add_subplot(gs[1, 0:2])
    scatter_ax = fig.add_subplot(gs[1, 2:])

    prediction_plot_ax.set_title(r'\textbf b', loc='left')
    scatter_ax.set_title(r'\textbf c', loc='left')

    for ax in observation_time_axes + [prediction_plot_ax, scatter_ax]:
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)

    return observation_time_axes, scatter_ax, prediction_plot_ax


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
        ax.plot(*observed_dataset.T, marker='x', ms=1.5, lw=0, color='grey',
                zorder=10)

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

    argument_parser.add_argument('-o', '--output', default='output')
    argument_parser.add_argument('--figsize', default=[4.685, 3.5], type=int,
                                 nargs=2)
    argument_parser.add_argument('--no_datasets', default=10, type=int)
    argument_parser.add_argument('--sigma', default=0.01, type=float)
    argument_parser.add_argument('--file_format', default='pdf')

    global args
    args = argument_parser.parse_args()
    global output_dir
    output_dir = common.setup_output_directory(args.output, subdir_name='simple_example')

    sigma = args.sigma

    # Generate data sets
    N_datasets = args.no_datasets

    T1 = np.linspace(0, 0.1, 11)
    T2 = np.linspace(0, 1, 11)
    T3 = np.linspace(.2, 1.2, 11)
    T4 = np.linspace(0.5, 1, 11)

    Ts = [T1, T2, T3, T4]

    all_T = np.unique(sorted(np.concatenate(Ts)))

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    observation_axes, prediction_ax, scatter_ax = create_axes(fig)

    global true_theta
    true_theta = np.array([1, 0.1])

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

    observation_axes[0].set_title(r'\textbf a', loc='left')
    # observation_axes[0].set_xlabel(r'$t$')
    # observation_axes[1].set_xlabel(r'$t$')
    # observation_axes[2].set_xlabel(r'$t$')
    # observation_axes[3].set_xlabel(r'$t$')

    observation_axes[0].set_xlabel(r'$t$')
    observation_axes[1].set_xlabel(r'$t$')
    observation_axes[2].set_xlabel(r'$t$')
    observation_axes[3].set_xlabel(r'$t$')

    observation_axes[0].set_title(r'$T_1$')
    observation_axes[1].set_title(r'$T_2$')
    observation_axes[2].set_title(r'$T_3$')
    observation_axes[3].set_title(r'$T_4$')

    rows = []
    for x, T in zip(estimates, ['$T_1$', '$T_2$', '$T_3$', '$T_4$']):
        row = pd.DataFrame(x, columns=[r'$\hat\theta_1$', r'$\hat\theta_2$'])
        row['time_range'] = T
        row['dataset_index'] = list(range(row.values.shape[0]))
        rows.append(row)

    estimates_df = pd.concat(rows, ignore_index=True)

    make_scatter_plots(estimates_df, scatter_ax)
    make_prediction_plots(estimates_df, datasets, prediction_ax)

    fig.savefig(os.path.join(output_dir, f"Fig1.{args.file_format}"))

    estimates_df.to_csv(os.path.join(output_dir, 'fitting_results.csv'))


def make_scatter_plots(df, ax, label=''):

    df['observation times'] = df['time_range']
    g = sns.scatterplot(ax=ax, data=df, x=df.columns[0], y=df.columns[1],
                        hue='observation times', style='observation times',
                        s=25)

    ax.legend(markerscale=0.5)

    g.legend_.set_title('')


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
