import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

import markovmodels

from markovmodels.fitting import infer_reversal_potential
from markovmodels.utilities import setup_output_directory
from markovmodels.model_generation import make_model_of_class


def create_axes(fig, no_rows):
    if args.adjust_kinetics:
        gs = GridSpec(no_rows, 2, figure=fig)

        return [[fig.add_subplot(gs[i, 0]) for i in range(no_rows)],
                [fig.add_subplot(gs[i, 1]) for i in range(no_rows)]]
    else:
        return [fig.subplots(no_rows)]


def main():
    description = ""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("input_file", help="CSV file listing model errors for each cell and protocol")
    parser.add_argument("--output_dir", "-o", help="Directory to output plots to.\
    By default a new directory will be generated", default=None)
    parser.add_argument("--normalise_diagonal", action="store_true")
    parser.add_argument("--vmax", "-m", default=None, type=float)
    parser.add_argument("--share_limits", action='store_true')
    parser.add_argument("--model", default='Beattie')
    parser.add_argument("--figsize", default=(8, 13), nargs=2, type=float)
    parser.add_argument('--experiment_name', default='newtonrun4', type=str)
    parser.add_argument('--removal_duration', '-r', default=5, type=float)
    parser.add_argument('--reversal', type=float, default=np.nan)
    parser.add_argument('--solver_type', default='hybrid')
    parser.add_argument('--ignore_protocols', nargs='+', default=['longap'])
    parser.add_argument('--wells', '-w', nargs='+')
    parser.add_argument('--protocols', nargs='+')
    parser.add_argument('--legend', action='store_true')
    parser.add_argument('--adjust_kinetics', action='store_true')
    parser.add_argument('--hue', default='well')
    parser.add_argument('--markers', default='protocol')
    parser.add_argument('--log_a', action='store_true')

    global args
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)

    if 'fitting_protocol' not in df.columns:
        df['fitting_protocol'] = df['protocol']

    if 'validation_protocol' not in df.columns:
        df['validation_protocol'] = df['protocol']

    if 'fitting_sweep' not in df.columns:
        df['fitting_sweep'] = df['sweep']

    if 'prediction_sweep' not in df.columns:
        df['prediction_sweep'] = df['sweep']


    df = markovmodels.fitting.get_best_params(df)
    df = df.drop_duplicates(subset=['well', 'fitting_protocol',
                                    'validation_protocol', 'fitting_sweep',
                                    'prediction_sweep'], keep='first')

    df = df[~df.protocol.isin(args.ignore_protocols)]
    df = df[~df.fitting_protocol.isin(args.ignore_protocols)]

    if args.wells:
        df = df[df.well.isin(args.wells)]

    if args.protocols:
        df = df[df.fitting_protocol.isin(args.protocols)]

    param_labels = make_model_of_class(args.model).get_parameter_labels()
    df[param_labels] = df[param_labels].astype(np.float64)

    global output_dir
    output_dir = setup_output_directory(args.output_dir, 'scatterplots')

    print(df.protocol.unique())

    param_combinations = [(p1, p2) for i, p1 in enumerate(param_labels[:-1])
                          for j, p2 in enumerate(param_labels[:-1]) if p1 != p2 and j < i]

    for p1, p2 in param_combinations:
        do_per_plots(df, p1, p2, output_dir, per_variable='protocol')
        do_per_plots(df, p1, p2, output_dir, per_variable='protocol', normalised=False)
        do_per_plots(df, p1, p2, output_dir, per_variable='well',
                     normalise_var='protocol')
        do_per_plots(df, p1, p2, output_dir, per_variable='well', normalised=False,
                     normalise_var='protocol')

    for protocol in df.protocol.unique():
        print(args.ignore_protocols)
        print(f"plotting {protocol}")
        do_per_cell_plots(protocol, df, 'p1', 'p2', output_dir)

    var_names = make_model_of_class(args.model).get_parameter_labels()

    markers = ['+', 'x', '1', '2', '3'] + list(range(12))
    marker_dict = {p: markers[i] for i, p in enumerate(df.protocol.unique())}
    markers = [marker_dict[p] for p in df.protocol]

    # Do pairplot
    sns.pairplot(data=df, hue=args.hue, vars=var_names)
    plt.savefig(os.path.join(output_dir, 'pairplot.pdf'))

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    ax = fig.subplots()

    df['staircase'] = df.fitting_protocol.isin(['staircaseramp1', 'staircaseramp2'])

    sns.scatterplot(data=df, x='p1', y='p2',
                    legend=args.legend,
                    hue='staircase', marker='x')
    default_params = make_model_of_class(args.model).get_default_parameters()
    if args.model == 'Beattie':
        ax.scatter([default_params[0]], [default_params[1]], marker='x', color='pink', label='default')
        ax.set_xlabel(r'$p_1$ (ms$^{-1}$)')
        ax.set_ylabel(r'$p_2$ (mV$^{-1}$)')

    fig.savefig(os.path.join(output_dir, "fig1.pdf"))
    plt.close(fig)

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    axes = create_axes(fig, 5)

    if args.log_a:
        # TODO Implement for all models
        for var in ['p1', 'p3', 'p5', 'p7']:
            df[var] = np.log10(df[var])


    plt.close(fig)

    # def adjust_rates(row):
    #     protocol = row['fitting_protocol']
    #     well = row['well']
    #     sweep = row['sweep']

    #     inferred_E_rev = infer_reversal_potential(protocol,
    #                                               data.current, data.time)

    #     offset = inferred_E_rev - args.reversal

    #     row[param_labels[0]] *= np.exp(row[param_labels[1]] * offset)
    #     row[param_labels[2]] *= np.exp(-row[param_labels[3]] * offset)
    #     row[param_labels[4]] *= np.exp(row[param_labels[5]] * offset)
    #     row[param_labels[6]] *= np.exp(-row[param_labels[7]] * offset)

    #     return row

    # if args.adjust_kinetics:
    #     adjusted_df = df.apply(adjust_rates, axis=1)

    style_dict = {p: i for i, p in enumerate(df.protocol.unique())}
    style = [style_dict[p] for p in df.protocol]

    for i in range(4):
        ax1 = axes[0][i]
        ax2 = axes[0][i]
        sns.scatterplot(df, x=param_labels[i*2], y=param_labels[i*2+1],
                        hue=args.hue, legend=args.legend, style=style,
                        ax=ax1)

        # if args.adjust_kinetics:
        #     ax2 = axes[1][i]
        #     sns.scatterplot(adjusted_df, x=param_labels[i*2], y=param_labels[i*2+1],
        #                     hue=args.hue, legend=args.legend, style=style,
        #                     ax=ax2)

        xmin = min(ax1.get_xlim()[0], ax2.get_xlim()[0])
        xmax = max(ax1.get_xlim()[1], ax2.get_xlim()[1])

        ax1.set_xlim((xmin, xmax))
        ax2.set_xlim((xmin, xmax))

        ymin = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
        ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])

        ax1.set_ylim((ymin, ymax))
        ax2.set_ylim((ymin, ymax))

    ax1 = axes[0][-1]
    sns.scatterplot(df, x='p9', y='p4',
                    hue=args.hue, legend=args.legend, style=style,
                    ax=ax1)

    if args.adjust_kinetics:
        ax2 = axes[1][-1]
        sns.scatterplot(df, x='p9', y='p4',
                        hue=args.hue, legend=args.legend, style=style,
                        ax=ax2)

    xmin = min(ax1.get_xlim()[0], ax2.get_xlim()[0])
    xmax = max(ax1.get_xlim()[1], ax2.get_xlim()[1])
    ymin = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    xmax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])

    if args.adjust_kinetics:
        axes[0][0].set_title('without offset adjustment')
        axes[1][0].set_title('with offset adjustment')

    fig.savefig(os.path.join(output_dir, "scatterplot_figure.pdf"))

    for ax in axes[0]:
        ax.cla()

    if args.adjust_kinetics:
        for ax in axes[1]:
            ax.cla()

    for i in range(2):
        ax = axes[0][i]
        sns.scatterplot(df, x=param_labels[i*2], y=param_labels[i*2+2],
                        hue=args.hue, legend=args.legend,
                        ax=ax, style=style)

        if args.adjust_kinetics:
            ax2 = axes[1][i]
            sns.scatterplot(adjusted_df, x=param_labels[i*2], y=param_labels[i*2+2],
                            hue=args.hue, legend=args.legend,
                            ax=ax2, style=style)

    for i in range(2):
        ax = axes[0][i+2]
        sns.scatterplot(df, x=param_labels[i*4+1], y=param_labels[i*4+3],
                        hue=args.hue, legend=args.legend,
                        ax=ax, style=style)

        if args.adjust_kinetics:
            ax2 = axes[1][i]
            sns.scatterplot(adjusted_df, x=param_labels[i*2+1], y=param_labels[i*2+3],
                            hue=args.hue, legend=args.legend,
                            ax=ax2, style=style)

    for i in range(4):
        ax1 = axes[0][i]
        if args.adjust_kinetics:
            ax2 = axes[1][i]
        else:
            ax2 = ax1

        xmin = min(ax1.get_xlim()[0], ax2.get_xlim()[0])
        xmax = max(ax1.get_xlim()[1], ax2.get_xlim()[1])

        ax1.set_xlim((xmin, xmax))
        ax2.set_xlim((xmin, xmax))

        ymin = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
        ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])

        ax1.set_ylim((ymin, ymax))
        ax2.set_ylim((ymin, ymax))

    if args.hue == 'well':
        hue = 'protocol'
    else:
        hue = 'well'

    ax1 = axes[0][-1]
    sns.scatterplot(df, x='p9', y='p4', hue=hue,
                    legend=args.legend, ax=ax1, style=style)

    if args.adjust_kinetics:
        ax2 = axes[1][-1]
        sns.scatterplot(adjusted_df, x='p9', y='p4',
                        hue=hue, legend=args.legend, ax=ax2,
                        style=style)

    xmin = min(ax1.get_xlim()[0], ax2.get_xlim()[0])
    xmax = max(ax1.get_xlim()[1], ax2.get_xlim()[1])
    ymin = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    xmax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])

    if args.adjust_kinetics:
        axes[0][0].set_title('without offset adjustment')
        axes[1][0].set_title('with offset adjustment')

    fig.savefig(os.path.join(output_dir, "scatterplot_figure2.pdf"))


def do_per_cell_plots(protocol, df, p1, p2, output_dir):
    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    axs = setup_per_cell_figure(fig, len(df.well.unique()))

    for well, ax in zip(df.well.unique(), axs):
        sub_df = df[df.well == well]
        ax.scatter(sub_df[p1].values, sub_df[p2].values, marker='.', color='grey')
        sub_df = sub_df[sub_df.protocol == protocol]
        ax.scatter(sub_df[p1].values, sub_df[p2].values, marker='x', color='red')
        ax.set_title(well)

    output_dir = os.path.join(output_dir, 'per_cell_plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig.savefig(os.path.join(output_dir, f"per_cell_{p1}_{p2}_{protocol}"))
    plt.close(fig)

def do_per_plots(df, p1, p2, output_dir, normalised=True, per_variable='protocol', normalise_var='well'):
    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    axs = setup_per_cell_figure(fig, len(df[per_variable].unique()))
    for n_var in df[normalise_var].unique():
        norm_df = df[df[normalise_var] == n_var].copy()

        if normalised:
            for p in (p1, p2):
                norm_df[p] = (norm_df[p] - norm_df[p].mean()) / norm_df[p].std()

        for ax in axs:
            ax.scatter(norm_df[p1].values, norm_df[p2].values, marker='.', color='grey')

        for var, ax in zip(sorted(df[per_variable].unique()), axs):
            sub_df = norm_df[norm_df[per_variable] == var]
            ax.scatter(sub_df[p1].values, sub_df[p2].values, marker='x', color='red')
            ax.set_title(var)

    if normalised:
        output_dir = os.path.join(output_dir, f'normalised_per_{per_variable}_plots')
    else:
        output_dir = os.path.join(output_dir, f'per_{per_variable}_plots')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if normalised:
        fname = f"per_{per_variable}_normalised_{p1}_{p2}"
    else:
        fname = f"per_{per_variable}_{p1}_{p2}"

    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)

def setup_per_cell_figure(fig, no_cells):

    w_cells = int(np.sqrt(no_cells / 1.8))
    h_cells = float(no_cells) / w_cells

    h_cells = int(h_cells) if h_cells * w_cells == no_cells else int(h_cells) + 1

    axs = fig.subplots(h_cells, w_cells)

    return axs.flatten()


if __name__ == "__main__":
    main()
