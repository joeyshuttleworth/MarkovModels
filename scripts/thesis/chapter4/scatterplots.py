import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pints
from matplotlib.gridspec import GridSpec

import markovmodels

from markovmodels.fitting import infer_reversal_potential, get_best_params
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
    parser.add_argument("--subtraction_df")
    parser.add_argument("--normalise_diagonal", action="store_true")
    parser.add_argument("--vmax", "-m", default=None, type=float)
    parser.add_argument("--model", default='Beattie')
    parser.add_argument("--figsize", default=(5.54, 7), nargs=2, type=float)
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


    df = get_best_params(df)
    df = df.drop_duplicates(subset=['well', 'fitting_protocol',
                                    'validation_protocol', 'fitting_sweep',
                                    'prediction_sweep'], keep='first')

    df = df[~df.protocol.isin(args.ignore_protocols)]
    df = df[~df.fitting_protocol.isin(args.ignore_protocols)]
    df = df.reset_index()

    # Combine first and last staircases
    df.protocol = ['staircaseramp1' if prot in ['staircaseramp2', 'staircaseramp1_2'] else prot
                   for prot in df.protocol]

    # Sort protocols but leave staircaseramp at the front
    global protocols
    protocols = ['staircaseramp1'] + [p for p in df.protocol.unique() if p != 'staircaseramp1']

    # protocols = list(sorted([p for p in df.protocol.unique() if p != 'staircaseramp']))\
        # + ['staircaseramp']

    transformations = make_model_of_class(args.model).transformations
    # Dictionary of units
    global units
    units = {}
    parameter_labels = make_model_of_class(args.model).get_parameter_labels()

    ts = make_model_of_class(args.model).transformations

    global logged_params
    logged_params = [p for t, p in zip(ts, parameter_labels) if isinstance(t, pints.LogTransformation)]

    for param_label, transformation in zip(parameter_labels, transformations):
        units[param_label] = r'mV$^{-1}$' if isinstance(transformation,
                                                        pints.IdentityTransformation)\
            else r'ms$^{-1}$'


    if args.wells:
        df = df[df.well.isin(args.wells)]

    if args.protocols:
        df = df[df.fitting_protocol.isin(args.protocols)]

    global param_labels
    param_labels = make_model_of_class(args.model).get_parameter_labels()
    df[param_labels] = df[param_labels].astype(np.float64)

    if args.adjust_kinetics:
        assert args.subtraction_df

        subtraction_df = pd.read_csv(args.subtraction_df)
        params_df = adjust_kinetics(model_class, params_df, subtraction_df, args.reversal)

    # Drop conductance parameter
    df = df.drop(param_labels[-1], axis='columns')
    param_labels = param_labels[:-1]

    global output_dir
    output_dir = setup_output_directory(args.output_dir, 'scatterplots')

    p1, p2 = param_labels[:2]
    do_coloured_scatterplots(df, p1, p2)

    beta, ll = do_multivariate_regression(df, param_labels)

    with np.printoptions(threshold=np.inf):
        print(f"log likelihood is {ll}")

    no_protocols = len(df.protocol.unique())
    no_wells = len(df.well.unique())

    beta_p, ll_p = do_multivariate_regression(df, param_labels, no_well_effect=True)
    beta_w, ll_w = do_multivariate_regression(df, param_labels, no_protocol_effect=True)

    with open(os.path.join(output_dir, 'likelihood_ratio_test.txt'), 'w') as fout:
        out_str = f"Likelihood ratio of well effect & protocol effect vs just protocol effect: {ll - ll_p}"

        fout.write(out_str)
        fout.write('\n')
        print(out_str)

        out_str = f"Likelihood ratio of well effect & protocol effect vs just well effect: {ll - ll_w}"
        fout.write(out_str)
        fout.write('\n')
        print(out_str)

    param_combinations = [(p1, p2) for i, p1 in enumerate(param_labels[:-1])
                          for j, p2 in enumerate(param_labels[:-1]) if p1 != p2 and j < i]

    for well in df.well.unique():
        for p1, p2 in param_combinations:
            do_per_plots(None, well, df, p1, p2, output_dir, beta=beta,
                         per_variable='protocol')

            do_per_plots(None, None, df, p1, p2, os.path.join(output_dir,
                                                              'no_effects'),
                         per_variable='protocol')

    for protocol in df.protocol.unique():
        for p1, p2 in param_combinations:
            do_per_plots(protocol, None, df, p1, p2, output_dir, beta=beta,
                         per_variable='well')

    for p1, p2 in param_combinations:
        do_per_plots_all(df, p1, p2, output_dir, per_variable='protocol')
        do_per_plots_all(df, p1, p2, output_dir, per_variable='protocol', normalised=False)
        do_per_plots_all(df, p1, p2, output_dir, per_variable='well',
                     normalise_var='protocol')
        do_per_plots_all(df, p1, p2, output_dir, per_variable='well', normalised=False,
                     normalise_var='protocol')

    markers = ['+', 'x', '1', '2', '3'] + list(range(12))
    marker_dict = {p: markers[i] for i, p in enumerate(df.protocol.unique())}
    markers = [marker_dict[p] for p in df.protocol]

    # Do pairplot
    sns.pairplot(data=df, hue=args.hue, vars=param_labels)
    plt.savefig(os.path.join(output_dir, 'pairplot.pdf'))

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    axs = fig.subplots(2)

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

    no_parameters = len(param_labels)

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    axes = create_axes(fig, int(no_parameters/2))

    plt.close(fig)

    style_dict = {p: i for i, p in enumerate(df.protocol.unique())}
    style = [style_dict[p] for p in df.protocol]

    for i in range(int(no_parameters / 2)):
        ax1 = axes[0][i]
        sns.scatterplot(df, x=param_labels[i*2], y=param_labels[i*2+1],
                        hue=args.hue, legend=args.legend, style=style,
                        ax=ax1)

        ax1.set_xlabel(f"{convert_to_latex(p1)} ({units[p1]})")
        ax1.set_xlabel(f"{convert_to_latex(p2)} ({units[p2]})")

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

    if args.hue == 'well':
        hue = 'protocol'
    else:
        hue = 'well'

    ax1 = axes[0][-1]


    # if args.adjust_kinetics:
    #     ax2 = axes[1][-1]
    #     sns.scatterplot(adjusted_df, x='p9', y='p4',
    #                     hue=hue, legend=args.legend, ax=ax2,
    #                     style=style)

    xmin = min(ax1.get_xlim()[0], ax2.get_xlim()[0])
    xmax = max(ax1.get_xlim()[1], ax2.get_xlim()[1])
    ymin = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    xmax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])

    if args.adjust_kinetics:
        axes[0][0].set_title('without offset adjustment')
        axes[1][0].set_title('with offset adjustment')

    fig.savefig(os.path.join(output_dir, "scatterplot_figure2.pdf"))


def do_coloured_scatterplots(df, p1, p2):
    no_rows = 3
    fig = plt.figure(figsize=args.figsize,
                     constrained_layout=True)
    axs = fig.subplots(no_rows)

    all_ax, well_ax, protocol_ax = axs

    p1_label = convert_to_latex(p1)
    p2_label = convert_to_latex(p2)

    sns.scatterplot(df, x=p1, y=p2, legend=False, ax=all_ax)
    # all_ax.set_title('well')

    sns.scatterplot(df, x=p1, y=p2, hue='well', legend=False, ax=well_ax)
    well_ax.set_title('coloured by well')

    sns.scatterplot(df, x=p1, y=p2, hue='protocol', legend=False, ax=protocol_ax)
    protocol_ax.set_title('coloured by protocol')

    for ax in axs:
        ax.set_xlabel(f"{convert_to_latex(p1)} ({units[p1]})")
        ax.set_ylabel(f"{convert_to_latex(p2)} ({units[p2]})")

        ax.spines[['top', 'right']].set_visible(False)

        if args.log_a:
            if p1 in logged_params:
                ax.set_xscale('log')
            if p2 in logged_params:
                ax.set_yscale('log')

    fig.savefig(os.path.join(output_dir,
                             "colour_scatterplot_fig"))
    plt.close(fig)


def do_per_plots(protocol, well, df, p1, p2, output_dir, beta=None,
                 per_variable='well'):
    # TODO use parameters from well only / protocol only models

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    axs = setup_per_cell_figure(fig, len(df[per_variable].unique()),
                                sharex=True, sharey=True)

    vars = sorted(df[per_variable].unique())

    p1_index = param_labels.index(p1)
    p2_index = param_labels.index(p2)

    no_protocols = len(protocols)

    wells = sorted(df.well.unique())

    for var, ax in zip(vars, axs):
        sub_df = df[df[per_variable] == var]
        ax.scatter(sub_df[p1].values, sub_df[p2].values, marker='.', color='grey')
        if protocol is not None and per_variable == 'well':
            sub_df = sub_df[sub_df.protocol == protocol]
        elif well is not None and per_variable == 'protocol':
            sub_df = sub_df[sub_df.well == well]
        ax.scatter(sub_df[p1].values, sub_df[p2].values, marker='x', color='red')
        ax.set_title(var)

        if beta is not None:
            if per_variable == 'well':
                well = var
                well_index = wells.index(var)
                protocol_index = protocols.index(protocol)
            elif per_variable == 'protocol':
                protocol = var
                protocol_index = protocols.index(var)
                well_index = wells.index(well)
            else:
                raise Exception(f"per_variable must be well or protocol, not {per_variable}")

            well_index = sorted(df.well.unique()).index(well)
            w_effect_index = no_protocols - 1 + well_index

            well_effect = beta[w_effect_index, [p1_index,
                                                p2_index]]
            if protocol_index < len(protocols) - 1:
                protocol_effect = beta[protocol_index, [p1_index,
                                                        p2_index]]
            else:
                protocol_effect = np.array([0, 0])


            if per_variable=='well':
                protocol_effects = [beta[i, [p1_index, p2_index]]
                                    for i in range(len(protocols) - 1)]

                well_only_effect = well_effect + sum(protocol_effects) / len(protocols)

                if args.log_a:
                    well_only_effect = inverse_log_transform(well_only_effect,
                                                             p1, p2)

                ax.scatter(*(well_only_effect).T, color='gold', marker='s')
            elif per_variable=='protocol':
                mean_well_effect = beta[no_protocols - 1:, [p1_index, p2_index]].mean(axis=0)
                protocol_only_effect = mean_well_effect + protocol_effect

                if args.log_a:
                    protocol_effect = inverse_log_transform(protocol_only_effect,
                                                            p1, p2)

                ax.scatter(*(protocol_effect).T, color='gold', marker='s')


            combined_effect = well_effect + protocol_effect

            if args.log_a:
                combined_effect = inverse_log_transform(combined_effect, p1, p2)

            ax.scatter(*(combined_effect).T, color='gold', marker='*')

            if args.log_a:
                if p1 in logged_params:
                    ax.set_xscale('log')
                if p2 in logged_params:
                    ax.set_yscale('log')


    for ax in axs:
        ax.set_xlabel(f"{convert_to_latex(p1)} ({units[p1]})")
        ax.set_ylabel(f"{convert_to_latex(p2)} ({units[p2]})")

    output_dir = os.path.join(output_dir, f'per_{per_variable}_plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if per_variable == 'well':
        fname = os.path.join(output_dir, f"per_well_{p1}_{p2}_{protocol}.pdf")
    if per_variable == 'protocol':
        fname = os.path.join(output_dir, f"per_protocol_{p1}_{p2}_{well}.pdf")
    else:
        fname = os.path.join(output_dir, f"all_{p1}_{p2}.pdf")
    fig.savefig(fname)
    plt.close(fig)


def do_per_plots_all(df, p1, p2, output_dir, normalised=True,
                     per_variable='protocol', normalise_var='well', beta=None):
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

def setup_per_cell_figure(fig, no_cells, sharex=True, sharey=True):

    w_cells = int(np.sqrt(no_cells / 1.8))
    h_cells = float(no_cells) / w_cells

    h_cells = int(h_cells) if h_cells * w_cells == no_cells else int(h_cells) + 1

    axs = fig.subplots(h_cells, w_cells,
                       sharex=sharex, sharey=sharey)

    for ax in axs.flatten():
        ax.spines[['top', 'right']].set_visible(False)

    return axs.flatten()


def do_multivariate_regression(params_df, param_labels,
                               no_protocol_effect=False, no_well_effect=False):
    """
    Set up a linear model for the parameter estimates with well-effects and protocol-effects

    @Returns:
    - a matrix of estimated well-effects and a matrix of estimated protocol-effects
    - the log_likelihood score
    """

    params_df = params_df.copy()
    if args.log_a:
        ts = make_model_of_class(args.model).transformations
        for i, t in enumerate(ts[:-1]):
            if type(t) is pints.LogTransformation:
                params_df[param_labels[i]] = np.log10(params_df[param_labels[i]])

    X, Y = setup_linear_model_coding(params_df, param_labels,
                                     no_protocol_effect=no_protocol_effect,
                                     no_well_effect=no_well_effect)

    no_protocols = len(protocols)

    if no_protocol_effect and no_well_effect:
        return np.array([[]]).astype(np.float64)

    # Do regression
    beta = np.linalg.inv(X.T @ X) @ X.T @ Y

    residuals = Y - (X @ beta)

    n = params_df.values.shape[0]

    sigma_ests = residuals.std(axis=0)

    log_likelihood = 0

    for i in range(len(param_labels)):
        log_likelihood += - (n / 2.0) *  np.log(2*np.pi*sigma_ests[i]**2) - (1.0/2*sigma_ests[i]**2) * np.sum(residuals[:, i]**2)

    return beta, log_likelihood


def setup_linear_model_coding(params_df, param_labels,
                              no_protocol_effect=False, no_well_effect=False):
    """
    Set-up the design matrxi for the linear parameter estimates model
    """

    no_protocols = len(protocols)
    wells = sorted(list(params_df.well.unique()))
    no_wells = len(wells)

    # Number of parameters (excluding conductance)
    no_parameters = len(param_labels)

    # Design matrix
    X = np.full((params_df.index.size, no_wells + no_protocols), 0).astype(int)
    # Create two 'views' of X for the protocol part and the well part
    Xp = X[:, :no_protocols]
    Xw = X[:, no_protocols:]

    assert(Xw.shape[1] == no_wells)

    # Data
    # Each row is a parameter estimate vector
    Y = params_df[param_labels].values


    for c, row in params_df.iterrows():
        protocol = row['protocol']
        well = row['well']

        protocol_index = protocols.index(protocol)
        well_index = wells.index(well)

        Xp[c, protocol_index] = 1
        Xw[c, well_index] = 1

        assert X[c, :].sum() == 2

    assert np.all(np.any(Xp > 0, axis=0))
    assert np.all(np.any(Xw > 0, axis=0))

    if no_protocol_effect and no_well_effect:
        return np.array([[]]).astype(np.float64), Y

    if no_protocol_effect:
        X = Xw

    elif no_well_effect:
        X = Xp

    else:
        # Drop one of the protocol effects
        Xp = Xp[:, :-1]
        X = np.hstack([Xp, Xw])

    return X, Y


def convert_to_latex(string):
    letters = ''.join([s for s in string if str.isalpha(s)])
    digits = ''.join([s for s in string if str.isdigit(s)])

    if digits:
        return f"${letters}_{{{digits}}}$"
    else:
        return f"${letters}$"


def inverse_log_transform(params, p1, p2):
    ts = make_model_of_class(args.model).transformations

    for i, p in enumerate((p1, p2)):
        j = param_labels.index(p)
        t = ts[j]
        if type(t) is pints.LogTransformation:
            params[i] = 10 ** params[i]

    return params


if __name__ == "__main__":
    main()
