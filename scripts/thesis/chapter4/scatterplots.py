import argparse
import os

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import pints
from matplotlib.gridspec import GridSpec

import markovmodels

from markovmodels.fitting import infer_reversal_potential, get_best_params, adjust_kinetics
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
    parser.add_argument("chrono_file")
    parser.add_argument("--output_dir", "-o", help="Directory to output plots to.\
    By default a new directory will be generated", default=None)
    parser.add_argument("--subtraction_df")
    parser.add_argument("--normalise_diagonal", action="store_true")
    parser.add_argument("--vmax", "-m", default=None, type=float)
    parser.add_argument("--model", default='Beattie')
    parser.add_argument("--figsize", default=(5.3, 7), nargs=2, type=float)
    parser.add_argument('--experiment_name', default='newtonrun4', type=str)
    parser.add_argument('--removal_duration', '-r', default=5, type=float)
    parser.add_argument('--reversal', type=float, default=np.nan)
    parser.add_argument('--solver_type', default='hybrid')
    parser.add_argument('--ignore_protocols', nargs='+', default=['longap'])
    parser.add_argument('--ignore_wells', '-w', nargs='+', default=['M06'])
    parser.add_argument('--protocols', nargs='+')
    parser.add_argument('--legend', action='store_true')
    parser.add_argument('--use_real_protocol_labels', action='store_true')
    parser.add_argument('--adjust_kinetics', action='store_true')
    parser.add_argument('--hue', default='well')
    parser.add_argument('--markers', default='protocol')
    parser.add_argument('--fontsize', type=int, default=9)
    parser.add_argument('--title_fontsize', type=int)
    parser.add_argument('--log_a', action='store_true')

    global args
    args = parser.parse_args()

    if args.fontsize:
        matplotlib.rcParams.update({'font.size': args.fontsize})

    global param_labels
    param_labels = make_model_of_class(args.model).get_parameter_labels()

    chrono_fname = args.chrono_file
    with open(chrono_fname, 'r') as fin:
        lines = fin.read().splitlines()
        protocol_order = [line.split(' ')[0] for line in lines]
        # protocol_order.remove('staircaseramp1_2')

    params_df = pd.read_csv(args.input_file)
    params_df = get_best_params(params_df)

    if args.adjust_kinetics:
        assert args.subtraction_df

        subtraction_df = pd.read_csv(args.subtraction_df)
        params_df = adjust_kinetics(args.model, params_df,
                                    subtraction_df, args.reversal)

    params_df.protocol = ['staircaseramp1' if prot in ['staircaseramp2', 'staircaseramp1_2'] else prot
                          for prot in params_df.protocol]

    print(params_df.protocol.unique())

    # Reorder and relabel protocols
    relabel_dict = {p: r"$d_{" f"{i + 1}" r"}$" for i, p
                    in enumerate(protocol_order) if p != 'staircaseramp1'}
    relabel_dict['staircaseramp1'] = r'$d_{1}$'

    params_df = params_df[~params_df.protocol.isin(args.ignore_protocols)]
    params_df = params_df.reset_index()
    # Combine first and last staircases

    if not args.use_real_protocol_labels:
        params_df['protocol'] = pd.Categorical(params_df['protocol'],
                                               categories=protocol_order,
                                               ordered=True)
        print(relabel_dict)
        params_df.protocol = params_df.protocol.cat.rename_categories(relabel_dict)
        protocols = params_df.protocol.unique()

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

    global output_dir
    output_dir = setup_output_directory(args.output_dir, 'scatterplots')

    if args.ignore_wells:
        # First highlight ignored wells
        p1, p2 = parameter_labels[:2]
        for well in args.ignore_wells:
            do_per_plots(None, well, params_df, p1, p2, output_dir, beta=None,
                         per_variable='protocol', prefix='ignored_wells')

        params_df = params_df[~params_df.well.isin(args.ignore_wells)]

    if args.protocols:
        params_df = params_df[params_df.protocol.isin(args.protocols)]

    do_coloured_scatterplots(params_df, param_labels[0], param_labels[1])

    # Drop conductance parameter
    params_df = params_df.drop(param_labels[-1], axis='columns')
    param_labels = param_labels[:-1]

    beta, ll = do_multivariate_regression(params_df, param_labels)

    with np.printoptions(threshold=np.inf):
        print(f"log likelihood is {ll}")

    no_protocols = len(params_df.protocol.unique())
    no_wells = len(params_df.well.unique())

    beta_p, ll_p = do_multivariate_regression(params_df, param_labels, no_well_effect=True)
    beta_w, ll_w = do_multivariate_regression(params_df, param_labels, no_protocol_effect=True)

    params = params_df[param_labels].values
    residuals = params - params.mean(axis=0)
    std_params  = residuals.std(axis=0, ddof=0).flatten()

    n_estimates = params_df.values.shape[0]

    _, ll_no_effects =  do_multivariate_regression(params_df, param_labels,
                                                   no_protocol_effect=True,
                                                   no_well_effect=True)

    with open(os.path.join(output_dir, 'likelihood_ratio_test.txt'), 'w') as fout:

        out_str = f"Likelihood of full model: {ll}"
        fout.write(out_str)
        fout.write('\n')
        print(out_str)

        out_str = f"Likelihood ratio of well effect & protocol effect vs just protocol effect: {ll - ll_p:.1f}"

        fout.write(out_str)
        fout.write('\n')
        print(out_str)

        out_str = f"Likelihood ratio of well effect & protocol effect vs just well effect: {ll - ll_w:.1f}"
        fout.write(out_str)
        fout.write('\n')
        print(out_str)

        out_str = f"well only likelihood {ll_w:.1f}"
        fout.write(out_str)
        fout.write('\n')
        print(out_str)

        out_str = f"protocol only likelihood {ll_p:.1f}"
        fout.write(out_str)
        fout.write('\n')
        print(out_str)

        out_str = f"no effect likelihood =  {ll_no_effects:.1f}"
        fout.write(out_str)
        fout.write('\n')
        print(out_str)

    param_combinations = [(p1, p2) for i, p1 in enumerate(param_labels[:-1])
                          for j, p2 in enumerate(param_labels[:-1]) if p1 != p2 and i < j]

    for well in params_df.well.unique():
        for p1, p2 in param_combinations:
            do_per_plots(None, well, params_df, p1, p2, output_dir, beta=beta,
                         per_variable='protocol')

            do_per_plots(None, None, params_df, p1, p2, os.path.join(output_dir,
                                                                     'no_effects'),
                         per_variable='protocol')

    for protocol in params_df.protocol.unique():
        for p1, p2 in param_combinations:
            do_per_plots(protocol, None, params_df, p1, p2, output_dir, beta=beta,
                         per_variable='well')

    markers = ['+', 'x', '1', '2', '3'] + list(range(12))
    marker_dict = {p: markers[i] for i, p in enumerate(params_df.protocol.unique())}
    markers = [marker_dict[p] for p in params_df.protocol]

    # Do pairplot
    sns.pairplot(data=params_df, hue=args.hue, vars=param_labels)
    plt.savefig(os.path.join(output_dir, 'pairplot.pdf'))

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    axs = fig.subplots(2)

    params_df['staircase'] = params_df.protocol.isin(['staircaseramp1', 'staircaseramp2'])

    # sns.scatterplot(data=params_df, x='p1', y='p2',
    #                 legend=args.legend,
    #                 hue='staircase', marker='x')

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

    style_dict = {p: i for i, p in enumerate(params_df.protocol.unique())}
    style = [style_dict[p] for p in params_df.protocol]

    for i in range(int(no_parameters / 2)):
        ax1 = axes[0][i]
        sns.scatterplot(params_df, x=param_labels[i*2], y=param_labels[i*2+1],
                        hue=args.hue, legend=args.legend, style=style,
                        ax=ax1)

        ax1.set_xlabel(f"{convert_to_latex(p1)} ({units[p1]})")
        ax1.set_xlabel(f"{convert_to_latex(p2)} ({units[p2]})")

    fig.savefig(os.path.join(output_dir, "scatterplot_figure.pdf"))


def do_coloured_scatterplots(params_df, p1, p2):
    no_rows = 3
    fig = plt.figure(figsize=args.figsize,
                     constrained_layout=True)
    axs = fig.subplots(no_rows, sharex=True)

    all_ax, well_ax, protocol_ax = axs

    p1_label = convert_to_latex(p1)
    p2_label = convert_to_latex(p2)

    sns.scatterplot(params_df, x=p1, y=p2, legend=False, ax=all_ax)
    # all_ax.set_title('well')

    sns.scatterplot(params_df, x=p1, y=p2, hue='well', legend=False, ax=well_ax)
    well_ax.set_title('coloured by well')

    sns.scatterplot(params_df, x=p1, y=p2, hue='protocol', legend=False, ax=protocol_ax)
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

    title_fontsize = args.title_fontsize
    axs[0].set_title('a', fontweight='bold', fontsize=title_fontsize, loc='left')
    axs[1].set_title('b', fontweight='bold', fontsize=title_fontsize, loc='left')
    axs[2].set_title('c', fontweight='bold', fontsize=title_fontsize, loc='left')

    fig.savefig(os.path.join(output_dir,
                             "colour_scatterplot_fig"))
    plt.close(fig)


def do_per_plots(protocol, well, params_df, p1, p2, output_dir, beta=None,
                 per_variable='well', prefix=''):
    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    axs, all_data_ax = setup_per_cell_figure(fig, len(params_df[per_variable].unique()),
                                             sharex=True, sharey=True)

    vars = params_df.copy().sort_values(by=['well', 'protocol'])[per_variable].unique()

    p1_index = param_labels.index(p1)
    p2_index = param_labels.index(p2)

    wells = sorted(list(params_df.well.unique()))
    protocols = sorted(list(params_df.protocol.unique()))
    no_protocols = len(protocols)

    if all_data_ax:
        if per_variable=='well':
            grey_df = params_df[params_df.protocol != protocol]
            sub_df = params_df[params_df.protocol == protocol]
        elif per_variable=='protocol':
            grey_df = params_df[params_df.well != well]
            sub_df = params_df[params_df.well == well]

        all_data_ax.scatter(grey_df[p1].values, grey_df[p2].values, marker='.', color='grey')
        all_data_ax.scatter(sub_df[p1].values, sub_df[p2].values, marker='x', color='red')

    for var, ax in zip(vars, axs):
        sub_df = params_df[params_df[per_variable] == var]
        if protocol is not None and per_variable == 'well':
            grey_df = sub_df[sub_df.protocol != protocol]
            ax.scatter(grey_df[p1].values, grey_df[p2].values, marker='.', color='grey')
            sub_df = sub_df[sub_df.protocol == protocol]
            ax.scatter(sub_df[p1].values, sub_df[p2].values, marker='x', color='red')
        elif well is not None and per_variable == 'protocol':
            grey_df = sub_df[sub_df.well != well]
            ax.scatter(grey_df[p1].values, grey_df[p2].values, marker='.', color='grey')
            sub_df = sub_df[sub_df.well == well]
            ax.scatter(sub_df[p1].values, sub_df[p2].values, marker='x', color='red')

        ax.set_title(var)

        if beta is not None:
            if per_variable == 'well':
                well = var
                well_index = wells.index(well)
                protocol_index = protocols.index(protocol)
            elif per_variable == 'protocol':
                protocol = var
                protocol_index = protocols.index(protocol)
                well_index = wells.index(well)
            else:
                raise Exception(f"per_variable must be well or protocol, not {per_variable}")

            well_index = sorted(params_df.well.unique()).index(well)
            w_effect_index = no_protocols - 1 + well_index

            well_effect = beta[w_effect_index, [p1_index,
                                                p2_index]]

            if protocol_index < len(protocols) - 1:
                protocol_effect = beta[protocol_index, [p1_index,
                                                        p2_index]]
            else:
                protocol_effect = np.array([0, 0])

            if per_variable=='well':
                protocol_effects = beta[:no_protocols - 1, [p1_index, p2_index]]
                well_only_effect = well_effect + sum(protocol_effects) / len(protocols)

                if args.log_a:
                    well_only_effect = inverse_log_transform(well_only_effect,
                                                             p1, p2)

                ax.scatter(*(well_only_effect).T, color='blue', marker='s')
            elif per_variable=='protocol':
                mean_well_effect = beta[no_protocols - 1:, [p1_index, p2_index]].mean(axis=0)
                protocol_only_effect = mean_well_effect + protocol_effect

                if args.log_a:
                    protocol_only_effect = inverse_log_transform(protocol_only_effect,
                                                            p1, p2)

                ax.scatter(*(protocol_only_effect).T, color='blue', marker='s')

            combined_effect = well_effect + protocol_effect

            if args.log_a:
                combined_effect = inverse_log_transform(combined_effect, p1, p2)

            ax.scatter(*(combined_effect).T, color='blue', marker='*')

    if args.log_a:
        if p1 in logged_params:
            for ax in axs:
                ax.set_xscale('log')
        if p2 in logged_params:
            for ax in axs:
                ax.set_yscale('log')

    for ax in axs[::2]:
        ax.set_ylabel(f"{convert_to_latex(p2)} ({units[p2]})")

    for ax in axs.flatten()[-2:]:
        ax.set_xlabel(f"{convert_to_latex(p1)} ({units[p1]})")

    if all_data_ax:
        all_data_ax.set_xlabel(f"{convert_to_latex(p1)} ({units[p1]})")
        all_data_ax.set_ylabel(f"{convert_to_latex(p2)} ({units[p2]})")
        if per_variable == 'well':
            all_data_ax.set_title('all wells')
        elif per_variable == 'protocol':
            all_data_ax.set_title('all protocols')

    output_dir = os.path.join(output_dir, f'per_{per_variable}_plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if per_variable == 'well':
        fname = f"per_well_{p1}_{p2}_{protocol}.pdf"
    elif per_variable == 'protocol':
        fname = f"per_protocol_{p1}_{p2}_{well}.pdf"
    else:
        fname = f"all_{p1}_{p2}.pdf"

    if prefix:
        fname = prefix + '_' + fname

    fname = os.path.join(output_dir, fname)

    fig.savefig(fname)
    plt.close(fig)


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

    protocols = sorted(list(params_df.protocol.unique()))
    wells = sorted(list(params_df.well.unique()))
    no_wells = len(wells)
    no_protocols = len(protocols)

    no_protocols = len(protocols)

    # Do regression
    if no_protocol_effect and no_well_effect:
        residuals = Y - Y.mean(axis=0)[None, :]
        beta = Y.mean(axis=0).flatten()

    else:
        beta = np.linalg.solve(X.T @ X, X.T @ Y)
        residuals = Y - (X @ beta)

    n = params_df.values.shape[0]
    sigma_ests = residuals.std(axis=0, ddof=0).flatten()

    log_likelihood = 0
    for i in range(len(param_labels)):
        log_likelihood += - (n / 2.0) *  np.log(2*np.pi*sigma_ests[i]**2) - (1.0/(2*sigma_ests[i]**2)) * np.sum(residuals[:, i]**2)

    return beta, log_likelihood


def setup_linear_model_coding(params_df, param_labels,
                              no_protocol_effect=False, no_well_effect=False):
    """
    Set-up the design matrxi for the linear parameter estimates model
    """

    protocols = sorted(list(params_df.protocol.unique()))
    wells = sorted(list(params_df.well.unique()))
    no_wells = len(wells)
    no_protocols = len(protocols)

    # Number of parameters (excluding conductance)
    no_parameters = len(param_labels)

    # Design matrix
    X = np.full((params_df.shape[0], no_wells + no_protocols), 0).astype(int)
    # Create two 'views' of X for the protocol part and the well part
    Xp = X[:, :no_protocols]
    Xw = X[:, no_protocols:]

    assert Xw.shape[1] == no_wells

    # Data
    # Each row is a parameter estimate vector
    Y = params_df[param_labels].values

    for i, (_, row) in enumerate(params_df.iterrows()):
        protocol = row['protocol']
        well = row['well']

        if well in args.ignore_wells:
            continue

        protocol_index = protocols.index(protocol)
        well_index = wells.index(well)

        Xp[i, protocol_index] = 1
        Xw[i, well_index] = 1

        assert X[i, :].sum() == 2

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


def setup_per_cell_figure(fig, no_cells, sharex=True, sharey=True):
    w_cells = 2
    h_cells = float(no_cells) / w_cells

    h_cells = int(h_cells) if h_cells * w_cells == no_cells else int(h_cells) + 1

    if h_cells * w_cells == no_cells - 1:
        axs = fig.subplots(h_cells + 1, w_cells,
                           sharex=sharex, sharey=sharey)
        all_data_ax = axs[-1, -1]
        axs = axs.flatten()[:-1]
    else:
        all_data_ax = None
        axs = fig.subplots(h_cells, w_cells,
                           sharex=sharex, sharey=sharey).flatten()

    for ax in axs.flatten():
        ax.spines[['top', 'right']].set_visible(False)

    if all_data_ax:
        all_data_ax.spines[['top', 'right']].set_visible(False)

    return axs.flatten(), all_data_ax


if __name__ == "__main__":
    main()
