import argparse
import os

import itertools
import logging
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
from numba import njit
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch, Rectangle

from matplotlib import rc

import markovmodels
from markovmodels.model_generation import make_model_of_class
from markovmodels.fitting import get_best_params, compute_predictions_df
from markovmodels.ArtefactModel import ArtefactModel
from markovmodels.utilities import setup_output_directory, get_data, get_all_wells_in_directory
from markovmodels.voltage_protocols import get_protocol_list, get_ramp_protocol_from_json, make_voltage_function_from_description
from markovmodels.voltage_protocols import remove_spikes, detect_spikes

multiprocessing_kws = {'maxtasksperchild': 1}

plt.rcParams["axes.formatter.use_mathtext"] = True

rc('font', **{'size': 12})
# rc('text', usetex=True)
# rc('figure', dpi=400, facecolor=[0]*4)
# rc('axes', facecolor=[0]*4)
# rc('savefig', facecolor=[0]*4)
rc('figure', autolayout=True)

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('data_directory', help='directory where data is stored')
    parser.add_argument('fitting_results', type=str)
    parser.add_argument('subtraction_df')
    parser.add_argument('chrono_file')
    parser.add_argument('--protocols', nargs='+')
    parser.add_argument('--use_mock_data', action='store_true')
    parser.add_argument('--use_raw_data', action='store_true')
    parser.add_argument('--ignore_protocols', nargs='+', default=['longap'], type=str)
    parser.add_argument('-w', '--wells', type=str, nargs='+')
    parser.add_argument('--removal_duration', type=float, default=5.0)
    parser.add_argument('--experiment_name', '-e', default='newtonrun4')
    parser.add_argument('--validation_protocols', default=['longap'], nargs='+')
    parser.add_argument('--figsize', '-f', nargs=2, type=float, default=[5.54, 7])
    parser.add_argument('--fig_title', '-t', default='')
    parser.add_argument('--nolegend', action='store_true')
    parser.add_argument('--dpi', '-d', default=500, type=int)
    parser.add_argument('--fontsize', type=int, default=12)
    parser.add_argument('--show_uncertainty', action='store_true')
    parser.add_argument('--shared_plot_limits', action='store_true')
    parser.add_argument('--no_voltage', action='store_true')
    parser.add_argument('--file_format', default='')
    parser.add_argument('--reversal', default=-91.71, type=float)
    parser.add_argument('--output', '-o')
    parser.add_argument('--no_cpus', '-c', default=1, type=int)
    parser.add_argument('--model_classes', nargs='+')
    parser.add_argument('--cases', nargs='+')

    global args
    args = parser.parse_args()

    if args.model_classes is None:
        args.model_classes = ['model2', 'model3', 'model10', 'Wang']

    global output_dir
    output_dir = setup_output_directory(args.output, 'chapter_4_heatmaps')

    if args.fontsize:
        matplotlib.rcParams.update({'font.size': args.fontsize})

    infer_reversal_params = np.loadtxt(os.path.join('data', 'BeattieModel_roomtemp_staircase_params.csv')).flatten().astype(np.float64)

    subtraction_df = pd.read_csv(args.subtraction_df)

    if args.protocols:
        subtraction_df = subtraction_df[subtraction_df.protocol.isin(args.protocols)]

    if not args.cases:
        args.cases = ['0a', '0b', '0c']

    cases = args.cases
    dirnames_dict = { '0a': 'Case0a',
                      '0b': 'Case0b',
                      '0c': 'Case0b'}

    dirnames = [dirnames_dict[case] for case in cases]

    # Get fitting results (dict of dicts)
    results_dict = {}
    params_dfs = []
    for model in args.model_classes:
        results_dict[model] = {}
        for case, dirname in zip(cases, dirnames):
            fname = os.path.join(args.fitting_results,
                                 dirname,
                                 model,
                                 "combine_fitting_results",
                                 "combined_fitting_results.csv")

            params_df = pd.read_csv(fname)

            if args.protocols:
                params_df = params_df[params_df.protocol.isin(args.protocols)]

            if args.wells:
                params_df = params_df[params_df.well.isin(args.wells)].copy()

            params_df['protocol'] = ['staircaseramp1_2' if protocol ==
                                     'staircaseramp2' else protocol for
                                     protocol in params_df.protocol]

            params_dfs.append(params_df)
            results_dict[model][case] = params_df

    protocol_dict = {}
    for protocol in np.unique(list(itertools.chain(*[list(params_df.protocol.unique()) for params_df in params_dfs])) + args.validation_protocols):
        v_func, desc = get_ramp_protocol_from_json(protocol, os.path.join(args.data_directory, 'protocols'),
                                              args.experiment_name)

        times = np.loadtxt(os.path.join(args.data_directory,
                                        f"{args.experiment_name}-{protocol}-times.csv")).astype(np.float64).flatten()

        protocol_dict[protocol] = desc, times


    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    axs = setup_grid(fig, args)
    model_axs, model_label_axs, case_label_axs, colour_bar_ax = axs

    tasks = []
    for i, model_class in enumerate(args.model_classes):
        for j, case in enumerate(cases):
            sub_df = results_dict[model_class][case]
            tasks.append([model_class, case, sub_df, args, output_dir,
                          protocol_dict, case])

    with multiprocessing.Pool(min(len(tasks), args.no_cpus),
                              **multiprocessing_kws) as pool:
        res = pool.starmap(map_func, tasks)
    res = list(zip(tasks, res))

    do_summary_statistics(res)

    vmax = max([df.n_score.values.astype(np.float64).max() for _, df in res])
    vmin = min([df.n_score.values.astype(np.float64).min() for _, df in res])
    vlim = (vmin, vmax)

    cbar_kws = {
        'orientation': 'horizontal',
        'fraction': .65,
        'drawedges': False,
        'label': 'normalised RMSE',
    }

    done_colour_bar = False
    for task, prediction_df in res:
        model_class, case, sub_df, args, output_dir, protocol_dict, fitting_case = task

        if done_colour_bar:
            cbar_ax = None
        else:
            cbar_ax = colour_bar_ax
            done_colour_bar = True

        i = args.model_classes.index(model_class)
        j = cases.index(case)
        ax = model_axs[i, j]

        hm = do_heatmap(ax, model_class, case, sub_df, subtraction_df,
                        protocol_dict, vlim, args, prediction_df=prediction_df,
                        cbar_ax=cbar_ax,
                        cbar_kws=cbar_kws)

    fig.savefig(os.path.join(output_dir, "averaged_well_heatmaps"))
    fig.clf()
    axs = setup_grid(fig, args)
    model_axs, model_label_axs, case_label_axs, cbar_ax = axs

    individual_fig = plt.figure(figsize=args.figsize)
    individual_ax = individual_fig.subplots()

    # Now iterate over each well
    for well in subtraction_df.well.unique():
        if args.wells:
            if well not in args.wells:
                continue
        if well not in prediction_df.well.unique():
            continue

        for task, prediction_df in res:
            model_class, case, sub_df, args, output_dir, protocol_dict, fitting_case = task
            i = args.model_classes.index(model_class)
            j = cases.index(fitting_case)
            ax = model_axs[i, j]
            ax.cla()
            do_heatmap(ax, model_class, case, sub_df, subtraction_df,
                       protocol_dict, vlim, args, well=well,
                       prediction_df=prediction_df, cbar_ax=cbar_ax,
                       cbar_kws=cbar_kws)

            individual_fig.clf()
            individual_ax = individual_fig.subplots()
            # Do heatmap on individual plot with heatmap
            do_heatmap(individual_ax, model_class, case, sub_df, subtraction_df,
                       protocol_dict, vlim, args, well=well,
                       prediction_df=prediction_df, fontsize=11,
                       cbar=True)

            individual_fig.savefig(os.path.join(output_dir,
                                                f"{well}_{case}_{model_class}_heatmap"))

        for ax in model_axs.flatten():
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
        fig.savefig(os.path.join(output_dir,
                                 f"{well}_heatmaps"))
        fig.clf()
        axs = setup_grid(fig, args)
        model_axs, model_label_axs, case_label_axs, cbar_ax = axs
    plt.close(fig)


def do_summary_statistics(res):
    """ Summarise the prediction data frame
    - Average RMSE prediction error
    - Min/Max RMSE prediction error across wells
    - Min/Max RMSE prediction error across protocols
    - Best/worst performing well
    - Best/worst performing fitting protocol
    """

    rows = []
    for task, prediction_df in res:
        row = {}
        model_class, case, sub_df, args, output_dir, protocol_dict, fitting_case = task
        prediction_df.n_score = prediction_df.n_score.astype(np.float64)

        for well in prediction_df.well.unique():
            if not np.all(np.isfinite(prediction_df[prediction_df.well == well].n_score.values)):
                logging.warning(f"{model_class} {case} well {well} contains NaN predictions")

        row['average_n_score'] = prediction_df['n_score'].min()
        row['best_well_score'] = prediction_df.groupby('well')['n_score'].mean().min()
        row['best_well'] = prediction_df.groupby('well')['n_score'].mean().idxmin()
        row['worst_well_score'] = prediction_df.groupby('well')['n_score'].mean().max()
        row['worst_well'] = prediction_df.groupby('well')['n_score'].mean().idxmax()
        row['best_protocol_score'] = prediction_df.groupby('fitting_protocol')['n_score'].mean().min()
        row['best_protocol'] = prediction_df.groupby('fitting_protocol')['n_score'].mean().idxmin()
        row['worst_protocol_score'] = prediction_df.groupby('fitting_protocol')['n_score'].mean().max()
        row['worst_protocol'] = prediction_df.groupby('fitting_protocol')['n_score'].mean().idxmax()
        row['fitting_case'] = fitting_case
        row['model_class'] = model_class

        rows.append(row)

    df = pd.DataFrame.from_records(rows)
    df.to_csv(os.path.join(output_dir, "cv_summary.csv"))

    return df



def map_func(model_class, case, params_df, args, output_dir, protocol_dict,
             fitting_case):
    subtraction_df = pd.read_csv(args.subtraction_df)

    if args.protocols:
        subtraction_df = subtraction_df[subtraction_df.protocol.isin(args.protocols)]

    ax = None

    if fitting_case in ['I', 'II'] or args.use_raw_data:
        data_label = 'before'
    else:
        data_label = ''

    if not args.use_mock_data:
        args.model = model_class
        prediction_df = compute_predictions_df(params_df, output_dir,
                                               protocol_dict, fitting_case,
                                               args.reversal, subtraction_df,
                                               model_class=model_class,
                                               args=args,
                                               label=f"{model_class}_{case}_predictions",
                                               data_label=data_label,
                                               hybrid=False,
                                               strict=False,
                                               tolerances=(1e-6, 1e-6)
                                               )
    else:
        protocols = sorted(params_df.protocol.unique() )
        print(f"{model_class} {case} protocols are {protocols}")

        rows = [{'fitting_sweep': 0, 'prediction_sweep': 0, 'well': well,
                 'fitting_protocol': f_p, 'validation_protocol': v_p, 'RMSE':
                 np.random.uniform(3e2, 1e4)} for v_p in protocols for f_p in
                protocols for well in ['Z01', 'Z02', 'Z03']]
        prediction_df = pd.DataFrame.from_records(rows)
    return prediction_df


def do_spread_of_predictions(ax, model_class, fitting_case, params_df,
                             subtraction_df, validation_protocol, well=None):
    pass


def do_heatmap(ax, model_class, fitting_case, params_df, subtraction_df,
               protocol_dict, vlim, args, well=None, prediction_df=None, fontsize=8,
               **kws):

    if fitting_case in ['I', 'II'] or args.use_raw_data:
        data_label = 'before'
    else:
        data_label = ''

    if prediction_df is None:
        args.model = model_class
        prediction_df = compute_predictions_df(params_df, output_dir,
                                               protocol_dict, fitting_case,
                                               args.reversal, subtractions_df,
                                               model_class=model_class,
                                               label=f"{model_class}_{fitting_case}_predictions",
                                               data_label=data_label,
                                               hybrid=True,
                                               strict=False,
                                               tolerances=(1e-6, 1e-6),
                                               args=args)

    args.chrono_file

    chrono_fname = os.path.join(args.chrono_file)
    with open(chrono_fname, 'r') as fin:
        lines = fin.read().splitlines()
        protocol_order = [line.split(' ')[0] for line in lines]
        protocol_order.insert(1, 'staircaseramp1_sweep2')
        protocol_order.insert(-1, 'staircaseramp1_2_sweep2')

    def rename_staircase_func(row):
        f_protocol, v_protocol, f_sweep, v_sweep = [row[key] for key in ['fitting_protocol', 'validation_protocol', 'fitting_sweep', 'prediction_sweep']]

        if f_protocol in ['staircaseramp1', 'staircaseramp1_2', 'staircaseramp2'] and f_sweep == 1:
            row['fitting_protocol'] = str(f_protocol) + "_sweep2"

        if v_protocol in ['staircaseramp1', 'staircaseramp1_2', 'staircaseramp2'] and v_sweep == 1:
            row['validation_protocol'] = str(v_protocol) + "_sweep2"

        return row

    prediction_df = prediction_df[~prediction_df.fitting_protocol.isin(args.ignore_protocols)]

    prediction_df.fitting_sweep = prediction_df.fitting_sweep.astype(int)
    prediction_df.prediction_sweep = prediction_df.prediction_sweep.astype(int)

    prediction_df = prediction_df.apply(rename_staircase_func, axis=1)

    protocol_order = [p for p in protocol_order if p in prediction_df.validation_protocol.unique()]
    # Reorder and relabel protocols
    relabel_dict = {p: r"$d_{" f"{i}" r"}$" for i, p
                    in enumerate(protocol_order)}

    # Move longap to front

    if 'longap' in protocol_order:
        protocol_order.remove('longap')
        protocol_order.insert(0, 'longap')

    prediction_df['fitting_protocol'] = pd.Categorical(prediction_df['fitting_protocol'],
                                                       categories=protocol_order,
                                                       ordered=True)

    prediction_df['validation_protocol'] = pd.Categorical(prediction_df['validation_protocol'],
                                                          categories=protocol_order,
                                                          ordered=True)

    relabel_dict['staircaseramp1'] = r'$d_{1}^{(1)}$'
    relabel_dict['staircaseramp1_sweep2'] = r'$d_{1}^{(2)}$'
    relabel_dict['staircaseramp1_2'] = r'$d_{1}^{(3)}$'
    relabel_dict['staircaseramp1_2_sweep2'] = r'$d_{1}^{(4)}$'

    prediction_df.fitting_protocol = prediction_df.fitting_protocol.cat.rename_categories(relabel_dict)
    prediction_df.validation_protocol = prediction_df.validation_protocol.cat.rename_categories(relabel_dict)

    prediction_df.n_score = prediction_df.n_score.astype(np.float64)

    prediction_df.to_csv(os.path.join(output_dir,
                                      f"{model_class}_Case{fitting_case}_predictions.csv"))

    prediction_df = prediction_df.sort_values(['fitting_protocol', 'validation_protocol'])

    if ax is None:
        return prediction_df

    if well is not None:
        sub_df = prediction_df[prediction_df.well == well].copy()
        sub_df = sub_df.sort_values(['fitting_protocol', 'validation_protocol'])
        if len(sub_df.index) == 0:
            # logging.warning(f"do_heatmap: No predictions found for well {well}")
            return

    else:
        # Average across wells
        agg_dict = {'n_score': 'mean'}
        sub_df = prediction_df.groupby(['fitting_protocol', 'validation_protocol'],
                                       observed=True).agg(agg_dict).reset_index()

    vmin, vmax = vlim

    cmap = sns.cm.mako_r
    norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)

    pivot_df = sub_df.pivot(columns='fitting_protocol',
                            index='validation_protocol', values='n_score')

    pivot_df.dropna(axis=0, inplace=True, how='all')
    pivot_df.dropna(axis=1, inplace=True, how='all')

    relabelled_order = [relabel_dict[p] for p in protocol_order
                        if relabel_dict[p] in pivot_df.columns.to_list()]

    pivot_df = pivot_df[relabelled_order]

    if pivot_df.values.shape[0] == 0:
        logging.warning("No values in pivot_df")
        return None

    if 'cbar' not in kws:
        kws['cbar'] = False
        if 'cbar_ax' in kws:
            if kws['cbar_ax'] is not None:
                kws['cbar'] = True

    # Show mean score in title
    mean_training_score = sub_df[sub_df.fitting_protocol == sub_df.validation_protocol]['n_score'].values.astype(np.float64).mean()
    mean_validation_score = sub_df[sub_df.fitting_protocol != sub_df.validation_protocol]['n_score'].values.astype(np.float64).mean()

    ax.set_title(r'$\mathcal{E}_{\text{train}} = $' f"{mean_training_score:.2E}" + \
    r',\\' r'$\mathcal{E}_{\text{predict}} = $' + f"{mean_validation_score:.2E}",
                 fontsize=fontsize)

    hm = sns.heatmap(pivot_df, ax=ax, square=True, norm=norm,
                     cmap=cmap, **kws)

    autoAxis = ax.axis()
    rec = Rectangle(
        (autoAxis[0] - 0.05, autoAxis[3] - 0.05),
        (autoAxis[1] - autoAxis[0] + 0.1),
        1.1,
        fill=False,
        color='yellow',
        lw=.75
        )

    rec = ax.add_patch(rec)
    rec.set_clip_on(False)

    ax.set_ylabel('validation protocol')
    ax.set_xlabel('fitting protocol')

    hm.set_yticklabels(hm.get_yticklabels(), rotation=0)

    return hm


def setup_grid(fig, args):
    # Row for each model, a colorbar, and case labels
    no_rows = 2 + len(args.model_classes)

    # Coumn for each 'case' and labels
    no_cases = 3
    no_models = len(args.model_classes)
    no_columns = 1 + no_cases

    gs = GridSpec(no_rows, no_columns, figure=fig, height_ratios=[.15] + [1] *
                  no_models + [0.25], width_ratios=[.3] + no_cases*[1])

    model_label_axs = [fig.add_subplot(gs[i, 0]) for i in range(1, no_rows - 1)]
    case_label_axs = [fig.add_subplot(gs[0, i]) for i in range(1, no_columns)]
    colour_bar_ax = fig.add_subplot(gs[-1, :])
    model_axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(1, no_columns)]
                          for i in range(1, no_rows - 1)])

    relabel_models_dict = {
        'model2': 'C-O-I',
        'model3': 'Beattie',
        'model10': 'Kemp',
        'Wang': 'Wang'
    }

    for i, (label_ax, model_label) in enumerate(zip(model_label_axs, args.model_classes)):
        label = relabel_models_dict[model_label]
        label_ax.text(.5, .5, label, horizontalalignment='center',
                      verticalalignment='center', fontsize=12)

    case_labels = ['Case I', 'Case II', 'Case III']
    for i, (label_ax, case_label) in enumerate(zip(case_label_axs, case_labels)):
        case_label = case_labels[i]
        label_ax.text(.5, .5, case_label, horizontalalignment='center',
                      verticalalignment='center', fontsize=12)

    for ax in list(model_axs.flatten()) + model_label_axs + case_label_axs:
        ax.set_axis_off()

    return model_axs, model_label_axs, case_label_axs, colour_bar_ax


if __name__ == "__main__":
    main()
