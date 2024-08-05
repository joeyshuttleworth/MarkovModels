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
from markovmodels.fitting import get_best_params, compute_predictions_df, get_ensemble_of_predictions, make_prediction
from markovmodels.ArtefactModel import ArtefactModel
from markovmodels.utilities import setup_output_directory, get_data, get_all_wells_in_directory
from markovmodels.voltage_protocols import get_protocol_list, get_ramp_protocol_from_json, make_voltage_function_from_description
from markovmodels.voltage_protocols import remove_spikes, detect_spikes

multiprocessing_kws = {'maxtasksperchild': 1}

plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

rc('font', **{'size': 9})
# rc('text', usetex=True)
# rc('figure', dpi=400, facecolor=[0]*4)
# rc('axes', facecolor=[0]*4)
# rc('savefig', facecolor=[0]*4)
rc('figure', autolayout=True)

cbar_kws = {
    'orientation': 'horizontal',
    'fraction': .75,
    'drawedges': False,
    'label': 'NRMSE',
}

relabel_models_dict = {
    'model2': 'C-O-I',
    'model3': 'Beattie',
    'model10': 'Kemp',
    'Wang': 'Wang'
}

model_colour_dict = {
    'model2': '#a6cee3',
    'Wang': '#1f78b4',
    'model10': '#b2df8a',
    'model3': '#33a02c',
}


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
    parser.add_argument('--ignore_wells', nargs='+', default=['M06'], type=str)
    parser.add_argument('--fontsize', type=int)
    parser.add_argument('-w', '--wells', type=str, nargs='+')
    parser.add_argument('--removal_duration', type=float, default=5.0)
    parser.add_argument('--experiment_name', '-e', default='newtonrun4')
    parser.add_argument('--validation_protocols', default=['longap'], nargs='+')
    parser.add_argument('--figsize', '-f', nargs=2, type=float, default=[5.54, 6.5])
    parser.add_argument('--fig_title', '-t', default='')
    parser.add_argument('--nolegend', action='store_true')
    parser.add_argument('--dpi', '-d', default=500, type=int)
    parser.add_argument('--show_uncertainty', action='store_true')
    parser.add_argument('--shared_plot_limits', action='store_true')
    parser.add_argument('--no_voltage', action='store_true')
    parser.add_argument('--file_format', default='')
    parser.add_argument('--reversal', default=-89.5, type=float)
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
    if args.ignore_wells:
        subtraction_df = subtraction_df[~subtraction_df.well.isin(args.ignore_wells)]

    if not args.cases:
        args.cases = ['0d', 'II']

    cases = args.cases
    dirnames_dict = { '0a': 'Case0a',
                      '0b': 'Case0b',
                      '0c': 'Case0b',
                      'II': 'CaseII',
                      '0d': 'Case0d',
                     }

    dirnames = [dirnames_dict[case] for case in cases]

    voltage_func = make_voltage_function_from_description()

    # Get fitting results (dict of dicts)
    results_dict = {}
    params_dfs = []
    params_df_dict = {}
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
                params_df = params_df[params_df.protocol.isin(args.protocols)].copy()

            if args.wells:
                params_df = params_df[params_df.well.isin(args.wells)].copy()

            params_df = params_df[~params_df.well.isin(args.ignore_wells)].copy()

            params_df['protocol'] = ['staircaseramp1_2' if protocol ==
                                     'staircaseramp2' else protocol for
                                     protocol in params_df.protocol]

            params_dfs.append(params_df)
            params_df_dict[(model, case)] = params_df
            results_dict[model][case] = params_df

    protocol_dict = {}
    for protocol in np.unique(list(itertools.chain(*[list(params_df.protocol.unique()) for params_df in params_dfs])) + args.validation_protocols):
        v_func, desc = get_ramp_protocol_from_json(protocol, os.path.join(args.data_directory, 'protocols'),
                                              args.experiment_name)

        times = np.loadtxt(os.path.join(args.data_directory,
                                        f"{args.experiment_name}-{protocol}-times.csv")).astype(np.float64).flatten()
        protocol_dict[protocol] = desc, times

    if args.figsize:
        individual_fig_height = 3.0
        individual_plot_figsize =  [args.figsize[0], individual_fig_height]

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
    vmax = max([df.n_score.values.astype(np.float64).max() for _, df in res])
    vmin = min([df.n_score.values.astype(np.float64).min() for _, df in res])
    vlim = (vmin, vmax)

    do_summary_statistics(res)

    # best_worst_fig_plot_figsize = args.figsize.copy()
    # best_worst_fig_plot_figsize[1] = 7.5

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    protocol_order = define_protocol_order(args.chrono_file)

    for task, prediction_df in res:
        model_class, case, sub_df, args, output_dir, protocol_dict, fitting_case = task
        # Compare best and worst wells
        fig.clf()
        # axs = fig.subplots(1, 3, width_ratios=[1, 1, 0.1])
        heatmap_axs, prediction_axs, voltage_ax  = setup_best_worst_fig(fig)
        best_ax, worst_ax, cbar_ax = heatmap_axs

        if fitting_case in ['I', 'II'] or args.use_raw_data:
            data_label = 'before'
        else:
            data_label = ''

        prediction_df = prediction_df[~prediction_df.well.isin(args.ignore_wells)]

        agg_dict = {'n_score': 'mean'}
        best_well = prediction_df.groupby('well').agg(agg_dict).idxmin()['n_score']
        worst_well = prediction_df.groupby('well').agg(agg_dict).idxmax()['n_score']
        print(f"best well: {best_well}")
        print(f"worst well: {worst_well}")

        # Find worst prediction in worst wells
        worst_well_predictions = prediction_df[prediction_df.well == worst_well].copy()
        worst_prediction = worst_well_predictions.groupby(['fitting_protocol', 'validation_protocol'])['n_score'].agg('max').idxmax()

        fitting_protocol, validation_protocol = worst_prediction
        sweep = 0

        # Plot voltage of worst prediction
        if not args.use_mock_data:
            worst_data, vp = get_data(worst_well, validation_protocol,
                                    args.data_directory, args.experiment_name, sweep=sweep)

            best_data, _ = get_data(worst_well, validation_protocol,
                                    args.data_directory, args.experiment_name, sweep=sweep)

            desc = vp.get_all_sections()
            desc = np.vstack((desc, [[desc[-1, 1], np.inf, -80.0, -80.0]]))
            times_fname = os.path.join(args.data_directory,
                                    f"{args.experiment_name}-{validation_protocol}-times.csv")
            times = np.loadtxt(times_fname).flatten()
            Vcmd = np.array([voltage_func(t, protocol_description=desc) for t in times])

            voltage_ax.plot(times * 1e-3, Vcmd, color='black')
            prediction_axs[0].plot(times * 1e-3, worst_data, color='red', alpha=.5, lw=.75)
            prediction_axs[1].plot(times * 1e-3, best_data, color='red', alpha=.5, lw=.75)

            worst_pred, _ = make_prediction(model_class, args, worst_well,
                                            validation_protocol, sweep,
                                            fitting_protocol, sweep, sub_df,
                                            subtraction_df, case,
                                            args.reversal, protocol_dict,
                                            worst_data, Vcmd,
                                            label=data_label,
                                            return_states=True )

            best_pred, _ = make_prediction(model_class, args, best_well,
 validation_protocol, sweep,
                                           fitting_protocol, sweep, sub_df,
                                           subtraction_df, case,
                                           args.reversal, protocol_dict,
                                           best_data, Vcmd,
                                           label=data_label,
                                           return_states=True )

            prediction_axs[0].plot(times * 1e-3, worst_pred, lw=.75, alpha=.5)
            prediction_axs[1].plot(times * 1e-3, best_pred, lw=.75, alpha=.5)

        best_worst_cbar_kws = cbar_kws.copy()
        best_worst_cbar_kws['orientation'] = 'vertical'
        best_worst_cbar_kws['label'] = ''

        do_heatmap(best_ax, model_class, case, sub_df.copy(), subtraction_df,
                   protocol_dict, vlim, args, well=best_well,
                   prediction_df=prediction_df, cbar=False)

        do_heatmap(worst_ax, model_class, case, sub_df.copy(), subtraction_df,
                   protocol_dict, vlim, args, well=worst_well,
                   prediction_df=prediction_df, cbar_ax=cbar_ax,
                   cbar_kws=best_worst_cbar_kws)

        # Highlight worst cell
        autoAxis = worst_ax.axis()
        fitting_protocol_i = protocol_order.index(fitting_protocol)
        validation_protocol_i = protocol_order.index(validation_protocol)

        no_protocols = len(protocol_order)
        rec = Rectangle(
            (autoAxis[0] - 0.05 + fitting_protocol_i,
             autoAxis[3] - 0.05 + validation_protocol_i),
            1.1,
            1.1,
            fill=False,
            color='yellow',
            lw=.75
            )

        rec_1 = worst_ax.add_patch(rec)
        rec_1.set_clip_on(False)

        autoAxis = best_ax.axis()
        rec = Rectangle(
            (autoAxis[0] - 0.05 + fitting_protocol_i,
             autoAxis[3] - 0.05 + validation_protocol_i),
            1.1, 1.1,
            fill=False,
            color='yellow',
            lw=.75
            )

        rec_2 = best_ax.add_patch(rec)
        rec_2.set_clip_on(False)

        for ax in prediction_axs:
            ax.set_xticklabels([])

        cbar_ax.set_title('NRMSE')

        mean_training_score = prediction_df[(prediction_df.fitting_protocol == prediction_df.validation_protocol)\
                                   & (prediction_df.well == best_well)]['n_score'].values.astype(np.float64).mean()

        mean_validation_score = prediction_df[(prediction_df.fitting_protocol
                                               != prediction_df.validation_protocol)\
                                              &(prediction_df.well == best_well)\
                                              &(prediction_df.fitting_sweep == prediction_df.prediction_sweep)
                                              ]['n_score'].values.astype(np.float64).mean()


        best_well_title = f"{best_well} " + '\n' \
            + r'$\mathcal{E}_{\mathrm{train}} = $' f"{mean_training_score:.2E}" + \
            ',\n' r'$\mathcal{E}_{\mathrm{predict}} = $' + f"{mean_validation_score:.2E}"

        best_ax.set_title(best_well_title)

        mean_training_score = prediction_df[(prediction_df.fitting_protocol == prediction_df.validation_protocol)\
                                     & (prediction_df.well == worst_well)]['n_score'].values.astype(np.float64).mean()
        mean_validation_score = prediction_df[(prediction_df.fitting_protocol
                                               != prediction_df.validation_protocol)\
                                              &(prediction_df.well == worst_well)\
                                              &(prediction_df.fitting_sweep == prediction_df.prediction_sweep)
                                              ]['n_score'].values.astype(np.float64).mean()
        worst_well_title = f"{worst_well} " + '\n' \
            + r'$\mathcal{E}_{\mathrm{train}} = $' f"{mean_training_score:.2E}" + \
            ',\n' r'$\mathcal{E}_{\mathrm{predict}} = $' + f"{mean_validation_score:.2E}"

        worst_ax.set_title(worst_well_title)
        worst_ax.axis('off')
        # worst_ax.set_xticks([])
        worst_ax.set_yticks([])
        best_ax.tick_params(axis='x', labelrotation=90.0)
        worst_ax.tick_params(axis='x', labelrotation=90.0)

        fig.savefig(os.path.join(output_dir, f"best_worst_{case}_{model_class}_heatmap_best_worst"))
        fig.clf()

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    axs = setup_grid(fig, args)
    model_axs, model_label_axs, case_label_axs, colour_bar_ax = axs

    individual_fig = plt.figure(figsize=individual_plot_figsize, constrained_layout=True)
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

        individual_fig.clf()
        individual_ax, individual_cbar_ax = individual_fig.subplots(1, 2, width_ratios=[1, 0.1])
        individual_cbar_kws = cbar_kws.copy()
        individual_cbar_kws['orientation'] = 'vertical'

        # Do heatmap on individual plot with heatmap
        do_heatmap(individual_ax, model_class, case, sub_df, subtraction_df,
                   protocol_dict, vlim, args,
                   prediction_df=prediction_df,
                   cbar_ax=individual_cbar_ax,
                   cbar=True, cbar_kws=individual_cbar_kws)

        individual_fig.savefig(os.path.join(output_dir,
                                            f"average_{case}_{model_class}_heatmap"))

    fig.savefig(os.path.join(output_dir, "averaged_well_heatmaps"))
    fig.clf()

    model_axs, colour_bar_ax, _\
        = setup_grid_single_case(fig, args)
    done_colour_bar = False

    for task, prediction_df in res:
        model_class, case, sub_df, args, output_dir, protocol_dict, fitting_case = task
        if fitting_case != 'II':
            continue

        if done_colour_bar:
            cbar_ax = None
            this_cbar_kws = {}
        else:
            cbar_ax = colour_bar_ax
            this_cbar_kws = cbar_kws.copy()
            this_cbar_kws['orientation'] = 'vertical'
            this_cbar_kws['label'] = ''
            print(this_cbar_kws)
            done_colour_bar = True

        i = args.model_classes.index(model_class)
        j = cases.index(case)
        ax = model_axs[i]

        ax.set_label(relabel_models_dict[model_class])

        hm = do_heatmap(ax, model_class, case, sub_df, subtraction_df,
                        protocol_dict, vlim, args, prediction_df=prediction_df,
                        cbar_ax=cbar_ax,
                        cbar_kws=this_cbar_kws)


        mean_training_score = prediction_df[(prediction_df.fitting_protocol == prediction_df.validation_protocol)]['n_score'].values.astype(np.float64).mean()
        mean_validation_score = prediction_df[(prediction_df.fitting_protocol != prediction_df.validation_protocol)]['n_score'].values.astype(np.float64).mean()

        ax.set_title(r'$\mathcal{E}_{\text{train}} = $' f"{mean_training_score:.2E}" + \
                     ",\n" r'$\mathcal{E}_{\text{predict}} = $' + f"{mean_validation_score:.2E}")


        validation_protocol = 'longap'
        sweep = 0

    colour_bar_ax.set_title('NRMSE')

    for ax in model_axs:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

    fig.savefig(os.path.join(output_dir, 'CaseII_heatmap_comparison'))
    fig.clf()

    axs = setup_grid(fig, args)
    model_axs, model_label_axs, case_label_axs, cbar_ax = axs

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
                       cbar_kws=cbar_kws.copy())

            individual_fig.clf()
            individual_ax, individual_cbar_ax = individual_fig.subplots(1, 2, width_ratios=[1, 0.1])
            # Do heatmap on individual plot with heatmap
            individual_cbar_kws = cbar_kws.copy()
            individual_cbar_kws['orientation'] = 'vertical'
            do_heatmap(individual_ax, model_class, case, sub_df, subtraction_df,
                       protocol_dict, vlim, args, well=well,
                       prediction_df=prediction_df,
                       cbar_kws=individual_cbar_kws,
                       cbar=True,
                       cbar_ax=individual_cbar_ax)

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

    if fitting_case in ['I', 'II', '0d'] or args.use_raw_data:
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
        if args.ignore_wells:
            prediction_df = prediction_df[~prediction_df.well.isin(args.ignore_wells)]

    else:
        protocols = sorted(params_df.protocol.unique() )
        rows = [{'fitting_sweep': 0, 'prediction_sweep': 0, 'well': well,
                 'fitting_protocol': f_p, 'validation_protocol': v_p, 'RMSE':
                 np.random.uniform(3e2, 1e4)} for v_p in protocols for f_p in
                protocols for well in ['Z01', 'Z02', 'Z03']]
        prediction_df = pd.DataFrame.from_records(rows)
        prediction_df['n_score'] = prediction_df['RMSE']

    return prediction_df


def define_protocol_order(chrono_fname):
    with open(chrono_fname, 'r') as fin:
        lines = fin.read().splitlines()
        protocol_order = [line.split(' ')[0] for line in lines]
        protocol_order.insert(1, 'staircaseramp1_sweep2')
        protocol_order.append('staircaseramp1_2_sweep2')
    return protocol_order


def do_heatmap(ax, model_class, fitting_case, params_df, subtraction_df,
               protocol_dict, vlim, args, well=None, prediction_df=None,
               **kws):

    params_df = params_df.copy()


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

    chrono_fname = os.path.join(args.chrono_file)
    protocol_order = define_protocol_order(chrono_fname)

    def rename_staircase_func(row):
        f_protocol, v_protocol, f_sweep, v_sweep = [row[key] for key in ['fitting_protocol', 'validation_protocol', 'fitting_sweep', 'prediction_sweep']]

        if f_protocol in ['staircaseramp1', 'staircaseramp1_2', 'staircaseramp2'] and f_sweep == 1:
            row['fitting_protocol'] = str(f_protocol) + "_sweep2"

        if v_protocol in ['staircaseramp1', 'staircaseramp1_2', 'staircaseramp2'] and v_sweep == 1:
            row['validation_protocol'] = str(v_protocol) + "_sweep2"

        return row

    prediction_df = prediction_df[~prediction_df.fitting_protocol.isin(args.ignore_protocols)]
    prediction_df = prediction_df[~prediction_df.well.isin(args.ignore_wells)]

    prediction_df.fitting_sweep = prediction_df.fitting_sweep.astype(int)
    prediction_df.prediction_sweep = prediction_df.prediction_sweep.astype(int)

    prediction_df = prediction_df.apply(rename_staircase_func, axis=1)

    # Reorder and relabel protocols
    relabel_dict = {p: r"$d_{" f"{i}" r"}$" for i, p
                    in enumerate(protocol_order)}

    # Move longap to front
    if 'longap' in protocol_order:
        protocol_order.remove('longap')
        protocol_order.insert(0, 'longap')

    relabel_dict['staircaseramp1'] = r'$d_{1}^{(1)}$'
    relabel_dict['staircaseramp1_sweep2'] = r'$d_{1}^{(2)}$'
    relabel_dict['staircaseramp1_2'] = r'$d_{1}^{(3)}$'
    relabel_dict['staircaseramp1_2_sweep2'] = r'$d_{1}^{(4)}$'

    prediction_df['fitting_protocol'] = pd.Categorical(prediction_df['fitting_protocol'],
                                                       categories=protocol_order,
                                                       ordered=True)

    prediction_df['validation_protocol'] = pd.Categorical(prediction_df['validation_protocol'],
                                                          categories=protocol_order,
                                                          ordered=True)

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
    ",\n" r'$\mathcal{E}_{\text{predict}} = $' + f"{mean_validation_score:.2E}")

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

    # Make tick label text smaller
    ax.tick_params(axis='both', labelsize=8)

    return hm


def setup_grid(fig, args):
    # Row for each model, a colorbar, and case labels
    no_rows = 2 + len(args.model_classes)

    no_cases = 2
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
                      verticalalignment='center')

    case_labels = ['Case IV', 'Case V']
    for i, (label_ax, case_label) in enumerate(zip(case_label_axs, case_labels)):
        case_label = case_labels[i]
        label_ax.text(.5, .5, case_label, horizontalalignment='center',
                      verticalalignment='center')

    for ax in list(model_axs.flatten()) + model_label_axs + case_label_axs:
        ax.set_axis_off()

    return model_axs, model_label_axs, case_label_axs, colour_bar_ax


def setup_grid_single_case(fig, args):
    # Row for each model, a colorbar, and case labels
    no_models = len(args.model_classes)
    no_columns = 3
    no_rows = 4

    gs = GridSpec(no_rows, no_columns, figure=fig, width_ratios=[1, 1, .1],
                  height_ratios=[0.1, 1, 1, 0.1])

    colour_bar_ax = fig.add_subplot(gs[:, -1])

    model_axs = [fig.add_subplot(gs[1, i]) for i in range(2)] \
        + [fig.add_subplot(gs[2, i]) for i in range(2)]

    caption_axs = np.array([fig.add_subplot(gs[0, i]) for i in range(2)] \
                         + [fig.add_subplot(gs[-1, i]) for i in range(2)]).flatten()

    for i, (ax, model) in enumerate(zip(caption_axs, args.model_classes)):
        cap = relabel_models_dict[model]
        ax.set_axis_off()
        ax.text(.5, .5, cap, fontsize='12' ,
                fontweight='bold',
                horizontalalignment='center')
    model_axs[0].set_axis_on()

    # for ax in model_axs:
    #     ax.spines[['top', 'right']].set_visible(False)

    # colour_bar_ax.set_axis_off()
    return model_axs, colour_bar_ax, caption_axs


def setup_best_worst_fig(fig):
    no_models = len(args.model_classes)
    no_columns = 3
    no_rows = 4

    gs = GridSpec(no_rows, no_columns, figure=fig, width_ratios=[1, 1, 0.05],
                  height_ratios=[0.5, 0.5, 0.25, 1.1]
                  )

    heatmap_axs = [fig.add_subplot(gs[-1, i]) for i in range(no_columns)]
    prediction_axs = [fig.add_subplot(gs[i, :]) for i in range(2)]
    voltage_ax = fig.add_subplot(gs[2, :])

    voltage_ax.set_xlabel(r'$t$ (ms)')

    for ax in prediction_axs:
        ax.set_ylabel(r'$I_\mathrm{Kr} (pA)$')

    for ax in prediction_axs + [voltage_ax]:
        ax.spines[['top', 'right']].set_visible(False)

    prediction_axs[0].set_title('a', fontweight='bold', loc='left')
    prediction_axs[1].set_title('b', fontweight='bold', loc='left')

    voltage_ax.set_title('c', fontweight='bold', loc='left')

    heatmap_axs[0].set_title('d', fontweight='bold', loc='left')
    heatmap_axs[1].set_title('e', fontweight='bold', loc='left')

    return heatmap_axs, prediction_axs, voltage_ax

if __name__ == "__main__":
    main()
