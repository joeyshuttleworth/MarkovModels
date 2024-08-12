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

from cycler import cycler
from matplotlib import rc

import markovmodels
from markovmodels.model_generation import make_model_of_class
from markovmodels.fitting import get_best_params, compute_predictions_df, get_ensemble_of_predictions, make_prediction, fit_leak_parameters_with_artefact, find_V_off
from markovmodels.ArtefactModel import ArtefactModel, no_artefact_parameters
from markovmodels.utilities import setup_output_directory, get_data, get_all_wells_in_directory
from markovmodels.voltage_protocols import get_protocol_list, get_ramp_protocol_from_json, make_voltage_function_from_description
from markovmodels.voltage_protocols import remove_spikes, detect_spikes

multiprocessing_kws = {'maxtasksperchild': 1}

plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams["text.usetex"] = True

rc('font', **{'size': 8})
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

relabel_case_dict = {
    '0a': 'Case I',
    '0b': 'Case II',
    '0c': 'Case III',
    '0d': 'Case IV',
    'II': 'Case V'
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
    parser.add_argument('--ignore_validation_protocols', nargs='+', default=[], type=str)
    parser.add_argument('--ignore_fitting_protocols', nargs='+', default=['longap'], type=str)
    parser.add_argument('--ignore_wells', nargs='+', default=['M06'], type=str)
    parser.add_argument('--fontsize', type=int)
    parser.add_argument('-w', '--wells', type=str, nargs='+')
    parser.add_argument('--removal_duration', type=float, default=5.0)
    parser.add_argument('--experiment_name', '-e', default='newtonrun4')
    parser.add_argument('--validation_protocols', default=['longap'], nargs='+')
    parser.add_argument('--figsize', '-f', nargs=2, type=float, default=[5.54, 7.2])
    parser.add_argument('--fig_title', '-t', default='')
    parser.add_argument('--nolegend', action='store_true')
    parser.add_argument('--dpi', '-d', default=500, type=int)
    parser.add_argument('--show_uncertainty', action='store_true')
    parser.add_argument('--shared_plot_limits', action='store_true')
    parser.add_argument('--no_voltage', action='store_true')
    parser.add_argument('--file_format', default='')
    parser.add_argument('--reversal', default=-89.83, type=float)
    parser.add_argument('--output', '-o')
    parser.add_argument('--no_cpus', '-c', default=1, type=int)
    parser.add_argument('--model_classes', nargs='+')
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--dont_plot_predictions', action='store_true')

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

    if not args.cases:
        args.cases = ['0a', '0b', '0c']

    cases = args.cases
    dirnames_dict = { '0a': 'Case0a',
                      '0b': 'Case0b',
                      '0c': 'Case0b',
                      '0d': 'Case0d',
                      'II': 'CaseII'
                     }

    dirnames = [dirnames_dict[case] for case in cases]

    voltage_func = make_voltage_function_from_description()

    # Get fitting results (dict of dicts)
    results_dict = {}
    params_dfs = []
    params_df_dict = {}

    if args.ignore_validation_protocols:
        args.validation_protocols = [p for p in args.validation_protocols\
                                     if p not in args.ignore_validation_protocols]

    for model in args.model_classes:
        results_dict[model] = {}
        for case, dirname in zip(cases, dirnames):
            fname = os.path.join(args.fitting_results,
                                 dirname,
                                 model,
                                 "combine_fitting_results",
                                 "combined_fitting_results.csv")

            params_df = pd.read_csv(fname)
            param_labels = make_model_of_class(model).get_parameter_labels()
            params_df = get_best_params(params_df, param_labels=param_labels)

            params_df.to_csv(os.path.join(output_dir, f"{model}_{case}_best_params.csv"))

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
        v_func, desc = get_ramp_protocol_from_json(protocol,
                                                   os.path.join(args.data_directory,
                                                                'protocols'),
                                                   args.experiment_name)



        times = np.loadtxt(os.path.join(args.data_directory,
                                        f"{args.experiment_name}-{protocol}-times.csv")).astype(np.float64).flatten()
        desc = np.vstack((desc, [[desc[-1, 1], np.inf, -80.0, -80.0]]))

        protocol_dict[protocol] = desc, times

    if 'I' in args.cases or 'II' in args.cases\
       and len([p for p in args.validation_protocols if p not in args.ignore_validation_protocols]) > 0:
        new_artefact_params_df = fit_artefact_parameters(params_df, args.validation_protocols,
                                                     protocol_dict, args)
        print(f"Fitted artefact parameters {new_artefact_params_df}")

        for model in args.model_classes:
            for case in args.cases:
                if case not in ['I', 'II']:
                    continue

                params_df = results_dict[model][case]

                _artefact_params_df = new_artefact_params_df.copy()

                for lab in [lab for lab in params_df.columns\
                            if lab not in _artefact_params_df.columns]:
                    _artefact_params_df[lab] = 0.0

                params_df = pd.concat([params_df,
                                       _artefact_params_df], ignore_index=True,
                                      axis=0)
                results_dict[model][case] = params_df

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
        heatmap_axs, prediction_axs, voltage_axs  = setup_best_worst_fig(fig)
        best_ax, worst_ax, cbar_ax = heatmap_axs

        if case in ['I', 'II', '0d'] or args.use_raw_data:
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
        worst_prediction = worst_well_predictions.groupby(['fitting_protocol', 'validation_protocol', 'fitting_sweep', 'prediction_sweep'])['n_score'].agg('max').idxmax()

        best_well_predictions = prediction_df[prediction_df.well == worst_well].copy()
        best_prediction = best_well_predictions.groupby(['fitting_protocol', 'validation_protocol',
                                                         'fitting_sweep', 'prediction_sweep'])['n_score'].agg('max').idxmin()

        fitting_protocol, validation_protocol, fit_sweep, predict_sweep\
            = worst_prediction
 
        voltage_axs[0].set_title(get_protocol_label(protocol_order, validation_protocol,
                                                    fit_sweep))

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

        # Plot voltage of worst prediction
        if not args.use_mock_data:
            worst_data, vp = get_data(worst_well, validation_protocol,
                                      args.data_directory,
                                      args.experiment_name, sweep=predict_sweep,
                                      label=data_label)

            desc = vp.get_all_sections()
            desc = np.vstack((desc, [[desc[-1, 1], np.inf, -80.0, -80.0]]))
            times_fname = os.path.join(args.data_directory,
                                    f"{args.experiment_name}-{validation_protocol}-times.csv")
            times = np.loadtxt(times_fname).flatten()
            Vcmd = np.array([voltage_func(t, protocol_description=desc) for t in times])

            voltage_axs[0].plot(times * 1e-3, Vcmd, color='black', lw=1)
            params_df = results_dict[model_class][case].copy()
            worst_pred, _ = make_prediction(model_class, args, worst_well,
                                            validation_protocol, predict_sweep,
                                            fitting_protocol, fit_sweep, params_df,
                                            subtraction_df.copy(), case,
                                            args.reversal, protocol_dict,
                                            worst_data, Vcmd,
                                            label=data_label,
                                            return_states=True )

            prediction_axs[0].plot(times * 1e-3, worst_data, alpha=.5, color='red',
                                   lw=.6)
            prediction_axs[0].plot(times * 1e-3, worst_pred, alpha=.5, lw=.9)

            # Highlight worst cell
            autoAxis = worst_ax.axis()

            _protocol_order = [prot for prot in protocol_order if prot\
                               not in args.ignore_validation_protocols]

            validation_protocols = [
                prot for prot in args.validation_protocols
                if prot not in args.ignore_validation_protocols
                                    ]

            fitting_protocol_i = _protocol_order.index(fitting_protocol) + int(fit_sweep)
            validation_protocol_i = _protocol_order.index(validation_protocol) + len(args.validation_protocols) + int(predict_sweep)

            for validation_protocol in args.validation_protocols:
                if validation_protocol_i > protocol_order.index(validation_protocol):
                    validation_protocol_i -= 1

                if fitting_protocol_i > protocol_order.index(validation_protocol):
                    fitting_protocol_i -= 1

            no_protocols = len(protocol_order)
            rec = Rectangle(
                #d6 is at the front of the order but absent from the heatmap
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

            fitting_protocol, validation_protocol, \
                fit_sweep, predict_sweep = best_prediction
            best_data, vp = get_data(best_well, validation_protocol,
                                     args.data_directory, args.experiment_name,
                                     label=data_label,
                                     sweep=predict_sweep)
            desc = vp.get_all_sections()

            fitting_protocol, validation_protocol, fit_sweep, predict_sweep \
                = best_prediction
            times_fname = os.path.join(args.data_directory,
                                    f"{args.experiment_name}-{validation_protocol}-times.csv")
            times = np.loadtxt(times_fname).flatten()
            Vcmd = np.array([voltage_func(t, protocol_description=desc) for t in times])

            voltage_axs[1].set_title(get_protocol_label(protocol_order, validation_protocol,
                                                        fit_sweep))

            params_df = results_dict[model_class][case].copy()
            best_pred, _ = make_prediction(model_class, args, best_well,
                                           validation_protocol, predict_sweep,
                                           fitting_protocol, fit_sweep, params_df,
                                           subtraction_df.copy(), case,
                                           args.reversal, protocol_dict,
                                           best_data, Vcmd,
                                           label=data_label,
                                           return_states=True
                                           )

            prediction_axs[1].plot(times * 1e-3, best_data, alpha=.5, color='red',
                                   lw=.6)
            prediction_axs[1].plot(times * 1e-3, best_pred, alpha=.5, lw=.9)
            voltage_axs[1].plot(times * 1e-3, Vcmd, color='black', lw=1)

            fitting_protocol_i = _protocol_order.index(fitting_protocol) + int(fit_sweep)
            validation_protocol_i = _protocol_order.index(validation_protocol) + len(args.validation_protocols) + int(predict_sweep)

            for protocol in args.validation_protocols:
                if validation_protocol_i > protocol_order.index(protocol):
                    validation_protocol_i -= 1

                    if fitting_protocol_i > protocol_order.index(protocol):
                        fitting_protocol_i -= 1

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

        for ax in voltage_axs:
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
            + r'$\mathcal{E}_{\mathrm{fit}} = $' f"{mean_training_score:.2f}" + \
            ',\n' r'$\mathcal{E}_{\mathrm{predict}} = $' + f"{mean_validation_score:.2f}"

        best_ax.set_title(best_well_title)

        mean_training_score = prediction_df[(prediction_df.fitting_protocol == prediction_df.validation_protocol)\
                                     & (prediction_df.well == worst_well)]['n_score'].values.astype(np.float64).mean()
        mean_validation_score = prediction_df[(prediction_df.fitting_protocol
                                               != prediction_df.validation_protocol)\
                                              &(prediction_df.well == worst_well)\
                                              &(prediction_df.fitting_sweep == prediction_df.prediction_sweep)
                                              ]['n_score'].values.astype(np.float64).mean()
        worst_well_title = f"{worst_well} " + '\n' \
            + r'$\mathcal{E}_{\mathrm{fit}} = $' f"{mean_training_score:.2f}" + \
            ',\n' r'$\mathcal{E}_{\mathrm{predict}} = $' + f"{mean_validation_score:.2f}"

        worst_ax.set_title(worst_well_title)
        # worst_ax.axis('off')
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

    comparison_fig = plt.figure(figsize=[args.figsize[0], 6.5])
    # Plot Case III only
    model_axs, colour_bar_ax = setup_grid_single_case(comparison_fig, args)
    done_colour_bar = False

    for task, prediction_df in res:
        model_class, case, sub_df, args, output_dir, protocol_dict, fitting_case = task
        if fitting_case != '0c':
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

        model_name = relabel_models_dict[model_class]
        ax.set_title(r'\textbf{' + model_name + r'}' + "\n"+ r'$\mathcal{E}_{\mathrm{fit}} = $' f"{mean_training_score:.2f}" + \
                     ",\n" r'$\mathcal{E}_{\mathrm{predict}} = $' + f"{mean_validation_score:.2f}")


        validation_protocol = 'longap'
        sweep = 0

    colour_bar_ax.set_title('NRMSE')

    for ax in model_axs[1:]:
        ax.set_xticklabels([])
        ax.set_xticklabels([])
        ax.set_xlabel('')
        ax.set_ylabel('')

    comparison_fig.savefig(os.path.join(output_dir, 'Case0c_heatmap_comparison'))
    comparison_fig.clf()
    plt.close(comparison_fig)

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
        # for ax in model_axs.flatten():
            # ax.xaxis.set_visible(False)
            # ax.yaxis.set_visible(False)
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

    ax = None

    if fitting_case in ['I', 'II', '0d'] or args.use_raw_data:
        data_label = 'before'
    else:
        data_label = ''

    if not args.use_mock_data:
        args.model = model_class

        if fitting_case in ['I', 'II']:
            tolerances = (1e-6, 1e-6)
        else:
            tolerances = (1e-8, 1e-8)

        if args.protocols:
            params_df = params_df[params_df.protocol.isin(args.protocols)].copy()

        prediction_df = compute_predictions_df(params_df, output_dir,
                                               protocol_dict, fitting_case,
                                               args.reversal, subtraction_df,
                                               model_class=model_class,
                                               args=args,
                                               label=f"{model_class}_{case}_predictions",
                                               data_label=data_label,
                                               hybrid=False,
                                               strict=False,
                                               tolerances=tolerances,
                                               plot=not args.dont_plot_predictions,
                                               validation_protocols=args.validation_protocols
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


def fit_artefact_parameters(params_df, protocols, protocol_dict, args):
    subtraction_df = pd.read_csv(args.subtraction_df)
    data_label = 'before'

    params_df = params_df.copy()

    V_off_model_class = 'model3'

    V_off_model = ArtefactModel(make_model_of_class(V_off_model_class))
    voltage_func = V_off_model.channel_model.voltage

    params = make_model_of_class(V_off_model_class).get_default_parameters()

    solver_current = V_off_model.make_hybrid_solver_current(hybrid=False,
                                                            njitted=False,
                                                            strict=False,
                                                            return_var='I_out')
    solver_states = V_off_model.make_hybrid_solver_states(hybrid=False,
                                                          njitted=False,
                                                          strict=False
                                                          )
    new_rows = []
    for protocol in protocols:
        # Protocol we use for simulation
        desc, times = protocol_dict[protocol]

        if protocol in args.ignore_validation_protocols:
            continue

        for well in params_df.well.unique():
            for sweep in params_df.sweep.unique():
                if params_df.set_index(['well', 'protocol', 'sweep']).index.isin([(well, protocol, sweep)]).any():
                   logging.warning(f"{[well, protocol, sweep]} not in df")
                   continue

                if not subtraction_df.set_index(['well', 'protocol', 'sweep']).index.isin([(well, protocol, sweep)]).any():
                   continue

                row = subtraction_df.set_index(['well', 'protocol', 'sweep']).sort_index().loc[(well, protocol, sweep)]
                Rseries, Cm = row[['Rseries', 'Cm']]
                Rseries = Rseries * 1e-9
                Cm = Cm * 1e9

                data, _ = get_data(well, protocol,
                                   args.data_directory,
                                   args.experiment_name, sweep=sweep,
                                   label=data_label)
                V_off_initial_params = np.concatenate([
                    params.copy(),
                    [args.reversal, .0, .0, .0, .0, .0, Cm, Rseries]
                ]).flatten()

                V_off, success = find_V_off(desc, times, data,
                                            V_off_model_class,
                                            V_off_initial_params,
                                            args.reversal,
                                            data_label=data_label,
                                            a_solver_current=solver_current,
                                            a_solver_states=solver_states )

                voltages = np.array([voltage_func(t, protocol_description=desc)
                                     for t in times])

                V_off_initial_params[-3] = V_off

                gleak, Eleak = \
                    fit_leak_parameters_with_artefact(V_off_model,
                                                      desc.astype(np.float64),
                                                      times, data, voltages,
                                                      default_parameters=V_off_initial_params,
                                                      a_solver_current=solver_current
                                                      )
                param_dict = {
                    'E_Kr': args.reversal,
                    'g_leak': gleak,
                    'E_leak': Eleak,
                    'V_off': V_off,
                    'C_m': Cm,
                    'R_s': Rseries,
                    'well': well,
                    'protocol': protocol,
                    'sweep': sweep
                }

                new_rows.append(param_dict)
    new_df = pd.DataFrame.from_records(new_rows)

    return new_df


def define_protocol_order(chrono_fname):
    with open(chrono_fname, 'r') as fin:
        lines = fin.read().splitlines()
        protocol_order = [line.split(' ')[0] for line in lines]
        protocol_order.insert(1, 'staircaseramp1_sweep2')
        protocol_order.append('staircaseramp1_2_sweep2')

    return protocol_order

def get_protocol_label(protocol_order, protocol, sweep):
    sweep = int(sweep) + 1
    p_protocol = protocol
    if p_protocol == 'staircaseramp1_2':
        p_protocol = 'staircaseramp1'

    protocol_order = [prot for prot in protocol_order
                      if prot not in ['staircaseramp1_sweep2',
                                      'staircaseramp1_2_sweep2']]

    prot_index = protocol_order.index(protocol)
    ret_str = r'$d_{' + str(protocol_order.index(protocol) + 1) \
        + r'}'
    if p_protocol == 'staircaseramp1':
        ret_str += r'^{(' + str(sweep) + r')}$'
    elif p_protocol == 'staircaseramp1_2':
        ret_str += r'^{(' + str(sweep + 2) + r')}$'
    else:
        ret_str += r'$'

    return ret_str

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
        if args.protocols:
            params_df = params_df[params_df.protocol.isin(args.protocols)].copy()
        prediction_df = compute_predictions_df(params_df, output_dir,
                                               protocol_dict, fitting_case,
                                               args.reversal, subtractions_df,
                                               model_class=model_class,
                                               label=f"{model_class}_{fitting_case}_predictions",
                                               data_label=data_label,
                                               hybrid=False,
                                               strict=False,
                                               args=args,
                                               plot=not args.dont_plot_predictions,
                                               validation_protocols=args.validation_protocols
                                               )

    chrono_fname = os.path.join(args.chrono_file)
    protocol_order = define_protocol_order(chrono_fname)

    def rename_staircase_func(row):
        f_protocol, v_protocol, f_sweep, v_sweep = [row[key] for key in ['fitting_protocol', 'validation_protocol', 'fitting_sweep', 'prediction_sweep']]

        if f_protocol in ['staircaseramp1', 'staircaseramp1_2', 'staircaseramp2'] and f_sweep == 1:
            row['fitting_protocol'] = str(f_protocol) + "_sweep2"

        if v_protocol in ['staircaseramp1', 'staircaseramp1_2', 'staircaseramp2'] and v_sweep == 1:
            row['validation_protocol'] = str(v_protocol) + "_sweep2"

        return row

    prediction_df = prediction_df[~prediction_df.fitting_protocol.isin(args.ignore_fitting_protocols)]
    prediction_df = prediction_df[~prediction_df.validation_protocol.isin(args.ignore_validation_protocols)]
    prediction_df = prediction_df[~prediction_df.well.isin(args.ignore_wells)]
    prediction_df = prediction_df[~prediction_df.fitting_protocol.isin(args.validation_protocols)]

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

    ax.set_title(r'$\mathcal{E}_{\mathrm{fit}} = $' f"{mean_training_score:.2f}" + \
    ",\n" r'$\mathcal{E}_{\mathrm{predict}} = $' + f"{mean_validation_score:.2f}")

    hm = sns.heatmap(pivot_df, ax=ax, square=True, norm=norm,
                     cmap=cmap, **kws)

    autoAxis = ax.axis()
    rec = Rectangle(
        (autoAxis[0] - 0.05, autoAxis[3] - 0.05),
        (autoAxis[1] - autoAxis[0] + 0.05),
        1.1,
        fill=False,
        color='yellow',
        lw=.75
        )

    if len(args.validation_protocols) > 0:
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

    no_cases = len(args.cases)
    no_models = len(args.model_classes)
    no_columns = 1 + no_cases

    gs = GridSpec(no_rows, no_columns, figure=fig, height_ratios=[.075] + [1] *
                  no_models + [0.075], width_ratios=[.075] + no_cases*[1])

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

    case_labels = ['Case I', 'Case II', 'Case III']

    case_labels = [relabel_case_dict[case] for case in args.cases]

    for i, (label_ax, case_label) in enumerate(zip(case_label_axs, case_labels)):
        case_label = case_labels[i]
        label_ax.text(.5, .5, case_label, horizontalalignment='center',
                      verticalalignment='center')

    for ax in model_label_axs + case_label_axs:
        ax.set_axis_off()

    return model_axs, model_label_axs, case_label_axs, colour_bar_ax


def setup_grid_single_case(fig, args):
    # Row for each model, a colorbar, and case labels
    no_models = len(args.model_classes)
    no_columns = 3
    no_rows = 2

    gs = GridSpec(no_rows, no_columns, figure=fig, width_ratios=[1, 1, .1],
                  height_ratios=[1, 1])

    colour_bar_ax = fig.add_subplot(gs[:, -1])

    model_axs = [fig.add_subplot(gs[0, i]) for i in range(2)] \
        + [fig.add_subplot(gs[1, i]) for i in range(2)]

    return model_axs, colour_bar_ax


def setup_best_worst_fig(fig):
    no_models = len(args.model_classes)
    no_columns = 3
    no_rows = 5

    gs = GridSpec(no_rows, no_columns, figure=fig, width_ratios=[1, 1, 0.05],
                  height_ratios=[0.25, 0.5, 0.25, .5, 1.25]
                  )

    heatmap_axs = [fig.add_subplot(gs[-1, i]) for i in range(no_columns)]
    prediction_axs = [fig.add_subplot(gs[2*i + 1, :]) for i in range(2)]
    voltage_axs = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[2, :])]

    for ax in prediction_axs:
        ax.set_ylabel(r'$I_\mathrm{Kr} (pA)$')

    prediction_axs[-1].set_xlabel(r'$t$ (ms)')

    for ax in list(prediction_axs) + list(voltage_axs):
        ax.spines[['top', 'right']].set_visible(False)

    subfigure_captions = [r'$\textbf{' + str(lab) + r'}$' for lab in
                          ['a', 'b', 'c', 'd', 'e', 'f']]

    for i, ax in enumerate(prediction_axs):
        prediction_axs[i].set_title(subfigure_captions[2*i + 1],
                                    fontweight='bold', loc='left')

    voltage_axs[0].set_title(subfigure_captions[0],
                          fontweight='bold', loc='left')

    voltage_axs[1].set_title(subfigure_captions[2],
                          fontweight='bold', loc='left')

    heatmap_axs[0].set_title(subfigure_captions[4],
                             fontweight='bold', loc='left')
    heatmap_axs[1].set_title(subfigure_captions[5],
                             fontweight='bold', loc='left')

    return heatmap_axs, list(reversed(prediction_axs)), list(reversed(voltage_axs))


if __name__ == "__main__":
    main()
