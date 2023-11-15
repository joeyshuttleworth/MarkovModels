import logging
import multiprocessing
import os

import matplotlib
import matplotlib.pyplot as plt
import regex as re
import seaborn as sns
import pandas as pd
import numpy as np
import markovmodels

from argparse import ArgumentParser
from markovmodels.model_generation import make_model_of_class
from markovmodels.ArtefactModel import ArtefactModel
from markovmodels.BeattieModel import BeattieModel
from markovmodels.fitting import infer_reversal_potential_with_artefact

matplotlib.use('agg')
pool_kws = {'maxtasksperchild': 1}


def fit_func(protocol, well, model_class, default_parameters=None, E_rev=None,
             randomise_initial_guess=True, prefix='', sweep=None):
    this_output_dir = os.path.join(output_dir, f"{prefix}{protocol}_{well}_sweep{sweep}")

    infer_E_rev = not args.dont_infer_Erev

    if args.dont_infer_Erev and args.use_artefact_model:
        raise Exception()

    fix_parameters = []
    if args.use_artefact_model:
        fix_parameters = [-1, -2, -3, -4, -5, -6, -7]
        fix_parameters = [i % len(default_parameters) for i in fix_parameters]

    res_df = markovmodels.fitting.fit_well_data(
        model_class, well, protocol,
        args.data_directory,
        args.max_iterations,
        output_dir=this_output_dir,
        default_parameters=default_parameters,
        removal_duration=args.removal_duration,
        repeats=args.repeats,
        infer_E_rev=infer_E_rev,
        experiment_name=args.experiment_name,
        E_rev=E_rev,
        randomise_initial_guess=randomise_initial_guess,
        solver_type=args.solver_type,
        sweep=sweep,
        use_artefact_model=args.use_artefact_model,
        fix_parameters=fix_parameters,
        data_label=args.data_label,
        artefact_default_kinetic_parameters=default_artefact_kinetic_parameters
    )

    res_df['well'] = well
    res_df['protocol'] = protocol
    res_df['sweep'] = sweep if sweep else -1

    return res_df


def main():
    parser = ArgumentParser()
    parser.add_argument('data_directory')
    parser.add_argument('--max_iterations', '-i', type=int, default=100000)
    parser.add_argument('--repeats', type=int, default=16)
    parser.add_argument('--dont_randomise_initial_guess', action='store_true')
    parser.add_argument('--wells', '-w', type=str, default=[], nargs='+')
    parser.add_argument('--protocols', type=str, default=[], nargs='+')
    parser.add_argument('--removal_duration', '-r', default=5, type=float)
    parser.add_argument('--cores', '-c', default=1, type=int)
    parser.add_argument('--model', '-m', default='Beattie', type=str)
    parser.add_argument('--experiment_name', default='newtonrun4', type=str)
    parser.add_argument('--no_chains', '-N', default=0, type=int)
    parser.add_argument('--chain_length', '-l', default=500, type=int)
    parser.add_argument('--figsize', '-f', help='mcmc chains to run', type=int)
    parser.add_argument('--use_parameter_file')
    parser.add_argument('--artefact_default_kinetic_param_file')
    parser.add_argument('--dont_refit', action='store_true')
    parser.add_argument('--dont_infer_Erev', action='store_true')
    parser.add_argument('--solver_type', default='hybrid')
    parser.add_argument('--selection_file')
    parser.add_argument('--ignore_protocols', nargs='+', default=[])
    parser.add_argument('--ignore_wells', nargs='+', default=[])
    parser.add_argument('--sweeps', nargs='+', default=None)
    parser.add_argument('--use_artefact_model', action='store_true')
    parser.add_argument('--subtraction_df_file')
    parser.add_argument('--qc_df_file')
    parser.add_argument('--data_label')
    parser.add_argument('--reversal', type=float)
    parser.add_argument('-o', '--output')

    global args
    args = parser.parse_args()

    global output_dir
    output_dir = args.output

    global experiment_name
    experiment_name = args.experiment_name

    if args.selection_file:
        with open(args.selection_file) as fin:
            selected_wells = fin.read().splitlines()
    else:
        selected_wells = None

    global default_artefact_kinetic_parameters
    default_artefact_kinetic_parameters = np.loadtxt(args.artefact_default_kinetic_param_file).flatten().astype(np.float64)

    if args.use_artefact_model and not (args.qc_df_file and args.subtraction_df_file):
        raise Exception('Cannot use artefact model without qc file')

    if args.use_artefact_model and not args.reversal:
        raise Exception('Nernst potential must be provided when using an artefact model')

    if args.qc_df_file:
        qc_df = pd.read_csv(args.qc_df_file)
        qc_df = qc_df[qc_df.drug == 'before']

    if args.subtraction_df_file:
        subtraction_df = pd.read_csv(args.subtraction_df_file)

    output_dir = markovmodels.utilities.setup_output_directory(
        args.output,
        f"fitting_{args.experiment_name}_{args.model}"
    )

    if len(args.wells) == 0:
        args.wells = markovmodels.utilities.get_all_wells_in_directory(args.data_directory, experiment_name)

    if len(args.protocols) == 0:
        protocols = markovmodels.voltage_protocols.get_protocol_list()
    else:
        protocols = args.protocols

    if args.selection_file:
        args.wells = [well for well in args.wells if well in selected_wells]

    if args.use_parameter_file:
        # Here we can use previous results to refit. Just use the best
        # parameters for each validation protocol as the initial guess
        best_params_df = pd.read_csv(args.use_parameter_file)
        if 'validation_protocol' in best_params_df:
            protocol_label = 'validation_protocol'
        else:
            protocol_label = 'protocol'

        best_params_df = get_best_params(best_params_df,
                                         protocol_label=protocol_label)
        assert(args.dont_randomise_initial_guess)

    else:
        best_params_df = None

    tasks = []
    protocols_list = []

    if args.sweeps:
        regex = re.compile(f"^{experiment_name}-([a-z|A-Z|0-9]*)-([A-Z|0-9]*)-sweep([0-9]).csv$")
    else:
        regex = re.compile(f"^{experiment_name}-([a-z|A-Z|0-9]*)-([A-Z|0-9]*).csv$")

    param_labels = make_model_of_class(args.model).get_parameter_labels()
    for f in filter(regex.match, os.listdir(args.data_directory)):
        groups = re.search(regex, f).groups()
        protocol = groups[0]
        well = groups[1]
        if protocol not in protocols or well not in args.wells:
            continue
        if protocol in args.ignore_protocols or well in args.ignore_wells:
            continue

        if best_params_df is not None:
            parameter_row = best_params_df[(best_params_df.well.astype(str) == str(well))
                                           & (best_params_df[protocol_label] == protocol)].head(1)
            starting_parameters = parameter_row[param_labels].values.flatten().astype(np.float64)
        else:
            starting_parameters = None

        if args.dont_refit:
            prefix = ''
        else:
            prefix = 'prelim_'

        sweep = int(groups[2])
        if args.use_artefact_model:
            print(qc_df[(qc_df.well == well) & (qc_df.protocol == protocol)
                        & (qc_df.sweep == sweep)])
            row = qc_df[(qc_df.well == well) & (qc_df.protocol == protocol) &
                        (qc_df.sweep == sweep)]
            # assert(row.shape[0] == 1)

            Rseries, Cm = row.iloc[0][['Rseries', 'Cm']]

            Rseries = Rseries * 1e-9
            Cm = Cm * 1e9

            row = subtraction_df[(subtraction_df.well == well) & (subtraction_df.protocol == protocol)
                                 & (subtraction_df.sweep == sweep)].iloc[0]
            gleak, Eleak = row[['pre-drug leak conductance', 'pre-drug leak reversal']]
            gleak = float(gleak)
            Eleak = float(Eleak)

            default_parameters = markovmodels.model_generation.make_model_of_class(args.model).get_default_parameters()
            starting_parameters = np.append(default_parameters, [gleak, Eleak, 0, 0, 0, Cm, Rseries])
        tasks.append([protocol, well, args.model, starting_parameters, args.reversal,
                      not args.dont_randomise_initial_guess, prefix, sweep])

        protocols_list.append(protocol)

    print(f"fitting tasks are {tasks}")
    assert len(tasks) > 0, "no valid protocol/well combinations provided"
    protocols_list = np.unique(protocols_list)
    pool_size = min(args.cores, len(tasks))

    with multiprocessing.Pool(pool_size, **pool_kws) as pool:
        res = pool.starmap(fit_func, tasks)

    fitting_df = pd.concat(res, ignore_index=True)

    print("=============\nfinished fitting first round\n=============")

    fitting_df.to_csv(os.path.join(output_dir, f"{prefix}fitting.csv"))

    params_df = get_best_params(fitting_df)
    params_df.to_csv(os.path.join(output_dir, f"{prefix}best_fitting.csv"))

    print(params_df)
    predictions_df = compute_predictions_df(params_df, output_dir,
                                            f"{prefix}predictions", args=args, model_class=args.model)

    # Plot predictions
    predictions_df.to_csv(os.path.join(output_dir, f"{prefix}predictions_df.csv"))

    # Select best parameters for each protocol
    best_params_df_rows = []
    print(predictions_df)
    for well in predictions_df.well.unique():
        for validation_protocol in predictions_df['validation_protocol'].unique():
            sub_df = predictions_df[(predictions_df.validation_protocol ==
                                     validation_protocol) & (predictions_df.well == well)]

            best_param_row = sub_df[sub_df.score == sub_df['score'].min()].head(1).copy()
            best_params_df_rows.append(best_param_row)

    best_params_df = pd.concat(best_params_df_rows, ignore_index=True)
    print(best_params_df)

    for task in tasks:
        if args.sweeps:
            protocol, well, model_class, default_parameters, Erev, randomise, _, sweep = task
        else:
            protocol, well, model_class, default_parameters, Erev, randomise, _ = task
        best_params_row = best_params_df[(best_params_df.well == well)
                                         & (best_params_df.validation_protocol == protocol)].head(1)
        param_labels = make_model_of_class(model_class).get_parameter_labels()
        best_params = best_params_row[param_labels].astype(np.float64).values.flatten()
        task[3] = best_params
        task[6] = ''
        task[5] = False

    if not args.dont_refit:
        with multiprocessing.Pool(pool_size, **pool_kws) as pool:
            res = pool.starmap(fit_func, tasks)

            fitting_df = pd.concat(res + [fitting_df], ignore_index=True)
            fitting_df.to_csv(os.path.join(output_dir, "fitting.csv"))

            predictions_df = compute_predictions_df(params_df, output_dir,
                                                    model_class=model_class,
                                                    args=args)

            predictions_df.to_csv(os.path.join(output_dir, "predictions_df.csv"))

            best_params_df = get_best_params(predictions_df, protocol_label='validation_protocol')
            print(best_params_df)

            best_params_df['protocol'] = best_params_df['validation_protocol']
            best_params_df.to_csv(os.path.join(output_dir, 'best_fitting.csv'))


def compute_predictions_df(params_df, output_dir, label='predictions',
                           model_class=None, fix_EKr=None,
                           adjust_kinetic_parameters=False, args=None):

    assert(not (fix_EKr is not None and adjust_kinetic_parameters))
    param_labels = make_model_of_class(model_class).get_parameter_labels()
    params_df = get_best_params(params_df, protocol_label='protocol')
    predictions_dir = os.path.join(output_dir, label)

    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    predictions_df = []
    protocols_list = params_df['protocol'].unique()

    trace_fig = plt.figure(figsize=args.figsize)
    trace_axs = trace_fig.subplots(2)

    all_models_fig = plt.figure(figsize=args.figsize)
    all_models_axs = all_models_fig.subplots(2)

    for sim_protocol in np.unique(protocols_list):
        prot_func, times, desc = markovmodels.voltage_protocols.get_ramp_protocol_from_csv(sim_protocol)
        full_times = pd.read_csv(
            os.path.join(args.data_directory,
                         f"{args.experiment_name}-{sim_protocol}-times.csv"))['time'].values.flatten()

        voltages = np.array([prot_func(t) for t in full_times])

        spike_times, spike_indices = markovmodels.voltage_protocols.detect_spikes(full_times, voltages,
                                                                                  threshold=10)
        _, _, indices = markovmodels.voltage_protocols.remove_spikes(full_times, voltages, spike_times,
                                                   time_to_remove=args.removal_duration)
        times = full_times[indices]

        colours = sns.color_palette('husl', len(params_df['protocol'].unique()))

        for well in params_df['well'].unique():
            for predict_sweep in params_df[params_df.protocol == sim_protocol].sweep.unique():
                try:
                    full_data = markovmodels.utilities.get_data(well, sim_protocol,
                                                                args.data_directory,
                                                                experiment_name=args.experiment_name,
                                                                label=args.data_label,
                                                                sweep=predict_sweep)
                except (FileNotFoundError, StopIteration) as exc:
                    print(str(exc))
                    continue

                subdir_name = f"{well}_{sim_protocol}_sweep{predict_sweep}_predictions"\
                    if predict_sweep is not None else f"{well}_{sim_protocol}_predictions"
                sub_dir = os.path.join(predictions_dir, subdir_name)

                if not args.use_artefact_model:
                    E_obs = \
                        markovmodels.infer_reversal_potential(sim_protocol,
                                                              full_data,
                                                              full_times,
                                                              forward_sim_output_dir=sub_dir
                                                              )

                    model = make_model_of_class(model_class,
                                                voltage=prot_func,
                                                times=full_times,
                                                E_rev=E_obs if not fix_EKr else fix_EKr,
                                                protocol_description=desc)

                    # Create dir for plot
                else:
                    # Use the artefact to forward simulate the voltages (using literature kinetics)
                    model = make_model_of_class(model_class, voltage=prot_func,
                                                times=times,
                                                E_rev=args.reversal,
                                                protocol_description=desc)

                    model = ArtefactModel(model)
                    forward_sim_parameters = default_artefact_kinetic_parameters.copy()
                    param_row = params_df[(params_df.well == well) &
                                          (params_df.protocol == sim_protocol) &\
                                          (params_df.sweep == predict_sweep)].iloc[0]

                    gleak, Eleak, V_off, Rseries, Cm = param_row[['gleak, Eleak, V_off, Rseries, Cm']]
                    forward_sim_parameters[[-7, -6, -5, -4, -3, -2, -1]] = gleak, Eleak, 0, 0, V_off, Rseries, Cm
                    E_obs = \
                        infer_reversal_potential_with_artefact(sim_protocol,
                                                               full_times,
                                                               full_data,
                                                               'model3',
                                                               args.reversal,
                                                               plot=True,
                                                               output_path=sub_dir,
                                                               forward_sim_output_dir=sub_dir,
                                                               )
                    V_off = args.reversal - E_obs

                # Probably not worth compiling solver
                solver = model.make_forward_solver_of_type(args.solver_type, njitted=False)
                data = full_data[indices]

                for i, protocol_fitted in enumerate(params_df['protocol'].unique()):
                    print(protocol_fitted)
                    for fitting_sweep in params_df[params_df.protocol == protocol_fitted].sweep:
                        # Get parameters
                        df = params_df[params_df.well == well]
                        df = df[(df.protocol == protocol_fitted) & (df.sweep == fitting_sweep)]
                        if df.empty:
                            continue
                        params = df.iloc[0][param_labels].values\
                                                         .astype(np.float64)\
                                                         .flatten()
                        try:
                            fitting_data = pd.read_csv(
                                os.path.join(args.data_directory,
                                             f"{args.experiment_name}-{protocol_fitted}-{well}-sweep{fitting_sweep}.csv"))
                        except FileNotFoundError as e:
                            print(str(e))
                            continue

                        fitting_current = fitting_data['current'].values.flatten()
                        fitting_times = fitting_data['time'].values.flatten()

                        if adjust_kinetic_parameters and not args.use_artefact_model:
                            fitting_E_rev = markovmodels.infer_reversal_potential(protocol_fitted, fitting_current,
                                                                                  fitting_times)
                            if not args.reversal:
                                Exception('reversal potential not provided')
                            else:
                                offset = E_obs - fitting_E_rev
                                params[0] *= np.exp(params[1] * offset)
                                params[2] *= np.exp(-params[3] * offset)
                                params[4] *= np.exp(params[5] * offset)
                                params[6] *= np.exp(-params[7] * offset)

                        if not os.path.exists(sub_dir):
                            os.makedirs(sub_dir)

                        # Set V_offset
                        if args.use_artefact_model:
                            params[-3] = V_off

                        full_prediction = solver(params)
                        prediction = full_prediction[indices]

                        score = np.sqrt(np.mean((data - prediction)**2))
                        predictions_df.append((well, protocol_fitted,
                                               fitting_sweep, predict_sweep, sim_protocol, score,
                                               * params))

                        if not np.all(np.isfinite(prediction)):
                            logging.warning(f"running {sim_protocol} with parameters\
                            from {protocol_fitted} gave non-finite values")
                        else:
                            # Output trace
                            trace_axs[0].plot(full_times, full_prediction, label='prediction')

                            trace_axs[1].set_xlabel("time / ms")
                            trace_axs[0].set_ylabel("current / nA")
                            trace_axs[0].plot(times, data, label='data', alpha=0.25, color='grey')
                            trace_axs[0].legend()
                            trace_axs[1].plot(full_times, voltages)
                            trace_axs[1].set_ylabel('voltage / mV')
                            fname = f"fitted_to_{protocol_fitted}_{fitting_sweep}.png" if protocol_fitted != sim_protocol or \
                                fitting_sweep != predict_sweep else "fit.png"

                            handles, labels = trace_axs[1].get_legend_handles_labels()
                            by_label = dict(zip(labels, handles))
                            plt.legend(by_label.values(), by_label.keys())

                            trace_fig.savefig(os.path.join(sub_dir, fname))

                            for ax in trace_axs:
                                ax.cla()

                            all_models_axs[0].plot(full_times, full_prediction,
                                                   label=f"{protocol_fitted}_{fitting_sweep}", color=colours[i])

                all_models_axs[1].set_xlabel("time / ms")
                all_models_axs[0].set_ylabel("current / nA")
                all_models_axs[0].plot(times, data, color='grey', alpha=0.5, label='data')
                # all_models_axs[0].legend()
                all_models_axs[0].set_title(f"{well} {sim_protocol} fits comparison")
                all_models_axs[0].set_ylabel("Current / nA")

                all_models_axs[1].plot(full_times, voltages)
                all_models_axs[1].set_ylabel('voltage / mV')

                all_models_fig.savefig(os.path.join(sub_dir, "all_fits.png"))

                for ax in all_models_axs:
                    ax.cla()

    predictions_df = pd.DataFrame(np.array(predictions_df), columns=['well',
                                                                     'fitting_protocol',
                                                                     'fitting_sweep',
                                                                     'prediction_sweep',
                                                                     'validation_protocol',
                                                                     'score'] +
                                  param_labels)
    predictions_df['RMSE'] = predictions_df['score']
    predictions_df['sweep'] = predictions_df.fitting_sweep

    plt.close(trace_fig)
    plt.close(all_models_fig)

    return predictions_df


def get_best_params(fitting_df, protocol_label='protocol'):
    best_params = []

    # Ensure score is a float - may be read from csv file
    fitting_df['score'] = fitting_df['score'].astype(np.float64)
    fitting_df = fitting_df[np.isfinite(fitting_df['score'])].copy()

    if 'sweep' not in fitting_df.columns:
        fitting_df['sweep'] = -1

    for protocol in fitting_df[protocol_label].unique():
        for well in fitting_df['well'].unique():
            for sweep in fitting_df['sweep'].unique():
                sub_df = fitting_df[(fitting_df['well'] == well)
                                    & (fitting_df[protocol_label] == protocol)].copy()
                sub_df = sub_df[sub_df.sweep == sweep]
                sub_df = sub_df.dropna()
                # Get index of min score
                if len(sub_df.index) == 0:
                    continue
                best_params.append(sub_df[sub_df.score == sub_df.score.min()].head(1).copy())

    if not best_params:
        raise Exception()

    return pd.concat(best_params, ignore_index=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
