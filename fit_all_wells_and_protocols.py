#!/usr/bin/env python3

import multiprocessing
import regex as re
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from MarkovModels import common
from MarkovModels.BeattieModel import BeattieModel
from MarkovModels.ClosedOpenModel import ClosedOpenModel
from MarkovModels.KempModel import KempModel

import os
import pandas as pd
import numpy as np


T=298
K_in=5
K_out=120


global Erev
Erev = common.calculate_reversal_potential(T=T, K_in=K_in, K_out=K_out)


def fit_func(protocol, well, model_class, default_parameters=None, E_rev=None):
    this_output_dir = os.path.join(output_dir, f"{protocol}_{well}")

    res_df = common.fit_well_data(model_class, well, protocol,
                                  args.data_directory, args.max_iterations,
                                  output_dir=this_output_dir, T=298, K_in=5,
                                  K_out=120,
                                  default_parameters=default_parameters,
                                  removal_duration=args.removal_duration,
                                  repeats=args.repeats, infer_E_rev=True,
                                  experiment_name=experiment_name)

    res_df['well'] = well
    res_df['protocol'] = protocol

    return res_df


def mcmc_func(protocol, well, model_class, initial_params):

    # Ignore files that have been commented out
    voltage_func, t_start, t_end, t_step, protocol_desc = common.get_ramp_protocol_from_csv(protocol)

    data = common.get_data(well, protocol, args.data_directory, experiment_name)

    times = pd.read_csv(os.path.join(args.data_directory, f"{experiment_name}-{protocol}-times.csv"))['time'].values

    voltages = np.array([voltage_func(t) for t in times])

    model = model_class(voltage=voltage_func, Erev=Erev,
                        protocol_description=protocol_desc,
                        times=times)

    if initial_params is None:
        initial_params = model.get_default_parameters()

    try:
        solver = model.make_hybrid_solver_current()
    except NotImplementedError:
        solver = model.make_forward_solver_current()

    if np.any(~np.isfinite(solver(initial_params))):
        initial_params = model.get_default_parameters()

    sigma2 = np.std(data[10:100])**2
    print("sigma2 is ", sigma2)

    spike_times, spike_indices = common.detect_spikes(times, voltages, threshold=10)

    _, _, indices = common.remove_spikes(times, voltages, spike_times,
                                         time_to_remove=args.removal_duration)

    return common.compute_mcmc_chains(model, solver=solver, times=times,
                                      indices=indices, data=data,
                                      chain_length=args.chain_length,
                                      no_chains=args.no_chains,
                                      starting_parameters=initial_params,
                                      sigma2=sigma2, burn_in=0,
                                      log_likelihood_func=None)


def main():
    Erev = common.calculate_reversal_potential(T=298, K_in=120, K_out=5)
    print(f"Erev is {Erev}")
    parser = common.get_parser(
        data_reqd=True, description="Fit a given well to the data from each\
        of the protocols. Output the resulting parameters to a file for later use")

    parser.add_argument('--max_iterations', '-i', type=int, default="100000")
    parser.add_argument('--repeats', type=int, default=16)
    parser.add_argument('--wells', '-w', type=str, default=[], nargs='+')
    parser.add_argument('--protocols', type=str, default=[], nargs='+')
    parser.add_argument('--removal_duration', '-r', default=5, type=float)
    parser.add_argument('--cores', '-c', default=1, type=int)
    parser.add_argument('--model', '-m', default='Beattie', type=str)
    parser.add_argument('--experiment_name', default='newtonrun4', type=str)
    parser.add_argument('--no_chains', '-N', default=0, help='mcmc chains to run', type=int)
    parser.add_argument('--chain_length', '-l', default=500, help='mcmc chains to run', type=int)
    parser.add_argument('--figsize', '-f', help='mcmc chains to run', type=int)
    parser.add_argument('--use_parameter_file')

    global args
    args = parser.parse_args()

    global output_dir
    output_dir = args.output

    global experiment_name
    experiment_name = args.experiment_name

    output_dir = common.setup_output_directory(args.output, f"fitting_{args.removal_duration:.2f}_removed_{args.model}")

    global model_class
    model_class = common.get_model_class(args.model)

    regex = re.compile(f"^{experiment_name}-([a-z|A-Z|0-9]*)-([A-Z][0-9][0-9]).csv$")

    if len(args.wells) == 0:
        args.wells = common.get_all_wells_in_directory(args.data_directory, experiment_name)

    if len(args.protocols) == 0:
        protocols = common.get_protocol_list()
    else:
        protocols = args.protocols

    print(args.wells, protocols)

    if args.use_parameter_file:
        best_params_df = pd.read_csv(args.use_parameter_file)
    else:
        best_params_df = None

    tasks = []
    protocols_list = []

    param_labels = model_class().get_parameter_labels()
    for f in filter(regex.match, os.listdir(args.data_directory)):
        protocol, well = re.search(regex, f).groups()
        if protocol not in protocols or well not in args.wells:
            continue

        if best_params_df is not None:
            parameter_row = best_params_df[(best_params_df.well == well)
                                           & (best_params_df.protocol == protocol)].head(1)
            starting_parameters = parameter_row[param_labels].values.flatten().astype(np.float64)
        else:
            starting_parameters = None

        tasks.append([protocol, well, model_class, starting_parameters])
        protocols_list.append(protocol)

    print(f"fitting tasks are {tasks}")

    assert len(tasks) > 0, "no valid protocol/well combinations provided"

    protocols_list = np.unique(protocols_list)
    pool = multiprocessing.Pool(min(args.cores, len(tasks)))
    res = pool.starmap(fit_func, tasks)
    print(res)

    fitting_df = pd.concat(res, ignore_index=True)

    print("=============\nfinished fitting first round\n=============")

    wells_rep = [task[1] for task in tasks]
    # protocols_rep = [task[0] for task in tasks]

    fitting_df.to_csv(os.path.join(output_dir, "prelim_fitting.csv"))

    params_df = get_best_params(fitting_df)

    params_df.to_csv(os.path.join(output_dir, "prelim_best_fitting.csv"))

    predictions_df = compute_predictions_df(params_df, 'prelim_predictions')

    # Plot predictions
    print(predictions_df)
    predictions_df.to_csv(os.path.join(output_dir, "prelim_predictions_df.csv"))

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
        protocol, well, model_class, _ = task
        best_params_row = best_params_df[(best_params_df.well == well)
                                         & (best_params_df.validation_protocol == protocol)].head(1)
        param_labels = model_class().get_parameter_labels()
        best_params = best_params_row[param_labels].astype(np.float64).values.flatten()
        task[-1] = best_params

    print(tasks)
    res = pool.starmap(fit_func, tasks)
    fitting_df = pd.concat(res + [fitting_df], ignore_index=True)
    fitting_df.to_csv(os.path.join(output_dir, "fitting.csv"))

    predictions_df = compute_predictions_df(params_df)
    predictions_df.to_csv(os.path.join(output_dir, "predictions_df.csv"))

    best_params_df = get_best_params(predictions_df, protocol_label='validation_protocol')
    print(best_params_df)

    best_params_df.to_csv(os.path.join(output_dir, 'best_fitting.csv'))

    for task in tasks:
        protocol, well, model_class, _ = task
        param_labels = model_class().get_parameter_labels()

        row = best_params_df[(best_params_df.well == well)
                             & (best_params_df.validation_protocol ==
                                protocol)][param_labels].copy().head(1).astype(np.float64)

        task[-1] = row.values.flatten()

    print(tasks)

    # do mcmc
    do_mcmc(tasks, pool)


def do_mcmc(tasks, pool):
    if args.chain_length > 0 and args.no_chains > 0:
        mcmc_dir = os.path.join(output_dir, 'mcmc_samples')

        if not os.path.exists(mcmc_dir):
            os.makedirs(mcmc_dir)

        # Do MCMC
        mcmc_res = pool.starmap(mcmc_func, tasks)
        for samples, task in zip(mcmc_res, tasks):
            protocol, well, model_class, _ = task
            model_name = model_class().get_model_name()

            np.save(os.path.join(mcmc_dir,
                                 f"mcmc_{model_name}_{well}_{protocol}.npy"), samples)


def compute_predictions_df(params_df, label='predictions'):

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
        prot_func, tstart, tend, tstep, desc = common.get_ramp_protocol_from_csv(sim_protocol)
        full_times = pd.read_csv(os.path.join(args.data_directory,
                                         f"{experiment_name}-{sim_protocol}-times.csv"))['time'].values.flatten()

        model = model_class(prot_func,
                            times=full_times)

        voltages = np.array([prot_func(t) for t in full_times])

        spike_times, spike_indices = common.detect_spikes(full_times, voltages,
                                                          threshold=10)
        _, _, indices = common.remove_spikes(full_times, voltages, spike_times,
                                             time_to_remove=args.removal_duration)
        times = full_times[indices]

        colours = sns.color_palette('husl', len(params_df['protocol'].unique()))

        for well in params_df['well'].unique():
            full_data = common.get_data(well, sim_protocol, args.data_directory, experiment_name=experiment_name)
            data = full_data[indices]

            # Probably not worth compiling solver
            model.protocol_description = desc
            model.Erev = common.infer_reversal_potential(sim_protocol, full_data, full_times)
            solver = model.make_forward_solver_current(njitted=False)

            for i, protocol_fitted in enumerate(params_df['protocol'].unique()):
                df = params_df[params_df.well == well]
                df = df[df.protocol == protocol_fitted]

                if df.empty:
                    continue

                param_labels = model.parameter_labels
                params = df.iloc[0][param_labels].values\
                                                 .astype(np.float64)\
                                                 .flatten()
                sub_dir = os.path.join(predictions_dir, f"{well}_{sim_protocol}_predictions")
                if not os.path.exists(sub_dir):
                    os.makedirs(sub_dir)

                prediction = solver(params)[indices]

                score = np.sqrt(np.mean((data - prediction)**2))
                predictions_df.append((well, protocol_fitted, sim_protocol, score, *params))

                if not np.all(np.isfinite(prediction)):
                    logging.warning(f"running {sim_protocol} with parameters\
                    from {protocol_fitted} gave non-finite values")
                else:
                    # Output trace
                    trace_axs[0].plot(times, prediction, label='prediction')

                    trace_axs[1].set_xlabel("time / ms")
                    trace_axs[0].set_ylabel("current / nA")
                    trace_axs[0].plot(times, data, label='data', alpha=0.25, color='grey')
                    trace_axs[0].legend()
                    trace_axs[1].plot(full_times, voltages)
                    trace_axs[1].set_ylabel('voltage / mV')
                    fname = f"fitted_to_{protocol_fitted}.png" if protocol_fitted != sim_protocol else "fit.png"
                    trace_fig.savefig(os.path.join(sub_dir, fname))

                    for ax in trace_axs:
                        ax.cla()

                    all_models_axs[0].plot(times, prediction, label=protocol_fitted, color=colours[i])

            all_models_axs[1].set_xlabel("time / ms")
            all_models_axs[0].set_ylabel("current / nA")
            all_models_axs[0].plot(times, data, color='grey', alpha=0.5, label='data')
            all_models_axs[0].legend()
            all_models_axs[0].set_title(f"{well} {sim_protocol} fits comparison")
            all_models_axs[0].set_ylabel("Current / nA")

            all_models_axs[1].plot(full_times, voltages)
            all_models_axs[1].set_ylabel('voltage / mV')

            all_models_fig.savefig(os.path.join(sub_dir, "all_fits.png"))

            for ax in all_models_axs:
                ax.cla()

    # TODO refactor so this can work with more than one model
    predictions_df = pd.DataFrame(np.array(predictions_df), columns=['well', 'fitting_protocol',
                                                                      'validation_protocol',
                                                                      'score'] + param_labels)
    predictions_df['RMSE'] = predictions_df['score']
    return predictions_df


def get_best_params(fitting_df, protocol_label='protocol'):
    protocols_list = fitting_df[protocol_label].unique()
    wells_list = fitting_df['well'].unique()
    best_params = []
    for protocol in np.unique(protocols_list):
        for well in np.unique(wells_list):
            sub_df = fitting_df[(fitting_df['well'] == well)
                                & (fitting_df[protocol_label] == protocol)].copy()

            # Get index of min score
            best_params.append(sub_df[sub_df.score == sub_df.score.min()].head(1).copy())

    return pd.concat(best_params, ignore_index=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
