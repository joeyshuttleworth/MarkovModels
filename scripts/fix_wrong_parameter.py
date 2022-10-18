#!/usr/bin/env python3

# Synthetic data study where we fix the maximal conductance parameter to the
# wrong value and fit the model. Make this parameter pertubation larger and larger
# and see what thisdoes to parameter spread, predictions, etc...

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
K_in=120
K_out=5


global Erev
Erev = common.calculate_reversal_potential(T=T, K_in=K_in, K_out=K_out)

pool_kws = {'maxtasksperchild': 1}


def main():
    Erev = common.calculate_reversal_potential(T=298, K_in=120, K_out=5)
    parser = common.get_parser(
        data_reqd=False, description="Fit a given well to the data from each\
        of the protocols. Output the resulting parameters to a file for later use")

    parser.add_argument('--max_iterations', '-i', type=int, default=100000)
    parser.add_argument('--repeats', type=int, default=16)
    parser.add_argument('--removal_duration', '-r', default=5, type=float)
    parser.add_argument('--cores', '-c', default=1, type=int)
    parser.add_argument('--model', '-m', default='Beattie', type=str)
    parser.add_argument('--figsize', '-f', help='mcmc chains to run', type=int)
    parser.add_argument('--use_parameter_file')
    parser.add_argument('--protocols', default=common.get_protocol_list(), nargs='+')
    parser.add_argument('--noise', default=0.01, type=float)
    parser.add_argument('--no_repeats', default=100, type=int)
    parser.add_argument('--no_parameter_steps', default=25, type=int)

    global args
    args = parser.parse_args()

    global output_dir
    output_dir = common.setup_output_directory(args.output,
                                               "fix_wrong_parameter")
    global model_class
    model_class = common.get_model_class(args.model)

    global true_params
    if args.use_parameter_file:
        true_params = pd.read_csv(args.use_parameter_file).values
    else:
        true_params = model_class().get_default_parameters()

    param_labels = model_class().get_parameter_labels()

    global data_sets
    data_sets = generate_synthetic_data_sets(args.protocols,
                                             args.no_repeats,
                                             parameters=true_params,
                                             noise=args.noise,
                                             output_dir=output_dir)
    fix_param = len(param_labels) - 1

    # Use multiprocessing to compute process multiple synthetic datasets in parallel
    tasks = []
    for i, data in enumerate(data_sets[0]):
        for protocol in args.protocols:
            # model and dataset index
            tasks.append([args.model, i, fix_param, protocol])

    pool_size = min(args.cores, len(tasks))

    with multiprocessing.Pool(pool_size, **pool_kws) as pool:
        res = pool.starmap(fit_func, tasks)

    fitted_params = np.vstack(res)

    rows = []
    for task in tasks:
        for i in range(res[0].shape[0]):
            rows.append(task)

    res_df = pd.DataFrame(rows, columns=('model_class', 'dataset_index', 'fix_param', 'protocol'))

    for i, param_label in enumerate(model_class().get_parameter_labels()):
        res_df[param_label] = fitted_params[:, i]
    res_df['well'] = res_df['dataset_index']

    res_df.to_csv(os.path.join(output_dir, 'results_df.csv'))

    datasets_df = []
    for protocol_index, protocol in enumerate(args.protocols):
        for i in range(args.no_repeats):
            datasets_df.append([protocol, protocol_index, i])

    datasets_df = pd.DataFrame(datasets_df, columns=('protocol', 'protocol_index', 'repeat'))

    predictions_df = compute_predictions_df(res_df, model_class, data_sets,
                                            datasets_df, args=args)
    predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'))


def fit_func(model_class_name, dataset_index, fix_param, protocol):
    protocol_index = args.protocols.index(protocol)
    times, data = data_sets[protocol_index][int(dataset_index)]
    voltage_func, t_start, t_end, t_step, protocol_desc = common.get_ramp_protocol_from_csv(protocol)

    model_class = common.get_model_class(model_class_name)
    param_val_lims = model_class(parameters=true_params).get_default_parameters()[fix_param] * np.array([0.75, 1.25])
    param_val_range = np.linspace(*param_val_lims, args.no_parameter_steps)

    mm = model_class(voltage=voltage_func,
                     protocol_description=protocol_desc,
                     times=times)

    voltages = np.array([voltage_func(t) for t in times])
    _, spike_indices = common.detect_spikes(times, voltages, window_size=0)

    indices = common.remove_indices(list(range(len(times))), [(spike, int(spike +
                                                                   args.removal_duration / t_step)) for spike in
                                                       spike_indices])

    res = []

    # Use the previously found parameters as an initial guess in the next
    # iteration
    params = mm.get_default_parameters()
    solver = mm.make_forward_solver_current()
    for fix_param_val in param_val_range:
        params[fix_param] = fix_param_val
        params, score = common.fit_model(mm, data, fix_parameters=[fix_param],
                                         repeats=args.repeats,
                                         max_iterations=args.max_iterations,
                                         starting_parameters=params,
                                         solver=solver)
        res.append(params)
    return np.vstack(res)


def generate_synthetic_data_sets(protocols, n_repeats, parameters=None, noise=0.01, output_dir=None):
    list_of_data_sets = []
    for protocol in protocols:
        prot, tstart, tend, tstep, desc = common.get_ramp_protocol_from_csv(protocol)
        times = np.linspace(tstart, tend, int((tend - tstart)/tstep))

        model_class = common.get_model_class(args.model)
        model = model_class(prot, times, Erev=Erev, parameters=parameters,
                        protocol_description=desc)
        mean = model.SimulateForwardModel()

        data_sets = [(times, mean + np.random.normal(0, noise, times.shape)) for i in
                    range(n_repeats)]

        if output_dir:
            fig = plt.figure(figsize=args.figsize)
            ax = fig.gca()
            for i, data_set in enumerate(data_sets):
                ax.plot(times, data_set[1], color='grey')
                ax.plot(times, mean)
                ax.set_xlabel('time / ms')
                ax.set_ylabel('current / nA')
                fig.savefig(os.path.join(output_dir, f"synthetic_data_{protocol}_{i}.png"), dpi=300)
                ax.cla()

                data_df = pd.DataFrame(np.vstack(data_set).T, columns=['time / ms', 'current / nA'])
                data_df.to_csv(os.path.join(output_dir, f"synthetic_data{protocol}_{i}.csv"))

        list_of_data_sets.append(data_sets)
    return list_of_data_sets


def compute_predictions_df(params_df, model_class, datasets, datasets_df,
                           label='predictions', args=None, output_dir=None):
    if output_dir:
        predictions_dir = os.path.join(output_dir, label)

        if not os.path.exists(predictions_dir):
            os.makedirs(predictions_dir)
        trace_fig = plt.figure(figsize=args.figsize)
        trace_axs = trace_fig.subplots(2)
        all_models_fig = plt.figure(figsize=args.figsize)
        all_models_axs = all_models_fig.subplots(2)

    predictions_df = []
    protocols_list = datasets_df['protocol'].unique()

    for sim_protocol in protocols_list:
        prot_func, tstart, tend, tstep, desc = common.get_ramp_protocol_from_csv(sim_protocol)
        protocol_index = datasets_df[datasets_df.protocol == sim_protocol]['protocol_index'].values[0]
        full_times = datasets[protocol_index][0][0]

        model = model_class(prot_func,
                            times=full_times)

        voltages = np.array([prot_func(t) for t in full_times])

        spike_times, spike_indices = common.detect_spikes(full_times, voltages,
                                                          threshold=10)
        _, _, indices = common.remove_spikes(full_times, voltages, spike_times,
                                             time_to_remove=args.removal_duration)
        times = full_times[indices]

        if output_dir:
            colours = sns.color_palette('husl', len(params_df['protocol'].unique()))

        for well in params_df['well'].unique():
            full_data = datasets[protocol_index][int(well)][1]
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
                if output_dir:
                    sub_dir = os.path.join(predictions_dir, f"{well}_{sim_protocol}_predictions")
                    if not os.path.exists(sub_dir):
                        os.makedirs(sub_dir)

                prediction = solver(params)[indices]

                score = np.sqrt(np.mean((data - prediction)**2))
                predictions_df.append((well, protocol_fitted, sim_protocol,
                                       score, *params))

                if not np.all(np.isfinite(prediction)):
                    logging.warning(f"running {sim_protocol} with parameters\
                    from {protocol_fitted} gave non-finite values")
                elif output_dir:
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

    # TODO refactor so this can work with more than one model
    predictions_df = pd.DataFrame(np.array(predictions_df), columns=['well', 'fitting_protocol',
                                                                      'validation_protocol',
                                                                      'score'] + param_labels)
    predictions_df['RMSE'] = predictions_df['score']
    return predictions_df


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
