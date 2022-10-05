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
K_in=5
K_out=120


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
    parser.add_argument('--fit_protocol', default='staircaseramp')
    parser.add_argument('--validation_protocol', default='staircaseramp')
    parser.add_argument('--noise', default=0.1, type=float)
    parser.add_argument('--no_dgp_repeats', default=100, type=int)
    parser.add_argument('--no_parameter_steps', default=25, type=int)

    global args
    args = parser.parse_args()

    global output_dir
    output_dir = args.output

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
    data_sets = generate_synthetic_data_sets(args.fit_protocol, args.no_dgp_repeats,
                                             parameters=true_params,
                                             noise=args.noise)

    fix_param = len(param_labels) - 1

    # Use multiprocessing to compute process multiple synthetic datasets in parallel
    tasks = []
    for i, data in enumerate(data_sets):
        # model and dataset index
        tasks.append((args.model, i, fix_param))

    pool_size = min(args.cores, len(tasks))

    with multiprocessing.Pool(pool_size, **pool_kws) as pool:
        res = pool.starmap(fit_func, tasks)

    fit_voltage_func, t_start, t_end, t_step, fit_protocol_desc = common.get_ramp_protocol_from_csv(args.fit_protocol)
    fit_times = np.linspace(t_start, t_end, int((t_end - t_start)/t_step))
    fit_voltages = np.array([fit_voltage_func(t) for t in fit_times])
    fit_model = model_class(voltage=fit_voltage_func,
                                 protocol_description=fit_protocol_desc)

    valid_voltage_func, t_start, t_end, t_step, valid_protocol_desc = common.get_ramp_protocol_from_csv(args.validation_protocol)
    valid_times = np.linspace(t_start, t_end, int((t_end - t_start)/t_step))
    valid_voltages = np.array([valid_voltage_func(t) for t in valid_times])
    valid_model = model_class(voltage=valid_voltage_func,
                                 protocol_description=valid_protocol_desc)
    correct_prediction = valid_model.SimulateForwardModel(true_params)
    true_mean = fit_model.SimulateForwardModel(true_params)

    results_df_rows = []
    for i, (task, result) in enumerate(zip(tasks, res)):
        model, data_index, fixed_param = task

        param_sets = [x[0] for x in result]

        for params in param_sets:
            # Save params to file
            # np.save(params, os.path.join(output_dir,
            #                              f"{args.model_class}_{data_index}_{fix_param:.4f}_{i}.csv"))

            prediction = valid_model.SimulateForwardModel(params)
            fit = fit_model.SimulateForwardModel(params)
            # TODO Compute error in fit and error in prediction
            error_in_prediction = np.sum((correct_prediction - prediction)**2)
            error_in_fit = np.sum((true_mean - fit)**2)
            results_df_row = [model, data_index, *params, error_in_prediction, error_in_fit]
            results_df_rows.append(results_df_row)

    column_names = ['model_name', 'dataset_index', *param_labels, 'error in prediction', 'error in fit']
    results_df = pd.DataFrame(results_df_rows, columns=column_names)
    results_df.to_csv(os.path.join(output_dir, 'results.csv'))


def fit_func(model_class_name, data_index, fix_param):
    times, data = data_sets[data_index]
    protocol = args.fit_protocol
    voltage_func, t_start, t_end, t_step, protocol_desc = common.get_ramp_protocol_from_csv(protocol)

    model_class = common.get_model_class(model_class_name)
    param_val_lims = model_class(parameters=true_params).get_default_parameters()[fix_param] * np.array([0.75, 1.25])
    param_val_range = np.linspace(*param_val_lims, args.no_parameter_steps)

    mm = model_class(voltage=voltage_func,
                     protocol_description=protocol_desc,
                     times=times)

    res = []

    # Use the previously found parameters as an initial guess in the next
    # iteration
    params = mm.get_default_parameters()
    solver = mm.make_forward_solver_current()
    for fix_param_val in param_val_range:
        # params = np.array([p for i, p in enumerate(params) if i != fix_param])
        mm.default_parameters[fix_param] = fix_param_val
        params, score = common.fit_model(mm, data, fix_parameters=[fix_param],
                                         repeats=args.repeats,
                                         max_iterations=args.max_iterations,
                                         starting_parameters=params,
                                         solver=solver)

        if fix_param == len(params):
            params = np.append(params, fix_param)
        else:
            params = np.insert(params, fix_param + 1, fix_param_val)

        res.append([params, score])

    return res


def generate_synthetic_data_sets(protocol, n_repeats, parameters=None, noise=0.01):
    prot, tstart, tend, tstep, desc = common.get_ramp_protocol_from_csv(protocol)
    times = np.linspace(tstart, tend, int((tend - tstart)/tstep))

    model_class = common.get_model_class(args.model)
    model = model_class(prot, times, Erev=Erev, parameters=parameters,
                       protocol_description=desc)
    mean = model.SimulateForwardModel()

    data_sets = [(times, mean + np.random.normal(0, noise, times.shape)) for i in
                 range(n_repeats)]

    return data_sets

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
