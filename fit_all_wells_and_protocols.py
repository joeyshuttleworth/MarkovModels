#!/usr/bin/env python3

import multiprocessing
import regex as re
import matplotlib.pyplot as plt
from MarkovModels import common
from MarkovModels.BeattieModel import BeattieModel

import os
import pandas as pd
import numpy as np

param_labels = BeattieModel().parameter_labels + ['E_Kr']


def fit_func(protocol, well):
    default_parameters = None
    this_output_dir = os.path.join(output_dir, f"{protocol}_{well}")

    params, scores = common.fit_well_data(BeattieModel, well, protocol, args.data_directory,
                                          args.max_iterations, output_dir=this_output_dir, T=298, K_in=5,
                                          K_out=120, default_parameters=default_parameters,
                                          removal_duration=args.removal_duration, repeats=args.repeats,
                                          infer_E_rev=True)

    if params is None or len(params) == 0:
        return None
    print(params, scores)

    fits_df = pd.DataFrame(np.column_stack((scores, params)),
                           columns=['score']+param_labels).sort_values(by='score')

    fits_df['well'] = well
    fits_df['protocol'] = protocol

    return fits_df

def main():
    Erev = common.calculate_reversal_potential(T=298, K_in=120, K_out=5)
    print(f"Erev is {Erev}")
    parser = common.get_parser(
        data_reqd=True, description="Fit a given well to the data from each\
        of the protocols. Output the resulting parameters to a file for later use")

    parser.add_argument('--max_iterations', '-i', type=int, default="100000")
    parser.add_argument('--repeats', type=int, default=8)
    parser.add_argument('--wells', '-w', type=str, default=[], nargs='+')
    parser.add_argument('--protocols', type=str, default=[], nargs='+')
    parser.add_argument('--removal_duration', '-r', default=5, type=int)
    parser.add_argument('--cores', '-c', default=1, type=int)

    param_labels = BeattieModel().parameter_labels

    global args
    args = parser.parse_args()

    global output_dir
    output_dir = args.output

    output_dir = common.setup_output_directory(args.output, f"fitting_{args.removal_duration:.2f}_removed")

    regex = re.compile("^newtonrun4-([a-z|A-Z|0-9]*)-([A-Z][0-9][0-9]).csv$")

    if len(args.wells) == 0:
        args.wells = common.get_all_wells_in_directory(args.data_directory, regex=regex, group=1)

    if len(args.protocols) == 0:
        protocols = common.get_protocol_list()
    else:
        protocols = args.protocols

    pool = multiprocessing.Pool(args.cores)

    print(args.wells, protocols)

    tasks = []
    protocols_list = []
    for f in filter(regex.match, os.listdir(args.data_directory)):
        protocol, well = re.search(regex, f).groups()
        if protocol not in protocols or well not in args.wells:
            continue
        else:
            tasks.append((protocol, well))
            protocols_list.append(protocol)

    print(f"fitting tasks are {tasks}")

    assert len(tasks) > 0, "no valid protocol/well combinations provided"

    protocols_list = np.unique(protocols_list)

    fitting_df = pd.concat(pool.starmap(fit_func, tasks))
    print("=============\nfinished fitting\n=============")

    wells_rep = [task[1] for task in tasks]
    protocols_rep = [task[0] for task in tasks]

    best_param_locs = []
    for protocol in np.unique(protocols_list):
        for well in np.unique(wells_rep):
            sub_df = fitting_df[(fitting_df['well'] == well)
                                & (fitting_df['protocol'] == protocol)]

            # Get min score
            best_param_locs.append(sub_df.score.idxmin())


    print(fitting_df)

    print(best_param_locs)

    params_df = fitting_df.loc[best_param_locs]

    print(params_df)

    model = BeattieModel()
    predictions_df = []

    trace_fig = plt.figure(figsize=(16, 12))
    trace_ax = trace_fig.subplots()

    all_models_fig = plt.figure(figsize=(16, 12))
    all_models_ax = all_models_fig.subplots()

    for sim_protocol in np.unique(protocols_list):
        prot_func, tstart, tend, tstep, desc = common.get_ramp_protocol_from_csv(sim_protocol)
        model.voltage = prot_func
        full_times = pd.read_csv(os.path.join(args.data_directory,
                                         f"newtonrun4-{sim_protocol}-times.csv"))['time'].values.flatten()

        voltages = np.array([prot_func(t) for t in full_times])
        spikes, _ = common.detect_spikes(full_times, voltages, 10)
        times, _, indices = common.remove_spikes(full_times, voltages, spikes, args.removal_duration)
        voltages = voltages[indices]

        model = BeattieModel(prot_func,
                             times=times,
                             Erev=Erev)

        for well in params_df['well'].unique():
            full_data = common.get_data(well, sim_protocol, args.data_directory)
            data = full_data[indices]

            # Probably not worth compiling solver
            model.protocol_description = desc
            Erev = common.infer_reversal_potential(sim_protocol, full_data, full_times)
            solver = model.make_forward_solver_current(njitted=False)

            for protocol_fitted in params_df['protocol'].unique():
                df = params_df[params_df.well == well]
                df = df[df.protocol == protocol_fitted]

                if df.empty:
                    continue

                row = df.values
                params = row[0, 0:-3].astype(np.float64)

                sub_dir = os.path.join(output_dir, f"{well}_{sim_protocol}_predictions")
                if not os.path.exists(sub_dir):
                    os.makedirs(sub_dir)

                prediction = solver(params)

                if not np.all(np.isfinite(prediction)):
                    continue

                RMSE = np.sqrt(np.mean((data - prediction)**2))
                predictions_df.append((well, protocol_fitted, sim_protocol, RMSE))


                # Output trace
                if np.isfinite(prediction).all():
                    trace_ax.plot(times, prediction, label='prediction')

                    trace_ax.set_xlabel("time / ms")
                    trace_ax.set_ylabel("current / nA")
                    trace_ax.plot(times, data, label='data', alpha=0.25, color='grey')
                    trace_ax.legend()
                    trace_fig.savefig(os.path.join(sub_dir, f"{protocol_fitted}_fit_predition.png"))
                    trace_ax.cla()

                    all_models_ax.plot(times, prediction, label=protocol_fitted)

            all_models_ax.set_xlabel("time / ms")
            all_models_ax.set_ylabel("current / nA")
            all_models_ax.plot(times, data, color='grey', alpha=0.5, label='data')
            all_models_ax.legend()
            all_models_ax.set_title(f"{well} {sim_protocol} fits comparison")
            all_models_ax.set_xlabel(f"Time /ms")
            all_models_ax.set_ylabel(f"Current / nA")
            all_models_fig.savefig(os.path.join(sub_dir, "all_fits.png"))
            all_models_ax.cla()

    predictions_df = pd.DataFrame(np.array(predictions_df), columns=['well', 'fitting_protocol',
                                                                     'validation_protocol',
                                                                     'RMSE'])
    print(predictions_df)

    predictions_df.to_csv(os.path.join(output_dir, "predictions_df.csv"))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
