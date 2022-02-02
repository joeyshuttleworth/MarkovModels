#!/usr/bin/env python3

import multiprocessing
import regex as re
from MarkovModels import common
from MarkovModels.BeattieModel import BeattieModel

import os
import pandas as pd
import numpy as np

def fit_func(protocol, well):
    default_parameters = None
    this_output_dir = os.path.join(output_dir, f"{protocol}_{well}")

    params, scores = common.fit_well_data(BeattieModel, well, protocol, args.data_directory,
                                          args.max_iterations, output_dir=this_output_dir, T=298, K_in=5,
                                          K_out=120, default_parameters=default_parameters,
                                          removal_duration=args.removal_duration, repeats=args.repeats,
                                          infer_E_rev=True)
    return params[np.argmin(scores), :].flatten()

def main():
    Erev = common.calculate_reversal_potential(T=298, K_in=120, K_out=5)
    print(f"Erev is {Erev}")
    parser = common.get_parser(
        data_reqd=True, description="Fit a given well to the data from each of the protocols. Output the resulting parameters to a file for later use")
    parser.add_argument('--max_iterations', '-i', type=int, default="100000")
    parser.add_argument('--repeats', type=int, default=8)
    parser.add_argument('--wells', '-w', type=str, default=[], nargs='+')
    parser.add_argument('--protocols', type=str, default=[], nargs='+')
    parser.add_argument('--removal_duration', '-r', default=5, type=int)

    global args
    args = parser.parse_args()

    global output_dir
    output_dir = args.output

    output_dir = common.setup_output_directory(None, f"fitting_{args.removal_duration:.2f}_removed")

    regex = re.compile("^newtonrun4-([a-z|A-Z|0-9]*)-([A-Z][0-9][0-9]).csv$")

    if len(args.wells) == 0:
        args.wells = common.get_all_wells_in_directory(args.data_directory, regex=regex, group=1)

    if len(args.protocols) == 0:
        protocols = common.get_protocol_list()
    else:
        protocols = args.protocols

    pool = multiprocessing.Pool(processes=os.cpu_count())

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

    print(tasks)
    fitted_params_list = np.row_stack(pool.starmap(fit_func, tasks))
    print(fitted_params_list)

    wells_rep = [task[1] for task in tasks]
    protocols_rep = [task[0] for task in tasks]

    print(wells_rep)

    param_names = ['p%i' % i for i in range(1,9)] + ['g_kr']

    params_df = pd.DataFrame(fitted_params_list, columns=param_names)

    params_df['well'] = wells_rep
    params_df['protocol'] = protocols_rep
    print(params_df)

    print("=============\nfinished fitting\n=============")

    model = BeattieModel()
    predictions_df = []

    wells = args.wells
    for well in wells:
        for protocol_fitted in protocols_list:
            df = params_df[params_df.well == well][params_df.protocol == protocol_fitted]
            row = df.values
            params = row[0, 0:-2].astype(np.float)
            print(params)

            for sim_protocol in protocols_list:
                prot_func, t_start, t_end, t_step, desc = common.get_ramp_protocol_from_csv(sim_protocol)
                model.protocol_description = desc
                model.voltage = prot_func

                times = pd.read_csv(os.path.join(args.data_directory, f"newtonrun4-{sim_protocol}-times.csv"))['time'].values
                voltages = np.array([prot_func(t) for t in times])
                spikes, _ = common.detect_spikes(times, voltages, 10)
                times, _, indices = common.remove_spikes(times, voltages, spikes, args.removal_duration)
                voltages = voltages[indices]
                model.times = times

                prediction = model.SimulateForwardModel(params)
                data = common.get_data(well, sim_protocol, args.data_directory)
                data = data[indices]

                RMSE = np.sqrt(np.mean((data - prediction)**2))

                predictions_df.append((well, protocol_fitted, sim_protocol, RMSE))

    predictions_df = pd.DataFrame(np.array(predictions_df), columns=['well', 'fitting_protocol',
                                                                     'validation_protocol',
                                                                     'RMSE'])
    print(predictions_df)

    predictions_df.to_csv(os.path.join(output_dir, "predictions_df.csv"))

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
