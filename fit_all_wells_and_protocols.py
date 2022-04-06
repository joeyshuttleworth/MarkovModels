#!/usr/bin/env python3

import multiprocessing
import regex as re
import matplotlib.pyplot as plt
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

Erev = common.calculate_reversal_potential(T=T, K_in=K_in, K_out=K_out)

def fit_func(protocol, well, model_class, E_rev=Erev):
    default_parameters = None
    this_output_dir = os.path.join(output_dir, f"{protocol}_{well}")

    res_df = common.fit_well_data(model_class, well, protocol, args.data_directory,
                                  args.max_iterations, output_dir=this_output_dir, T=298, K_in=5,
                                  K_out=120, default_parameters=default_parameters,
                                  removal_duration=args.removal_duration, repeats=args.repeats,
                                  infer_E_rev=True, experiment_name=experiment_name)

    res_df['well'] = well
    res_df['protocol'] = protocol

    return res_df


def mcmc_func(protocol, well, model_class, initial_params=None):


    # Ignore files that have been commented out
    voltage_func, t_start, t_end, t_step, protocol_desc = common.get_ramp_protocol_from_csv(protocol)

    data = common.get_data(well, protocol, args.data_directory, experiment_name)

    times = pd.read_csv(os.path.join(args.data_directory, f"{experiment_name}-{protocol}-times.csv"))['time'].values

    voltages = np.array([voltage_func(t) for t in times])

    model = model_class(voltage=voltage_func, parameters=initial_params,
                        Erev=Erev, protocol_description=protocol_desc)

    if initial_params is None:
        initial_params = model.get_default_parameters()

    solver = model.make_hybrid_solver_current()

    if np.any(~np.isfinite(solver(initial_params))):
        initial_params = model.get_default_parameters()

    sigma2 = solver()[0:100].std()**2

    spike_times, spike_indices = common.detect_spikes(times, voltages, threshold=10)

    _, _, indices = common.remove_spikes(times, voltages, spike_times,
                                         time_to_remove=args.removal_duration)

    return common.compute_mcmc_chains(solver, times, indices, data,
                                      chain_length=args.chain_length,
                                      starting_parameters=initial_params,
                                      sigma2=sigma2, burn_in=0,
                                      likelihood_func=None)


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
    parser.add_argument('--removal_duration', '-r', default=5, type=float)
    parser.add_argument('--cores', '-c', default=1, type=int)
    parser.add_argument('--model', '-m', default='Beattie', type=str)
    parser.add_argument('--experiment_name', default='newtonrun4', type=str)
    parser.add_argument('--no_chains', '-N', default=0, help='mcmc chains to run', type=int)
    parser.add_argument('--chain_length', '-l', default=5000, help='mcmc chains to run', type=int)

    global args
    args = parser.parse_args()

    global output_dir
    output_dir = args.output

    global experiment_name
    experiment_name = args.experiment_name

    output_dir = common.setup_output_directory(args.output, f"fitting_{args.removal_duration:.2f}_removed_{args.model}")

    global model_class
    if args.model == 'Beattie':
        model_class = BeattieModel
    elif args.model == 'Kemp':
        model_class = KempModel
    elif args.model == 'CO':
        model_class = ClosedOpenModel
    else:
        assert False

    regex = re.compile(f"^{experiment_name}-([a-z|A-Z|0-9]*)-([A-Z][0-9][0-9]).csv$")

    if len(args.wells) == 0:
        args.wells = common.get_all_wells_in_directory(args.data_directory, experiment_name)

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
            tasks.append([protocol, well, model_class])
            protocols_list.append(protocol)

    print(f"fitting tasks are {tasks}")

    assert len(tasks) > 0, "no valid protocol/well combinations provided"

    protocols_list = np.unique(protocols_list)
    res = pool.starmap(fit_func, tasks)
    print(res)

    if args.chain_length > 0 and args.no_chains > 0:
        mcmc_dir = os.path.join(output_dir, 'mcmc_samples')

        if not os.path.exists(mcmc_dir):
            os.makedirs(mcmc_dir)

        for res_df, task in zip(res, tasks):
            # Select best score
            mle_row = res_df[res_df.score == res_df.score.max()]
            param_labels = task[2]().parameter_labels
            mle = mle_row[param_labels].values[0, :].flatten()
            if np.all(np.isfinite(mle)):
                print(mle)
                task.append(mle)

        print(tasks)
        # Do MCMC
        mcmc_res = pool.starmap(mcmc_func, tasks)
        for samples, task in zip(mcmc_res, tasks):
            protocol, well, model_class = task[0], task[1], task[2]
            model_name = model_class().get_model_name()

            np.save(os.path.join(mcmc_dir,
                                 f"mcmc_{model_name}_{well}_{protocol}.npy"), samples)

    fitting_df = pd.concat(res)

    print("=============\nfinished fitting\n=============")

    wells_rep = [task[1] for task in tasks]
    # protocols_rep = [task[0] for task in tasks]

    fitting_df.to_csv(os.path.join(output_dir, "fitting.csv"))

    best_param_locs = []
    for protocol in np.unique(protocols_list):
        for well in np.unique(wells_rep):
            sub_df = fitting_df[(fitting_df['well'] == well)
                                & (fitting_df['protocol'] == protocol)]

            # Get index of min score
            best_param_locs.append(sub_df.score.idxmin())

    params_df = fitting_df.loc[best_param_locs]

    params_df.to_csv(os.path.join(output_dir, "best_fitting.csv"))

    # Plot predictions
    predictions_df = []

    trace_fig = plt.figure(figsize=(16, 12))
    trace_axs = trace_fig.subplots(2)

    all_models_fig = plt.figure(figsize=(16, 12))
    all_models_axs = all_models_fig.subplots(2)
    for sim_protocol in np.unique(protocols_list):
        prot_func, tstart, tend, tstep, desc = common.get_ramp_protocol_from_csv(sim_protocol)
        times = pd.read_csv(os.path.join(args.data_directory,
                                         f"{experiment_name}-{sim_protocol}-times.csv"))['time'].values.flatten()

        model = model_class(prot_func,
                            times=times,
                            Erev=Erev)

        voltages = np.array([prot_func(t) for t in times])

        for well in params_df['well'].unique():
            full_data = common.get_data(well, sim_protocol, args.data_directory, experiment_name=experiment_name)
            data = full_data

            # Probably not worth compiling solver
            model.protocol_description = desc
            Erev = common.infer_reversal_potential(sim_protocol, full_data, times)
            solver = model.make_forward_solver_current(njitted=False)

            for protocol_fitted in params_df['protocol'].unique():
                df = params_df[params_df.well == well]
                df = df[df.protocol == protocol_fitted]

                if df.empty:
                    continue

                param_labels = model.parameter_labels
                params = df.iloc[0][param_labels].values\
                                                 .astype(np.float64)\
                                                 .flatten()
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
                    trace_axs[0].plot(times, prediction, label='prediction')

                    trace_axs[1].set_xlabel("time / ms")
                    trace_axs[0].set_ylabel("current / nA")
                    trace_axs[0].plot(times, data, label='data', alpha=0.25, color='grey')
                    trace_axs[0].legend()

                    trace_axs[1].plot(times, voltages)
                    trace_axs[1].set_ylabel('voltage / mV')
                    trace_fig.savefig(os.path.join(sub_dir, f"{protocol_fitted}_fit_predition.png"))

                    for ax in trace_axs:
                        ax.cla()

                    all_models_axs[0].plot(times, prediction, label=protocol_fitted)

            all_models_axs[1].set_xlabel("time / ms")
            all_models_axs[0].set_ylabel("current / nA")
            all_models_axs[0].plot(times, data, color='grey', alpha=0.5, label='data')
            all_models_axs[0].legend()
            all_models_axs[0].set_title(f"{well} {sim_protocol} fits comparison")
            all_models_axs[0].set_ylabel("Current / nA")

            all_models_axs[1].plot(times, voltages)
            all_models_axs[1].set_ylabel('voltage / mV')

            all_models_fig.savefig(os.path.join(sub_dir, "all_fits.png"))

            for ax in all_models_axs:
                ax.cla()

    predictions_df = pd.DataFrame(np.array(predictions_df), columns=['well', 'fitting_protocol',
                                                                     'validation_protocol',
                                                                     'RMSE'])
    print(predictions_df)
    predictions_df.to_csv(os.path.join(output_dir, "predictions_df.csv"))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
