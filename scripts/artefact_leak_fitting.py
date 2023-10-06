#! /usr/bin/env python3

import argparse
import multiprocessing
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

import markovmodels
from markovmodels.ArtefactModel import ArtefactModel
from markovmodels.model_generation import make_model_of_class
from subtract_leak import subtract_leak
# from numba import njit
from quality_control.leak_fit import fit_leak_lr

import matplotlib
matplotlib.use('Agg')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("qc_estimates_file")
    parser.add_argument("subtraction_results_file")
    parser.add_argument("selection_file")
    parser.add_argument('traces_directory')
    parser.add_argument('--ramp_start', type=float, default=300)
    parser.add_argument('--ramp_end', type=float, default=600)
    parser.add_argument("--experiment_name")
    parser.add_argument("--parameters", default=None)
    parser.add_argument("-w", "--wells", nargs='+')
    parser.add_argument("--model", default='model3')
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--no_plot", action='store_true')
    parser.add_argument('-P', '--protocols', nargs='+', default=['staircaseramp'])
    parser.add_argument('--noise', default=0.00, type=float)
    parser.add_argument('--Erev', '-e', type=float)
    parser.add_argument('--cpus', '-c', default=1, type=int)
    parser.add_argument('--use_hybrid_solver', action='store_true')
    parser.add_argument('--sampling_frequency', default=0.1, type=float)
    parser.add_argument('--figsize', type=int, nargs=2, default=[8, 12])

    global args
    args = parser.parse_args()

    if not args.wells:
        args.wells = []

    Erev = markovmodels.utilities.calculate_reversal_potential()\
        if args.Erev is None\
        else args.Erev

    global output_dir
    output_dir = markovmodels.utilities.setup_output_directory(args.output, 'artefact_leak_fitting_%s' % args.model)

    global parameters
    model = markovmodels.model_generation.make_model_of_class(args.model)
    if args.parameters is not None:
        param_labels = model.get_parameter_labels()
        parameters = pd.read_csv(args.parameters)[param_labels].values[0, :]
    else:
        parameters = model.get_default_parameters()

    with open(os.path.join(args.selection_file)) as fin:
        global passed_wells
        passed_wells = fin.read().splitlines()

    qc_df = pd.read_csv(args.qc_estimates_file)
    qc_df = qc_df[(qc_df.drug == 'before')
                  & (qc_df.protocol.isin(['staircaseramp1', 'staircaseramp2']))
                  & (qc_df.well.isin(passed_wells))
                  ][['Rseries', 'Cm', 'protocol', 'sweep', 'well']]

    leak_df = pd.read_csv(args.subtraction_results_file)
    leak_df = leak_df[(leak_df.protocol.isin(['staircaseramp1']))
                        & (leak_df.well.isin(passed_wells))][['pre-drug leak conductance',
                                                              'pre-drug leak reversal',
                                                              'protocol', 'sweep', 'well']]

    leak_df = leak_df.set_index(['protocol', 'well', 'sweep']).sort_index()
    qc_df = qc_df.set_index(['protocol', 'well', 'sweep']).sort_index()

    tasks = []
    for (index, leak_row) in leak_df.iterrows():
        protocol, well, sweep = index
        if well not in args.wells and args.wells:
            continue

        if well not in passed_wells:
            continue

        gleak = leak_row['pre-drug leak conductance'] * 1e-3
        Eleak = leak_row['pre-drug leak reversal']

        qc_row = qc_df.loc[protocol, well, sweep][['Rseries', 'Cm']]
        Rseries = qc_row['Rseries'].values[0] * 1e-9
        Cm = qc_row['Cm'].values[0] * 1e9
        noise, gkr = estimate_noise_and_conductance(well, protocol, sweep)

        tasks.append((protocol, well, Rseries, Cm, gleak, Eleak, noise, gkr, Erev))

    with multiprocessing.Pool(args.cpus) as pool:
        res = pool.starmap(generate_data, tasks)

    dfs = []
    for fname, task in zip(res, tasks):
        protocol, well, Rseries, Cm, gleak, Eleak, noise, gkr, Erev = task
        if well not in args.wells and args.wells:
            continue
        _args = args
        _args.data_directory = output_dir
        df = subtract_leak(well, protocol, _args, output_dir)
        df['noise'] = noise
        df['gkr'] = gkr
        df['noise'] = noise
        df['Rseries'] = Rseries
        df['Cm'] = Cm
        df['Erev'] = Erev
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    plot_overlaid_traces(df)
    df.to_csv(os.path.join(output_dir, 'subtract_leak_df.csv'))


def plot_overlaid_traces(df):
    fig = plt.figure(figsize=args.figsize)
    axs = fig.subplots(2)

    for key, row in df.iterrows():

        indices = ['protocol',
                   'well',
                   'Rseries',
                   'Cm',
                   'pre-drug leak conductance',
                   'pre-drug leak reversal',
                   'noise',
                   'gkr',
                   'Erev',
                   'sweep']

        protocol, well, Rseries, Cm, gleak, Eleak, noise, gkr, Erev, sweep = [row[index] for index in indices]

        # Plot original subtracted trace
        subtracted_trace = pd.read_csv(os.path.join(output_dir, "subtracted_traces",
                                                    f"{args.experiment_name}-{protocol}-{well}-sweep{sweep}.csv"))['current'].to_numpy()
        times_df = pd.read_csv(os.path.join(output_dir, 'subtracted_traces',
                                            f"{args.experiment_name}-{protocol}-times.csv"))
        times = times_df['time'].to_numpy().flatten()

        _parameters = parameters.copy()
        _parameters[-1] = gkr

        prot_func, _, desc = markovmodels.voltage_protocols.get_ramp_protocol_from_csv(protocol)

        c_model = make_model_of_class(args.model,
                                      voltage=prot_func, times=times, E_rev=Erev,
                                      default_parameters=_parameters,
                                      protocol_description=desc,
                                      tolerances=[1e-6, 1e-6])

        model = ArtefactModel(c_model, E_leak=Eleak, g_leak=gleak, C_m=Cm, R_s=Rseries)

        states = model.make_hybrid_solver_states(hybrid=False)()
        Vm = states[:, -1].flatten()

        true_IKr = c_model.define_auxiliary_function()(states[:, :-1].T, _parameters, Vm)

        protocol_voltages = np.array([prot_func(t) for t in times])
        ideal_current_known_leak = c_model.SimulateForwardModel()

        axs[0].plot(times, subtracted_trace, label='usual subtraction')
        axs[0].plot(times, true_IKr, label='true current')
        axs[0].plot(times, ideal_current_known_leak, label='ideal-clamp model with known leak')
        axs[0].legend()

        axs[1].plot(times, protocol_voltages, label='Vcmd')
        axs[1].plot(times, Vm, label='Vm')

        if not os.path.exists(os.path.join(output_dir, 'comparison_plots')):
            os.makedirs(os.path.join(output_dir, 'comparison_plots'))

        fig.savefig(os.path.join(output_dir, 'comparison_plots',
                                 f"{protocol}-{well}-{sweep}"))
        for ax in axs:
            ax.cla()


def estimate_noise_and_conductance(well, protocol, sweep):
    # get data
    before_filename = f"{args.experiment_name}-{protocol}-{well}-before-sweep{sweep}.csv"
    before_trace_df = pd.read_csv(os.path.join(args.traces_directory, before_filename))
    before_trace = before_trace_df[before_trace_df.columns[-1]].to_numpy().flatten().astype(np.float64)

    after_filename = f"{args.experiment_name}-{protocol}-{well}-after-sweep{sweep}.csv"
    after_trace_df = pd.read_csv(os.path.join(args.traces_directory, after_filename))
    after_trace = after_trace_df[after_trace_df.columns[-1]].to_numpy().flatten().astype(np.float64)

    times_filename = f"{args.experiment_name}-{protocol}-times.csv"
    times = pd.read_csv(os.path.join(args.traces_directory, times_filename))['time'].to_numpy()
    times = times * 1e3
    prot_func, _, desc = markovmodels.voltage_protocols.get_ramp_protocol_from_csv(protocol)

    protocol_voltages = np.array([prot_func(t) for t in times])
    dt = times[1] - times[0]

    g_leak_before, E_leak_before, _, _, _, x, y = fit_leak_lr(
        protocol_voltages, before_trace, dt=dt,
        ramp_start=args.ramp_start,
        ramp_end=args.ramp_end
    )

    g_leak_after, E_leak_after, _, _, _, x, y = fit_leak_lr(
        protocol_voltages, after_trace, dt=dt,
        ramp_start=args.ramp_start,
        ramp_end=args.ramp_end
    )

    before_corrected = before_trace - g_leak_before * (protocol_voltages - E_leak_before)
    after_corrected = after_trace - g_leak_after * (protocol_voltages - E_leak_after)
    subtracted_trace = before_corrected - after_corrected

    noise = before_trace[:200].std()

    Erev = markovmodels.utilities.calculate_reversal_potential()

    prot_func, _times, desc = markovmodels.voltage_protocols.get_ramp_protocol_from_csv(protocol)

    c_model = make_model_of_class(args.model,
                                  voltage=prot_func, times=times, E_rev=Erev,
                                  default_parameters=parameters,
                                  protocol_description=desc)

    solver = c_model.make_forward_solver_current(njitted=True)

    default_parameters = c_model.get_default_parameters()

    # @njit
    def min_func(g_kr):
        p = default_parameters.copy()
        p[-1] = g_kr
        return np.sum((solver(p) - subtracted_trace) ** 2)

    # Minimise SSE to find best conductance
    res = scipy.optimize.minimize_scalar(min_func, method='bounded', bounds=[0, 1e5])

    gkr = res.x

    # Plot stuff
    if not args.no_plot:
        fig = plt.figure(figsize=args.figsize)
        ax = fig.subplots()

        ax.plot(times, subtracted_trace, label='subtracted trace', color='grey')
        p = c_model.get_default_parameters()
        p[-1] = gkr
        ax.plot(times, solver(p))

    if not os.path.exists(os.path.join(output_dir, "conductance_estimation")):
        os.makedirs(os.path.join(output_dir, "conductance_estimation"))
    fig.savefig(os.path.join(output_dir, "conductance_estimation", f"{well}-{protocol}-sweep{sweep}"))
    plt.close(fig)
    return noise * 1e-3, gkr * 1e-3


def generate_data(protocol, well, Rseries, Cm, gleak, Eleak, noise, gkr, Erev=None):
    Erev = markovmodels.utilities.calculate_reversal_potential()

    prot_func, _times, desc = markovmodels.voltage_protocols.get_ramp_protocol_from_csv(protocol)

    no_samples = int((_times[-1] - _times[0]) / args.sampling_frequency) + 1

    times = np.linspace(_times[0], _times[-1],
                        no_samples)

    if not os.path.exists(os.path.join(output_dir, f"{args.experiment_name}-{protocol}-times.csv")):
        times_df = pd.DataFrame(times.T, columns=('time',))
        times_df.to_csv(os.path.join(output_dir, f"{args.experiment_name}-{protocol}-times.csv"))

    _parameters = parameters.copy()
    _parameters[-1] = gkr

    c_model = make_model_of_class(args.model,
                                  voltage=prot_func, times=times, E_rev=Erev,
                                  default_parameters=_parameters,
                                  protocol_description=desc)

    model = ArtefactModel(c_model, E_leak=Eleak, g_leak=gleak, C_m=Cm, R_s=Rseries)

    solver = model.make_forward_solver_current(njitted=False)
    I_out = solver()
    # voltages = np.array([prot_func(t) for t in times])
    # I_leak = gleak * (voltages - Eleak)
    mean = I_out

    data = np.random.normal(mean, noise, times.shape)

    # Output data
    out_fname = os.path.join(output_dir, f"{args.experiment_name}-{protocol}-{well}-before-sweep1.csv")
    pd.DataFrame(data.T, columns=('current',)).to_csv(out_fname)

    # Assume 0 conductance after drug addition
    p = model.get_default_parameters()
    p[-8] = 0
    data_after = np.random.normal(solver(p), noise, times.shape)
    out_fname = os.path.join(output_dir, f"{args.experiment_name}-{protocol}-{well}-after-sweep1.csv")
    pd.DataFrame(data_after.T, columns=('current',)).to_csv(out_fname)

    if not args.no_plot:
        fig = plt.figure(figsize=(14, 12))
        axs = fig.subplots(3)
        axs[0].plot(times, mean, label='mean')
        axs[0].plot(times, data, label='data', color='grey', alpha=0.5)
        axs[0].legend()
        V_m = model.make_hybrid_solver_states(hybrid=False)()[:, -1]
        axs[1].plot(times, V_m, label='Vm')
        axs[1].set_xlabel('time / ms')
        axs[1].set_ylabel('Vm / mV')
        axs[0].set_ylabel('current / nA')
        axs[2].plot(times, [model.voltage(t) for t in times], label='voltage / mV')
        fig.savefig(os.path.join(output_dir, f"plot-{protocol}plot-{well}.png"))
        plt.close(fig)

    # return the filename for convinience
    return out_fname


if __name__ == "__main__":
    main()
