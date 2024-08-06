#! /usr/bin/env python3

import argparse
import multiprocessing
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

import markovmodels
import pcpostprocess
from markovmodels.ArtefactModel import ArtefactModel, no_artefact_parameters
from markovmodels.model_generation import make_model_of_class
import seaborn as sns
    if use_artefacts:
        c_param_labels = model.channel_model.get_parameter_labels()
    else:
        c_param_labels = param_labels
from numba import njit
from quality_control.leak_fit import fit_leak_lr
import markovmodels.utilities as utilities
from markovmodels.fitting import infer_reversal_potential, find_V_off, fit_leak_parameters_with_artefact, _find_conductance, adjust_kinetics
from markovmodels.voltage_protocols import get_ramp_protocol_from_json, make_voltage_function_from_description
from markovmodels.utilities import calculate_reversal_potential, get_data

import matplotlib
from matplotlib.pyplot import cycler
matplotlib.use('Agg')

# params_for_Erev = np.loadtxt(os.path.join('data', 'Beattie_Sinusoidal_params.csv'),
#                              delimiter=', ').flatten().astype(np.float64)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('postprocess_data_dir')
    parser.add_argument("--experiment_name", default='25112022_MW')
    parser.add_argument("--parameters", default=None)
    parser.add_argument("-w", "--wells", nargs='+')
    parser.add_argument("--ignore_wells", nargs='+', default=['M06'])
    parser.add_argument("--sweeps", nargs='+', default=[])
    parser.add_argument("--model", default='model3')
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--no_plot", action='store_true')
    parser.add_argument('-P', '--protocols', nargs='+', default=['staircaseramp'])
    parser.add_argument('--noise', default=0.00, type=float)
    parser.add_argument('--reversal', '-e', type=float, default=-89.5)
    parser.add_argument('--cpus', '-c', default=1, type=int)
    parser.add_argument('--use_hybrid_solver', action='store_true')
    parser.add_argument('--sampling_frequency', default=0.1, type=float)
    parser.add_argument('--figsize', type=int, nargs=2, default=[5.3, 6])
    parser.add_argument('--no_noise', action='store_true')
    parser.add_argument('--removal_duration', type=float, default=5.0)

    global args
    args = parser.parse_args()

    if not args.sweeps:
        args.sweeps = ['0', '1']

    if not args.wells:
        args.wells = []

    Erev = calculate_reversal_potential()\
    if args.reversal is None\
    else args.reversal

    global output_dir
    output_dir = markovmodels.utilities.setup_output_directory(args.output, 'artefact_leak_fitting_%s' % args.model)

    global parameters
    model = markovmodels.model_generation.make_model_of_class(args.model)
    if args.parameters is not None:
        param_labels = model.get_parameter_labels()
        parameters = pd.read_csv(args.parameters)[param_labels].values[0, :]
    else:
        parameters = model.get_default_parameters()

    selection_file = os.path.join(args.postprocess_data_dir, 'passed_wells.txt')
    with open(os.path.join(selection_file)) as fin:
        global passed_wells
        passed_wells = fin.read().splitlines()
        passed_wells = [w for w in passed_wells if w not in args.ignore_wells]

    subtraction_results_file = os.path.join(args.postprocess_data_dir,
                                            'subtraction_qc.csv')

    qc_df = pd.read_csv(subtraction_results_file)
    qc_df = qc_df[(qc_df.protocol.isin(['staircaseramp1']))
                      & (qc_df.well.isin(passed_wells))]

    leak_df = qc_df.set_index(['protocol', 'well', 'sweep']).sort_index()

    tasks = []
    for index, leak_row in leak_df.iterrows():
        protocol, well, sweep = index
        if well not in args.wells and args.wells:
            continue

        if well not in passed_wells:
            continue

        if str(sweep) not in args.sweeps:
            continue

        gleak = leak_row['gleak_before']
        Eleak = leak_row['E_leak_before']
        Rseries = leak_row['Rseries'] * 1e-9
        Cm = leak_row['Cm'] * 1e9
        E_obs = leak_row['E_rev']
        noise, gkr = estimate_noise_and_conductance(well, protocol, sweep,
                                                    gleak, Eleak, Rseries, Cm, args.reversal)

        if args.no_noise:
            noise = 0
        tasks.append((protocol, well, Rseries, Cm, gleak, Eleak, noise, gkr, E_obs, Erev, args,
                      output_dir))

    print(f"tasks are {tasks}")
    with multiprocessing.Pool(args.cpus) as pool:
        res = pool.starmap(generate_data, tasks)

    dfs = []
    for fname, task in zip(res, tasks):
        protocol, well, Rseries, Cm, gleak, Eleak, noise, gkr, E_obs, Erev, _, _ = task
        if well not in args.wells and args.wells:
            continue
        _args = parser.parse_args()
        _args.data_directory = output_dir
        _args.Erev = _args.reversal
        print(_args)
        df = subtract_leak(well, protocol, _args, output_dir)
        df['noise'] = noise
        df['gkr'] = gkr
        df['noise'] = noise
        df['Rseries'] = Rseries
        df['Cm'] = Cm
        df['Erev'] = Erev
        if 'sweep' not in df:
            df['sweep'] = 0
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(output_dir, 'subtract_leak_df.csv'))

    plot_overlaid_traces(df)
    do_scatterplots(df, leak_df)

    compare_synth_real_postprocess_data(df, leak_df)


def compare_synth_real_postprocess_data(df, leak_df):
    fig = plt.figure(figsize=args.figsize, constrainted_layout=True)
    ax = fig.subplots()

    leak_df = leak_df.reset_index()

    plot_dir = os.path.join(output_dir, "compare_real_synth_traces")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    labels = []
    for protocol in df.protocol.unique():
        times_df = pd.read_csv(os.path.join(output_dir, 'subtracted_traces',
                                            f"{args.experiment_name}-{protocol}-times.csv"))
        times = times_df.to_numpy().flatten()

        all_synth_traces = []
        all_real_traces = []
        wells = []
        sweeps = []
        for well in df.well.unique():
            for sweep in df.sweep.unique():
                # Get parameters
                sub_df = df[(df.well == well) & (df.protocol == protocol) &
                            (df.sweep == sweep)].copy()
                sub_leak_df = leak_df[(leak_df.well == well) & (leak_df.protocol == protocol) &
                                      (df.sweep == sweep)].copy()
                if sub_leak_df.shape[0] == 0:
                    continue

                wells.append(well)
                sweeps.append(sweep)

                gleak, Eleak = sub_leak_df[['pre-drug leak conductance', 'pre-drug leak reversal']]
                gkr, noise, Cm, Rseries = sub_df[['gkr', 'noise', 'Cm', 'Rseries']]

                # Get synth trace
                synth_sub_trace = pd.read_csv(os.path.join(output_dir, "subtracted_traces",
                                                           f"{args.experiment_name}-{protocol}-{well}-sweep{sweep}.csv")).to_numpy()

                subtracted_traces_dir = os.path.join(args.postprocess_data_dir, 'traces')
                real_sub_trace = pd.read_csv(os.path.join(subtracted_traces_dir,
                                                          f"{args.experiment_name}-{protocol}-{well}-sweep{sweep}.csv")).to_numpy()

                # Plot both traces normalised
                ax.plot(times, synth_sub_trace/synth_sub_trace.std(), alpha=.5, label='synth data')

                times_filename = f"{args.experiment_name}-staircaseramp1-times.csv"
                real_times = pd.read_csv(os.path.join(subtracted_traces_dir, times_filename))['time'].to_numpy().flatten()
                ax.plot(real_times, real_sub_trace/real_sub_trace.std(), alpha=.5, label='real data')

                ax.set_ylabel('normalised post-processed current')
                ax.set_xlabel('times (ms)')
                ax.legend()

                fig.savefig(os.path.join(plot_dir, f"{protocol}-{well}-sweep{sweep}-postprocess"))
                ax.cla()
                label = f"{protocol}-{well}-{sweep}"
                labels.append(label)
                all_synth_traces.append(synth_sub_trace.flatten())
                all_real_traces.append(real_sub_trace.flatten())

        deflection_plot_dir = os.path.join(output_dir, 'deflection_plots')
        if not os.path.exists(deflection_plot_dir):
            os.makedirs(deflection_plot_dir)

        all_real_traces = np.vstack(all_real_traces)
        all_synth_traces = np.vstack(all_synth_traces)

        # Find the average normalised real trace
        average_real_trace = np.mean(all_real_traces.T / all_real_traces.std(axis=1), axis=1).T
        average_synth_trace = np.mean(all_synth_traces.T / all_synth_traces.std(axis=1), axis=1).T

        for i, (well, sweep) in enumerate(zip(wells, sweeps)):
            trace_name = f"{well}_{sweep}_deflection_plots"
            real_deflection = (all_real_traces[i, :].T / all_real_traces[i, :].std()).flatten().T - average_real_trace
            synth_deflection = (all_synth_traces[i, :].T / all_synth_traces[i, :].std()).flatten().T - average_synth_trace
            ax.plot(times, real_deflection, label='deflection from mean (real data)')
            ax.plot(times, synth_deflection, label='deflection from mean (synth data)')
            ax.legend()
            fig.savefig(os.path.join(deflection_plot_dir, trace_name))
            ax.cla()

    plt.close(fig)


def do_scatterplots(df, qc_df):
    fig = plt.figure(figsize=args.figsize, constrainted_layout=True)
    ax = fig.subplots()

    df = df.reset_index().set_index(['protocol', 'well', 'sweep'])
    leak_df = qc_df.reset_index().set_index(['protocol', 'well', 'sweep'])
    # Values used for data generation
    df['gleak'] = leak_df['pre-drug leak conductance']
    df['Eleak'] = leak_df['pre-drug leak reversal']
    df[r'$\hat g_\mathrm{leak}$'] = df['pre-drug leak conductance']
    df[r'$\hat E_\mathrm{leak}$'] = df['pre-drug leak reversal']

    df[r'$\hat{E_\text{obs}}$'] = df['fitted_E_rev']
    df[r'$E_\text{obs}$'] = qc_df['fitted_E_rev']

    sns.scatterplot(data=df, x='pre-drug leak conductance',
                    y=r'$\hat g_\mathrm{leak}$', ax=ax)

    xs = np.quantile(df['pre-drug leak conductance'], (0, 1))
    ax.plot(xs, xs, '--', color='grey')
    fig.savefig(os.path.join(output_dir, "g_leak_scatterplot"))

    sns.scatterplot(data=df, x='pre-drug leak reversal',
                    y=r'$\hat E_\mathrm{leak}$', ax=ax)
    xs = np.quantile(df['pre-drug leak reversal'], (0, 1))
    ax.plot(xs, xs, '--', color='grey')
    fig.savefig(os.path.join(output_dir, "E_leak_scatterplot"))

    # Now do matrix scatterplot
    df = df[['gleak', 'Eleak', 'Rseries', 'Cm', 'gkr']]

    df['passed QC'] = df.index.get_level_values('well').isin(passed_wells)
    df = df[df.index.get_level_values('protocol') == 'staircaseramp1']
    df = df[df.index.get_level_values('sweep') == 1]
    grid = sns.pairplot(df, hue='passed QC', hue_order=[False, True])

    grid.savefig(os.path.join(output_dir, "QC_estimates_scatter_matrix.pdf"))
    plt.close(grid.figure)


def plot_overlaid_traces(df):
    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    axs = fig.subplots(2)

    for key, row in df.iterrows():

        indices = ['protocol',
                   'well',
                   'Rseries',
                   'Cm',
                   'pre-drug leak conductance',
                   'pre-drug leak reversal',
                   'post-drug leak conductance',
                   'post-drug leak reversal',
                   'noise',
                   'gkr',
                   'Erev',
                   'sweep']

        protocol, well, Rseries, Cm, gleak, Eleak, gleak_after, Eleak_after, noise, gkr, Erev, sweep = [row[index] for index in indices]

        # gleak, gleak_after = gleak * 1e-3, gleak_after * 1e-3

        # Plot original subtracted trace
        subtracted_trace = pd.read_csv(os.path.join(output_dir, "subtracted_traces",
                                                    f"{args.experiment_name}-{protocol}-{well}-sweep{sweep}.csv")).to_numpy()
        times_df = pd.read_csv(os.path.join(output_dir, 'subtracted_traces',
                                            f"{args.experiment_name}-{protocol}-times.csv"))
        times = times_df.to_numpy().flatten()

        _parameters = parameters.copy()
        _parameters[-1] = gkr

        prot_func, desc = get_protocol(protocol, args)

        c_model = make_model_of_class(args.model,
                                      voltage=prot_func, times=times, E_rev=Erev,
                                      default_parameters=_parameters,
                                      protocol_description=desc)

        model = ArtefactModel(c_model, E_leak=Eleak, g_leak=gleak, C_m=Cm, R_series=Rseries)

        solver = model.make_hybrid_solver_states(hybrid=False,
                                                 return_var='I_out')
        states = solver(_parameters)
        Vm = states[:, -1].flatten()

        # Plot reversal estimation with V=Vm
        infer_reversal_potential_with_artefact('staircaseramp1', times, subtracted_trace,
                                               'model3', model.get_default_parameters(), Erev,
                                               removal_duration=5,
                                               output_path=os.path.join(output_dir, 'reversal_plots',
                                                                        f"{protocol}-{well}-sweep{sweep}_Vm"),
                                               plot=True)

        # Plot reversal estimate again but with gKr = 0
        no_g_p = model.get_default_parameters()
        gkr_index = c_model.GKr_index
        no_g_p[gkr_index] = .0
        Vm2 = solver(no_g_p)[:, -1].flatten()

        true_IKr = model.define_auxiliary_function()(states.T, model.get_default_parameters(), Vm)

        before_trace = pd.read_csv(os.path.join(output_dir,
                                                f"{args.experiment_name}-{protocol}"
                                                f"-{well}-before-sweep{sweep}.csv")).to_numpy().flatten()

        before_corrected_Vm = before_trace - gleak * (Vm - Eleak)

        gkr_index = c_model.GKr_index
        p = model.get_default_parameters()
        p[gkr_index] = 0.0
        Vm = solver(p)[:, -1].flatten()
        after_trace = pd.read_csv(os.path.join(output_dir,
                                               f"{args.experiment_name}-{protocol}"
                                               f"-{well}-after-sweep{sweep}.csv")).to_numpy().flatten()
        after_corrected_Vm = after_trace - gleak_after * (Vm - Eleak_after)

        subtracted_Vm = before_corrected_Vm - after_corrected_Vm

        times = times.flatten()
        protocol_voltages = np.array([prot_func(t) for t in times])
        ideal_current_known_leak = c_model.SimulateForwardModel()

        axs[0].plot(times, subtracted_trace, label='subtracted trace (Vcmd)', alpha=.5)
        axs[0].plot(times, subtracted_Vm, label='subtracted trace (Vm)', alpha=.5)
        axs[0].plot(times, true_IKr, label='true current')
        axs[0].plot(times, ideal_current_known_leak, label='ideal-clamp model')
        axs[0].legend()

        states = solver()
        Vm = states[:, -1].flatten()

        axs[1].plot(times, protocol_voltages, label='Vcmd')
        axs[1].plot(times, Vm, label='Vm')

        if not os.path.exists(os.path.join(output_dir, 'comparison_plots')):
            os.makedirs(os.path.join(output_dir, 'comparison_plots'))

        fig.savefig(os.path.join(output_dir, 'comparison_plots',
                                 f"{protocol}-{well}-{sweep}"))
        for ax in axs:
            ax.cla()

    # Now overlay all traces (normalised)
    for protocol in df.protocol.unique():
        sub_df = df[df.protocol == protocol]

        # Plot original subtracted trace
        subtracted_trace = pd.read_csv(os.path.join(output_dir, "subtracted_traces",
                                                    f"{args.experiment_name}-{protocol}-{well}-sweep{sweep}.csv")).to_numpy()

        times_df = np.loadtxt(os.path.join(output_dir, 'subtracted_traces',
                                           f"{args.experiment_name}-{protocol}-times.csv"))
        times = times_df.to_numpy().flatten()

        _parameters = parameters.copy()
        _parameters[-1] = gkr

        prot_func, desc = get_protocol(protocol, args)

        voltages = np.array([prot_func(t) for t in times])

        axs[1].plot(times, voltages, color='black')

        c_model = make_model_of_class(args.model,
                                      voltage=prot_func, times=times, E_rev=Erev,
                                      default_parameters=_parameters,
                                      protocol_description=desc,
                                      tolerances=[1e-6, 1e-6])
        n_traces = sub_df.shape[0]
        seaborn_palette = sns.color_palette("husl", n_traces)

        reference_IKr = c_model.SimulateForwardModel()
        reference_IKr = reference_IKr/reference_IKr.std()
        fig.clf()
        axs = fig.subplots(3)
        axs[0].plot(times, reference_IKr, "--", color='grey')

        for index, row in sub_df.iterrows():
            protocol, well, Rseries, Cm, gleak, Eleak, _, _, noise, gkr, Erev, sweep = [row[index] for index in indices]

            # gleak = gleak * 1e-3
            model = ArtefactModel(c_model, E_leak=Eleak, g_leak=gleak, C_m=Cm, R_series=Rseries)

            state_solver = model.make_hybrid_solver_states(hybrid=False,
                                                           return_var='I_out')
            states = state_solver()

            Vm = states[:, -1].flatten()

            colour = seaborn_palette[index]

            subtracted_trace = pd.read_csv(os.path.join(output_dir, "subtracted_traces",
                                                        f"{args.experiment_name}-{protocol}-{well}-sweep{sweep}.csv")).to_numpy()

            I_Kr = c_model.define_auxiliary_function()(states[:, :-1].T, _parameters, Vm)
            I_Kr = I_Kr / I_Kr.std()

            axs[0].plot(times*1e-3, subtracted_trace / subtracted_trace.std(),
                        label=f"{well} sweep {sweep}", color=colour, alpha=.25)
            axs[1].plot(times*1e-3, I_Kr, label=r'$I_\mathrm{Kr}$' f"{well} sweep {sweep}")
            axs[2].plot(times*1e-3, Vm, label=f"{well} sweep {sweep}", color=colour)

        axs[0].set_ylabel('Normalised subtracted current')
        axs[1].set_ylabel(r'$I_\mathrm{Kr}$ normalised')
        axs[2].set_ylabel(r'$V_m$')

        axs[2].set_xlabel(r'$t$ (s)')
        fig.savefig(os.path.join(output_dir, f"{protocol}-overlaid-normalised"))


def estimate_noise_and_conductance(well, protocol, sweep, gleak, Eleak, Rseries, Cm, E_rev,
                                   use_V_off=True):
    traces_dir = os.path.join(args.postprocess_data_dir, 'traces')
    # get data
    # TODO Add .csv to the end of these filenames
    before_filename = f"{args.experiment_name}-{protocol}-{well}-before-sweep{sweep}.csv"
    before_trace = pd.read_csv(os.path.join(traces_dir, before_filename)).values.flatten()

    after_filename = f"{args.experiment_name}-{protocol}-{well}-after-sweep{sweep}.csv"
    after_trace = pd.read_csv(os.path.join(traces_dir, after_filename)).values.flatten()

    times_filename = f"{args.experiment_name}-{protocol}-times.csv"
    times = pd.read_csv(os.path.join(traces_dir, times_filename), header=None).values.flatten()

    protocol_dir = os.path.join(args.postprocess_data_dir, 'traces',
                                'protocols')

    prot_func, desc = get_ramp_protocol_from_json(protocol, protocol_dir,
                                                  args.experiment_name)

    # Assume first step is leak ramp
    first_step = [line for line in desc if line[2] != line[3]][0]
    ramp_start, ramp_end = first_step[:2]

    protocol_voltages = np.array([prot_func(t) for t in times])
    dt = times[1] - times[0]

    model_class = 'model3'

    markov_model_leak = ArtefactModel(make_model_of_class(model_class,
                                                          protocol_description=desc,
                                                          times=times))

    gleak, Eleak = fit_leak_parameters_with_artefact(markov_model_leak, desc,
                                                     times, before_trace,
                                                     protocol_voltages)

    noise = before_trace[:200].std()

    Erev = infer_reversal_potential(desc, before_trace,
                                    times, plot=False,
                                    )

    c_model = make_model_of_class(args.model,
                                  voltage=prot_func, times=times, E_rev=Erev,
                                  default_parameters=parameters,
                                  protocol_description=desc)

    model = ArtefactModel(c_model, R_series=Rseries, C_m=Cm, g_leak=gleak,
                          E_leak=Eleak)
    default_parameters = model.get_default_parameters().flatten()

    solver = model.make_forward_solver_current(njitted=False,
                                               return_var='I_out')

    assert np.all(np.isfinite(solver()))

    indices = np.array([i for i in range(len(times))])
    spike_times, spike_indices = \
    markovmodels.voltage_protocols.detect_spikes(times, protocol_voltages)
    _, _, indices = markovmodels.voltage_protocols.remove_spikes(times,
                                                                 protocol_voltages,
                                                                 spike_times,
                                                                 time_to_remove=args.removal_duration)

    gkr_index = c_model.GKr_index

    voltages = np.array([model.voltage(t) for t in model.times])
    leak_ramp_i = [i for i, l in enumerate(desc) if l[2] != l[3]][0]
    ramp_start = desc[leak_ramp_i, 0]
    ramp_end = desc[leak_ramp_i, 1]

    istart = np.argmax(times > ramp_start)
    iend = np.argmax(times > ramp_end)

    dt = times[1] - times[0]
    pp_gleak, pp_Eleak, _, _, _, _, _ = fit_leak_lr(
        voltages, before_trace, dt=dt,
        ramp_start=ramp_start,
        ramp_end=ramp_end
    )

    params_df = pd.DataFrame(c_model.get_default_parameters()[None, :],
                             columns=c_model.get_parameter_labels())
    params_df['well'] = well
    params_df['protocol'] = protocol
    params_df['sweep'] = sweep

    E_rev_df = pd.DataFrame.from_records([
        {
            'well': well,
            'protocol': protocol,
            'sweep': sweep,
            'E_rev': Erev
        }
    ]
                                         )
    params_df = adjust_kinetics(model_class, params_df, E_rev_df, args.reversal)

    param_labels = c_model.get_parameter_labels()
    ideal_params = params_df[param_labels].values.flatten()

    c_solver = c_model.make_hybrid_solver_current(hybrid=False)
    Ileak_ideal = pp_gleak * (voltages - pp_Eleak)
    # Find gkr which best fits the data
    def ideal_opt_g(g):
        p = ideal_params.copy()
        p[-1] = g
        IKr = c_solver(p)
        I_out = IKr + Ileak_ideal
        return np.sum((I_out[indices] - before_trace[indices]) ** 2)

    options = {
    }
    res = scipy.optimize.minimize_scalar(ideal_opt_g, bracket=[0, 10 *
                                                               (before_trace[indices][:5000] /
                                                                (protocol_voltages[indices][:5000] -
                                                                 Erev)).max()], options=options )

    if res.success:
        ideal_gkr = res.x
    else:
        raise ValueError('Failed to find ideal gkr')

    p = c_model.get_default_parameters()

    p[-1] = ideal_gkr
    ideal_current = Ileak_ideal + c_solver(p)

    V_off, success = find_V_off(desc, model.times, before_trace, 'model3',
                                default_parameters, args.reversal,
                                data_label='before')

    if not success:
        raise ValueError('Couldnt infer V_iff')

    p_w_V_off = default_parameters.copy()
    p_w_V_off[-3] = V_off

    gkr_w_V_off = _find_conductance(solver, desc, times, before_trace, indices,
                            voltages, p_w_V_off, args.reversal,
                            gkr_index, model)
    p_w_V_off[c_model.GKr_index] = gkr_w_V_off

    gkr = _find_conductance(solver, desc, times, before_trace, indices,
                            voltages, default_parameters, args.reversal,
                            gkr_index, model)
    p_no_V_off = default_parameters.copy()
    p_no_V_off[c_model.GKr_index] = gkr

    # Minimise SSE to find best conductance
    # Plot stuff
    if not args.no_plot:
        fig = plt.figure(figsize=args.figsize, constrained_layout=True)
        voltage_ax, ax = fig.subplots(2, height_ratios=[0.25, 1])

        p = default_parameters.copy()
        p[-no_artefact_parameters - 1] = gkr
        ax.plot(times*1e-3, before_trace, label='raw pre-drug trace', color='grey', alpha=.5)
        # ax.plot(times*1e-3, solver(p_no_V_off.flatten()), label='Case V')
        ax.plot(times*1e-3, solver(p_w_V_off.flatten()), label='with artefacts')
        ax.plot(times*1e-3, ideal_current, label='without artefacts)')

        handles, labels = plt.gca().get_legend_handles_labels()
        order = [0,2,1]
        plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])

        ax.legend()

        ax.set_xlabel(r'$t$ (s)')
        ax.set_ylabel(r'$I$ (pA)')

        ax.set_ylim(np.quantile(before_trace, [0.01, 0.999]))

        state_solver = model.make_hybrid_solver_states(hybrid=False, njitted=False)
        states = state_solver(p_w_V_off, times=times)
        Vm_V_off = states[:, -1].flatten()
        states = state_solver(p_no_V_off, times=times)
        Vm_no_V_off = states[:, -1].flatten()

        voltage_ax.plot(times*1e-3, voltages, color='black')
        # voltage_ax.plot(times*1e-3, Vm_no_V_off)
        voltage_ax.plot(times*1e-3, Vm_V_off)

        ax.set_title('b', loc='left', fontweight='bold')
        voltage_ax.set_title('a', loc='left', fontweight='bold')

        ax.spines[['top', 'right']].set_visible(False)
        voltage_ax.spines[['top', 'right']].set_visible(False)

        if not os.path.exists(os.path.join(output_dir, "conductance_estimation")):
            os.makedirs(os.path.join(output_dir, "conductance_estimation"))

        ax.set_xticklabels([])
        fig.savefig(os.path.join(output_dir, "conductance_estimation", f"{well}-{protocol}-sweep{sweep}"))
        plt.close(fig)

    print(f"noise, gkr: {noise} {gkr}")

    if use_V_off:
        return noise, gkr_w_V_off

    return noise, gkr


def generate_data(protocol, well, Rseries, Cm, gleak, Eleak, noise, gkr, E_obs, Erev, args,
                  output_dir):
    if Erev is None:
        Erev = markovmodels.utilities.calculate_reversal_potential()

    trace_dir = os.path.join(args.postprocess_data_dir,
                             'traces')
    trace, prot = get_data(well, protocol, trace_dir,
                           args.experiment_name, label='before')

    desc = prot.get_all_sections()
    desc = np.vstack((desc, [[desc[-1, 1], np.inf, -80.0, -80.0]]))
    voltage_func = make_voltage_function_from_description(desc)

    traces_dir = os.path.join(args.postprocess_data_dir, 'traces')
    times = np.loadtxt(os.path.join(traces_dir,
                                       f"{args.experiment_name}-{protocol}-times.csv"))

    if not os.path.exists(os.path.join(output_dir, f"{args.experiment_name}-{protocol}-times.csv")):
        np.savetxt(os.path.join(output_dir, f"{args.experiment_name}-{protocol}-times.csv"),
                   times)


    c_model = make_model_of_class(args.model,
                                  voltage=voltage_func, times=times, E_rev=Erev,
                                  protocol_description=desc)

    a_model = ArtefactModel(c_model, C_m=Cm, R_series=Rseries,
                            g_leak=gleak, E_leak=Eleak)

    a_solver_current = a_model.make_hybrid_solver_current(hybrid=False, njitted=False,
                                                        strict=False,
                                                        return_var='I_out')
    a_solver_states = a_model.make_hybrid_solver_states(hybrid=False, njitted=False,
                                                      strict=False)

    aux_func = a_model.define_auxiliary_function(return_var='I_out')

    _parameters = a_model.get_default_parameters()
    _parameters[c_model.GKr_index] = gkr

    default_parameters = a_model.get_default_parameters()

    V_off_model_class = 'model3'

    V_off, success =  find_V_off(desc, times, trace,
                                 V_off_model_class,
                                 _parameters,
                                 Erev, a_solver_states=a_solver_states,
                                 a_solver_current=a_solver_current,
                                 aux_func=aux_func
                                 )

    model = ArtefactModel(c_model, E_leak=Eleak, g_leak=gleak, C_m=Cm,
                          R_series=Rseries, V_off=V_off)

    # Output Iout
    model.auxiliary_function = njit(model.define_auxiliary_function(return_var = 'I_out'))
    solver = model.make_hybrid_solver_current(njitted=False,
                                              hybrid=False)
    I_out = solver()
    mean = I_out

    data = np.random.normal(mean, noise, times.shape)

    # Output data
    out_fname = os.path.join(output_dir, f"{args.experiment_name}-{protocol}-{well}-before-sweep1.csv")
    pd.DataFrame(data.T, columns=('current',)).to_csv(out_fname)

    gkr_index = len(c_model.get_default_parameters()) - 1
    # Assume 0 conductance after drug addition
    _p = model.get_default_parameters()
    _p[gkr_index] = 0.0
    state_solver = model.make_hybrid_solver_states(hybrid=False,
                                                   njitted=True)
    Vm2 = state_solver(_p)[:, -1].flatten()

    V_m = state_solver()[:, -1]

    data_after = np.random.normal(solver(_p), noise, times.shape)
    out_fname = os.path.join(output_dir, f"{args.experiment_name}-{protocol}-{well}-after-sweep1.csv")
    pd.DataFrame(data_after.T, columns=('current',)).to_csv(out_fname)

    if not args.no_plot:
        fig = plt.figure(figsize=(14, 12))
        axs = fig.subplots(3)
        axs[0].plot(times, mean, label='mean')
        axs[0].plot(times, data, label='data', color='grey', alpha=0.5)
        axs[0].legend()
        axs[1].plot(times, V_m, label='Vm')
        axs[1].plot(times, Vm2, label='Vm2')
        axs[1].set_xlabel('time / ms')
        axs[1].set_ylabel('Vm / mV')
        command_V = [model.voltage(t) for t in times]
        axs[1].plot(times, command_V, label='command voltage (mV)')
        axs[1].legend()
        axs[0].set_ylabel('current / pA')
        axs[2].plot(times, np.abs(V_m - command_V), label='error in Vm')
        axs[2].plot(times, np.abs(Vm2 - command_V), label='error in Vm2')
        axs[2].plot(times, np.abs(Vm2 - V_m), label='Vm2 - Vm')
        axs[2].set_yscale('log')
        axs[2].legend()
        fig.savefig(os.path.join(output_dir, f"plot-{protocol}plot-{well}.png"))
        plt.close(fig)

    # return the filename for convinience
    return out_fname

def get_protocol(protocol_name, args):
    trace_dir = os.path.join(args.postprocess_data_dir,
                             'traces')

    well = None
    data, voltage_protocol = utilities.get_data(well, protocol_name,
                                                trace_dir,
                                                args.experiment_name)

    desc = voltage_protocol.get_all_sections()

    return make_voltage_function_from_description(desc), desc


def subtract_leak(well, protocol, args, output_dir=None):
    nsweeps = 1
    sweep2_fname = f"{args.experiment_name}-{protocol}-{well}-before-sweep2.csv"
    if os.path.exists(os.path.join(args.data_directory, sweep2_fname)):
        nsweeps = 2

    # if not args.no_plot:
    #     protocol_axs, before_axs, after_axs, corrected_axs, subtracted_ax, \
    #         long_protocol_ax = setup_subtraction_grid(fig, nsweeps)

    protocol_dir = os.path.join(args.postprocess_data_dir, 'traces',
                                'protocols')
    protocol_func, desc = get_ramp_protocol_from_json(protocol, protocol_dir,
                                                      experiment_name=args.experiment_name)

    # TODO
    # Find ramp start and end from desc
    leak_ramp = [line for line in desc if line[2] != line[3]][0]
    ramp_start = leak_ramp[0]
    ramp_end = leak_ramp[1]

    observation_times = np.loadtxt(os.path.join(
        args.data_directory, f"{args.experiment_name}-{protocol}-times.csv"))
    protocol_voltages = np.array([protocol_func(t) for t in observation_times])
    dt = observation_times[1] - observation_times[0]

    df = []
    for sweep in range(1, nsweeps + 1):
        before_filename = f"{args.experiment_name}-{protocol}-{well}-before-sweep{sweep}.csv"
        after_filename = f"{args.experiment_name}-{protocol}-{well}-after-sweep{sweep}.csv"

        indices_to_plot = [i for i, t in enumerate(observation_times) if t
                           <= ramp_end * 2]

        tracename = 'subtracted'

        try:
            before_trace_df = pd.read_csv(os.path.join(args.data_directory, before_filename))
            before_trace = before_trace_df[before_trace_df.columns[-1]].to_numpy().flatten().astype(np.float64)
        except FileNotFoundError as exc:
            before_trace = None
            print(str(exc))

        try:
            after_trace_df = pd.read_csv(os.path.join(args.data_directory, after_filename))
            after_trace = after_trace_df[after_trace_df.columns[-1]].to_numpy().flatten().astype(np.float64)
        except FileNotFoundError as exc:
            after_trace = None
            print(str(exc))

        if before_trace is not None and np.all(np.isfinite(before_trace)):
            g_leak_before, E_leak_before, _, _, _, x, y = fit_leak_lr(
                protocol_voltages, before_trace, dt=dt,
                ramp_start=ramp_start,
                ramp_end=ramp_end
            )

            n = len(x)
            # msres = (((x - E_leak_before) * g_leak_before - y)**2 / (n - 2)).sum()

            infer_reversal_potential(desc,
                                     before_trace - g_leak_before * (protocol_voltages - E_leak_before),
                                     observation_times, plot=True,
                                     # output_path=os.path.join(reversal_plot_dir,
                                     #                          f"{well}_{protocol}_sweep{sweep}_before"),
                                     known_Erev=args.Erev
                                     )
        else:
            g_leak_before = np.nan
            E_leak_before = np.nan

        if after_trace is not None and np.all(np.isfinite(after_trace)):
            g_leak_after, E_leak_after, _, _, _, x, y = fit_leak_lr(
                protocol_voltages, after_trace, dt=dt,
                ramp_start=ramp_start,
                ramp_end=ramp_end
            )
            n = len(x)
            # msres = (((x - E_leak_before) * g_leak_before - y)**2 / (n - 2)).sum()

            infer_reversal_potential(desc, before_trace,
                                     observation_times, plot=True,
                                     # output_path=os.path.join(reversal_plot_dir,
                                     #                          f"{well}_{protocol}_sweep{sweep}_after"))
                                     )
        else:
            g_leak_after = np.nan
            E_leak_after = np.nan

        if before_trace is not None:
            before_corrected = before_trace - (g_leak_before * (protocol_voltages - E_leak_before))
            infer_reversal_potential(desc, before_corrected,
                                     observation_times,
                                     # output_path=os.path.join(reversal_plot_dir,
                                     # f"{protocol}_{well}_before_drug_leak_corrected"),
                                     plot=not args.no_plot)

        if after_trace is not None:
            if not args.dont_correct_post:
                after_corrected = after_trace - (g_leak_after * (protocol_voltages - E_leak_after))
                infer_reversal_potential(desc, after_corrected,
                                     observation_times,
                                     # output_path=os.path.join(reversal_plot_dir,
                                     #                          f"{protocol}_{well}_after_drug_leak_corrected"),
                                     plot=not args.no_plot)

            else:
                after_corrected = np.full(after_trace.shape, 0)

        if before_trace is not None and after_trace is not None:
            subtracted_trace = before_corrected - after_corrected
        else:
            subtracted_trace = np.array([np.nan])

        if np.all(np.isfinite(subtracted_trace)):
            fitted_E_rev = infer_reversal_potential(protocol,
                                                    subtracted_trace,
                                                    observation_times,
                                                    known_Erev=args.Erev,
                                                    # output_path=os.path.join(reversal_plot_dir,
                                                    #                          f"{protocol}_{well}_subtracted"),
                                                    plot=not args.no_plot)

        else:
            fitted_E_rev = np.nan

        passed1 = False

        if before_trace is not None and after_trace is not None:
            subtracted_trace_df = pd.DataFrame(np.column_stack(
                (observation_times, subtracted_trace)), columns=('time', 'current'))

            fname = f"{args.experiment_name}-{protocol}-{well}-sweep{sweep}.csv"
            subtracted_trace_df.to_csv(os.path.join(full_subtracted_trace_dir, fname))

            subtracted_trace_df['time'].to_csv(os.path.join(
                full_subtracted_trace_dir, f"{args.experiment_name}-{protocol}-times.csv"))

            # Check that the current isn't negative on the first step after the leak ramp
            first_step = [(i, v) for i, v in enumerate(protocol_voltages) if v > 30]
            lst = []
            for i, (j, voltage) in enumerate(first_step):
                if j - i > first_step[0][0]:
                    # Moved past the first step
                    break
                lst.append(j)
                # Ignore first few timesteps
            first_step_indices = lst[10:-10]

            ax_col = sweep - 1

            tracename, trace = ('subtracted', subtracted_trace)
            estimated_noise = trace[0:200].std()
            trace = trace[first_step_indices]
            n = len(trace)
            if trace.mean() > -2*estimated_noise:
                print(f"{protocol} {well} {tracename} \tpassed QC6")
                passed1 = True
            else:
                print(f"{protocol}, {well}, {tracename} \tfailed QC6")
                passed1 = False

        # Can we infer reversal potential from subtracted trace
        Erev = infer_reversal_potential(protocol, subtracted_trace,
                                        observation_times,
                                        plot=False)

        if Erev > -50 or Erev < -120:
            print(f"{protocol}, {well} \tfailed QC.Erev")
            passed_Erev = False
        else:
            print(f"{protocol}, {well} \tpassed QC.Erev")
            passed_Erev = True

        if after_trace is not None:
            R_leftover = np.sqrt(np.sum(after_corrected**2)/(np.sum(before_corrected**2)))
        else:
            R_leftover = np.nan

        df.append((protocol, well, sweep, tracename, fitted_E_rev,
                   passed1, passed_Erev, R_leftover,
                   g_leak_before, g_leak_after, E_leak_before,
                   E_leak_after))

    df = pd.DataFrame(df, columns=('protocol', 'well', 'sweep', 'before/after',
                                   'fitted_E_rev', 'passed QC6',
                                   'passed QC.Erev', 'R_leftover', 'pre-drug'
                                   ' leak conductance', 'post-drug leak'
                                   ' conductance', 'pre-drug leak reversal',
                                   'post-drug leak reversal'))

    return df


if __name__ == "__main__":
    main()
