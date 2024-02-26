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
import seaborn as sns
from numba import njit
from quality_control.leak_fit import fit_leak_lr
from markovmodels.fitting import infer_reversal_potential, infer_reversal_potential_with_artefact

import matplotlib
matplotlib.use('Agg')

# params_for_Erev = np.loadtxt(os.path.join('data', 'Beattie_Sinusoidal_params.csv'),
#                              delimiter=', ').flatten().astype(np.float64)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("qc_estimates_file")
    parser.add_argument("subtraction_results_file")
    parser.add_argument("selection_file")
    parser.add_argument('traces_directory')
    parser.add_argument('subtracted_traces_directory')
    parser.add_argument('--ramp_start', type=float, default=300)
    parser.add_argument('--ramp_end', type=float, default=900)
    parser.add_argument("--experiment_name")
    parser.add_argument("--parameters", default=None)
    parser.add_argument("-w", "--wells", nargs='+')
    parser.add_argument("-s", "--sweeps", nargs='+')
    parser.add_argument("--model", default='model3')
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--no_plot", action='store_true')
    parser.add_argument('-P', '--protocols', nargs='+', default=['staircaseramp'])
    parser.add_argument('--noise', default=0.00, type=float)
    parser.add_argument('--reversal', '-e', type=float)
    parser.add_argument('--cpus', '-c', default=1, type=int)
    parser.add_argument('--use_hybrid_solver', action='store_true')
    parser.add_argument('--sampling_frequency', default=0.1, type=float)
    parser.add_argument('--figsize', type=int, nargs=2, default=[8, 12])
    parser.add_argument('--no_noise', action='store_true')
    parser.add_argument('--dont_correct_post', action='store_true')

    global args
    args = parser.parse_args()

    if not args.wells:
        args.wells = []

    Erev = markovmodels.utilities.calculate_reversal_potential()\
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
                                                            'protocol', 'sweep', 'well',
                                                            'fitted_E_rev']]

    leak_df = leak_df.set_index(['protocol', 'well', 'sweep']).sort_index()
    qc_df = qc_df.set_index(['protocol', 'well', 'sweep']).sort_index()

    tasks = []
    for (index, leak_row) in leak_df.iterrows():
        protocol, well, sweep = index
        if well not in args.wells and args.wells:
            continue

        if well not in passed_wells:
            continue

        if str(sweep) not in args.sweeps:
            print(sweep)
            continue

        gleak = leak_row['pre-drug leak conductance']
        Eleak = leak_row['pre-drug leak reversal']
        qc_row = qc_df.loc[protocol, well, sweep][['Rseries', 'Cm']]
        Rseries = qc_row['Rseries'].values[0] * 1e-9
        Cm = qc_row['Cm'].values[0] * 1e9
        E_obs = leak_row['fitted_E_rev']
        noise, gkr = estimate_noise_and_conductance(well, protocol, sweep,
                                                    gleak, Eleak, Rseries, Cm, E_obs)

        if args.no_noise:
            noise = 0
        tasks.append((protocol, well, Rseries, Cm, gleak, Eleak, noise, gkr, E_obs, Erev))

    print(f"tasks are {tasks}")
    with multiprocessing.Pool(args.cpus) as pool:
        res = pool.starmap(generate_data, tasks)

    dfs = []
    for fname, task in zip(res, tasks):
        protocol, well, Rseries, Cm, gleak, Eleak, noise, gkr, E_obs, Erev = task
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
        if sweep not in df:
            df['sweep'] = 1
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(output_dir, 'subtract_leak_df.csv'))

    plot_overlaid_traces(df)
    do_scatterplots(df, leak_df)

    compare_synth_real_postprocessed_data(df, leak_df)


def compare_synth_real_postprocessed_data(df, leak_df):
    fig = plt.figure(figsize=args.figsize)
    ax = fig.subplots()

    leak_df = leak_df.reset_index()

    plot_dir = os.path.join(output_dir, "compare_real_synth_traces")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    labels = []
    for protocol in df.protocol.unique():
        times_df = pd.read_csv(os.path.join(output_dir, 'subtracted_traces',
                                            f"{args.experiment_name}-{protocol}-times.csv"))
        times = times_df['time'].to_numpy().flatten()

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
                                                           f"{args.experiment_name}-{protocol}-{well}-sweep{sweep}.csv"))['current'].to_numpy()
                real_sub_trace = pd.read_csv(os.path.join(args.subtracted_traces_directory,
                                                          f"{args.experiment_name}-{protocol}-{well}-sweep{sweep}.csv"))['current'].to_numpy()

                # Plot both traces normalised
                ax.plot(times, synth_sub_trace/synth_sub_trace.std(), alpha=.5, label='synth data')

                times_filename = f"{args.experiment_name}-staircaseramp1-times.csv"
                real_times = pd.read_csv(os.path.join(args.subtracted_traces_directory, times_filename))['time'].to_numpy().flatten()
                ax.plot(real_times, real_sub_trace/real_sub_trace.std(), alpha=.5, label='real data')

                ax.set_ylabel('normalised post-processed current')
                ax.set_xlabel('times (ms)')
                ax.legend()

                fig.savefig(os.path.join(plot_dir, f"{protocol}-{well}-sweep{sweep}-postprocessed"))
                ax.cla()
                label = f"{protocol}-{well}-{sweep}"
                labels.append(label)
                all_synth_traces.append(synth_sub_trace.flatten())
                all_real_traces.append(real_sub_trace.flatten())

        deflection_plot_dir = os.path.join(output_dir, 'deflection_plots')
        if not os.path.exists(deflection_plot_dir):
            os.makedirs(deflection_plot_dir)

        print(all_real_traces)
        all_real_traces = np.vstack(all_real_traces)
        all_synth_traces = np.vstack(all_synth_traces)

        print(all_real_traces.shape)

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
    fig = plt.figure(figsize=args.figsize)
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
    fig = plt.figure(figsize=args.figsize)
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
                                      protocol_description=desc)

        model = ArtefactModel(c_model, E_leak=Eleak, g_leak=gleak, C_m=Cm, R_series=Rseries)

        solver = model.make_hybrid_solver_states(hybrid=False)
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

        print('voltage with gkr=0', Vm2)

        E_rev_g_0 = infer_reversal_potential('staircaseramp1', subtracted_trace, times,
                                             output_path=os.path.join(output_dir, 'reversal_plots',
                                                                      f"{protocol}-{well}-sweep{sweep}_Vm_gkr=0"),
                                             known_Erev=Erev,
                                             voltages=Vm2)

        true_IKr = model.define_auxiliary_function()(states.T, model.get_default_parameters(), Vm)

        before_trace = pd.read_csv(os.path.join(output_dir,
                                                f"{args.experiment_name}-{protocol}"
                                                f"-{well}-before-sweep{sweep}.csv"))['current'].to_numpy().flatten()

        before_corrected_Vm = before_trace - gleak * (Vm - Eleak)

        gkr_index = c_model.GKr_index
        p = model.get_default_parameters()
        p[gkr_index] = 0.0
        Vm = solver(p)[:, -1].flatten()
        after_trace = pd.read_csv(os.path.join(output_dir,
                                               f"{args.experiment_name}-{protocol}"
                                               f"-{well}-after-sweep{sweep}.csv"))['current'].to_numpy().flatten()
        after_corrected_Vm = after_trace - gleak_after * (Vm - Eleak_after)

        subtracted_Vm = before_corrected_Vm - after_corrected_Vm

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
                                                    f"{args.experiment_name}-{protocol}-{well}-sweep{sweep}.csv"))['current'].to_numpy()

        times_df = pd.read_csv(os.path.join(output_dir, 'subtracted_traces',
                                            f"{args.experiment_name}-{protocol}-times.csv"))
        times = times_df['time'].to_numpy().flatten()

        _parameters = parameters.copy()
        _parameters[-1] = gkr

        prot_func, _, desc = markovmodels.voltage_protocols.get_ramp_protocol_from_csv(protocol)

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

            state_solver = model.make_hybrid_solver_states(hybrid=False)
            states = state_solver()

            Vm = states[:, -1].flatten()

            colour = seaborn_palette[index]

            subtracted_trace = pd.read_csv(os.path.join(output_dir, "subtracted_traces",
                                                        f"{args.experiment_name}-{protocol}-{well}-sweep{sweep}.csv"))['current'].to_numpy()

            I_Kr = c_model.define_auxiliary_function()(states[:, :-1].T, _parameters, Vm)
            I_Kr = I_Kr / I_Kr.std()

            axs[0].plot(times, subtracted_trace / subtracted_trace.std(),
                        label=f"{well} sweep {sweep}", color=colour, alpha=.25)
            axs[1].plot(times, I_Kr, label=r'$I_\mathrm{Kr}$' f"{well} sweep {sweep}")
            axs[2].plot(times, Vm, label=f"{well} sweep {sweep}", color=colour)

        axs[0].set_ylabel('Normalised subtracted current')
        axs[1].set_ylabel(r'$I_\mathrm{Kr}$ normalised')
        axs[2].set_ylabel(r'$V_m$')

        axs[2].set_xlabel(r'$t$ (ms)')
        fig.savefig(os.path.join(output_dir, f"{protocol}-overlaid-normalised"))


def estimate_noise_and_conductance(well, protocol, sweep, gleak, Eleak, Rseries, Cm, E_obs):
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

    c_model = make_model_of_class(args.model,
                                  voltage=prot_func, times=times, E_rev=args.reversal,
                                  default_parameters=parameters,
                                  protocol_description=desc)

    solver = c_model.make_forward_solver_current(njitted=False)

    reference_trace = solver() / c_model.get_default_parameters()[-1]

    @njit
    def min_func(g_kr):
        return np.sum((g_kr * reference_trace - subtracted_trace) ** 2)

    # Minimise SSE to find best conductance
    res = scipy.optimize.minimize_scalar(min_func, method='bounded', bounds=[0, 1e5])

    gkr = res.x
    print('gkr is', gkr)

    # Plot stuff
    if not args.no_plot:
        fig = plt.figure(figsize=args.figsize)
        ax = fig.subplots()

        ax.plot(times, subtracted_trace, label='subtracted trace', color='grey')
        p = c_model.get_default_parameters()
        p[-1] = gkr
        ax.plot(times, solver(p), label='ideal current')

    if not os.path.exists(os.path.join(output_dir, "conductance_estimation")):
        os.makedirs(os.path.join(output_dir, "conductance_estimation"))
    fig.savefig(os.path.join(output_dir, "conductance_estimation", f"{well}-{protocol}-sweep{sweep}"))
    plt.close(fig)
    return noise, gkr


def generate_data(protocol, well, Rseries, Cm, gleak, Eleak, noise, gkr, E_obs, Erev=None):
    if Erev is None:
        Erev = markovmodels.utilities.calculate_reversal_potential()

    prot_func, _times, desc = markovmodels.voltage_protocols.get_ramp_protocol_from_csv(protocol)

    times_df = pd.read_csv(os.path.join(args.subtracted_traces_directory,
                                        f"{args.experiment_name}-{protocol}-times.csv"))
    times = times_df['time'].to_numpy().flatten()

    if not os.path.exists(os.path.join(output_dir, f"{args.experiment_name}-{protocol}-times.csv")):
        times_df = pd.DataFrame(times.T*1e-3, columns=('time',))
        times_df.to_csv(os.path.join(output_dir, f"{args.experiment_name}-{protocol}-times.csv"))

    _parameters = parameters.copy()
    _parameters[-1] = gkr

    c_model = make_model_of_class(args.model,
                                  voltage=prot_func, times=times, E_rev=Erev,
                                  default_parameters=_parameters,
                                  protocol_description=desc)
    # V_off = args.reversal - E_obs
    V_off = 0

    model = ArtefactModel(c_model, E_leak=Eleak, g_leak=gleak, C_m=Cm,
                          R_series=Rseries, V_off=V_off)

    # Output Iout
    model.auxiliary_function = njit(model.define_auxiliary_function(return_var = 'I_post'))
    solver = model.make_forward_solver_current(njitted=True)
    I_out = solver()
    # voltages = np.array([prot_func(t) for t in times])
    # I_leak = gleak * (voltages - Eleak)
    mean = I_out

    data = np.random.normal(mean, noise, times.shape)

    # Output data
    out_fname = os.path.join(output_dir, f"{args.experiment_name}-{protocol}-{well}-before-sweep1.csv")
    pd.DataFrame(data.T, columns=('current',)).to_csv(out_fname)

    gkr_index = len(c_model.get_default_parameters()) - 1
    # Assume 0 conductance after drug addition
    _p = model.get_default_parameters()
    _p[gkr_index] = 0.0
    print(_p)
    state_solver = model.make_hybrid_solver_states(hybrid=False,
                                                   njitted=True)
    Vm2 = state_solver(_p)[:, -1].flatten()
    print(Vm2)

    V_m = state_solver()[:, -1]
    print(V_m)

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


if __name__ == "__main__":
    main()
