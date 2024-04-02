# Example command: python3 scripts/thesis/chapter4/optimisation_results.py ~/data/25112022_MW_FF_processed/traces 0a ~/data/sydney_fitting/25112022MW/Case0a/model3/combine_fitting_results/combined_fitting_results.csv model3 -w B09 --experiment_name 25112022_MW --sweep 1 --output tmp 


import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from matplotlib import rc

import markovmodels
from markovmodels.model_generation import make_model_of_class
from markovmodels.fitting import get_best_params
from markovmodels.ArtefactModel import ArtefactModel
from markovmodels.utilities import setup_output_directory, get_data, get_all_wells_in_directory
from markovmodels.voltage_protocols import get_protocol_list, get_ramp_protocol_from_json, make_voltage_function_from_description
from markovmodels.voltage_protocols import remove_spikes, detect_spikes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


mpl.rcParams['axes.formatter.useoffset'] = False


rc('font', **{'size': 12})
# rc('text', usetex=True)
# rc('figure', dpi=400, facecolor=[0]*4)
# rc('axes', facecolor=[0]*4)
rc('savefig', facecolor=[0]*4)
rc('figure', autolayout=True)

_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', help='directory where data is stored')
    parser.add_argument('fitting_case', type=str)
    parser.add_argument('fitting_results', type=str)
    parser.add_argument('model_class')
    parser.add_argument('--E_rev', type=float, default=-91.71)
    parser.add_argument('--default_parameters_file')
    parser.add_argument('--infer_reversal_potential', action='store_true')
    parser.add_argument('--removal_duration', type=float, default=5.0)
    parser.add_argument('--experiment_name', '-e', default='newtonrun4')
    parser.add_argument('--protocols', type=str, nargs='+')
    parser.add_argument('-w', '--wells', type=str, nargs='+')
    parser.add_argument('-s', '--sweeps', type=int, nargs='+')
    parser.add_argument('--figsize', '-f', nargs=2, type=float, default=[5.7, 9])
    parser.add_argument('--fig_title', '-t', default='')
    parser.add_argument('--nolegend', action='store_true')
    parser.add_argument('--dpi', '-d', default=500, type=int)
    parser.add_argument('--fontsize', type=int, default=12)
    parser.add_argument('--show_uncertainty', action='store_true')
    parser.add_argument('--shared_plot_limits', action='store_true')
    parser.add_argument('--no_voltage', action='store_true')
    parser.add_argument('--file_format', default='')
    parser.add_argument('--reversal', default=-91.71, type=float)
    parser.add_argument('--output')

    global args
    args = parser.parse_args()

    output_dir = setup_output_directory(args.output, 'chapter_4_optimisation_results')

    params_df = pd.read_csv(args.fitting_results)

    params_df.sweep = [max(0, sweep) for sweep in params_df.sweep]

    # Case describing how was the was model fitted
    if args.fitting_case == '0a':
        args.adjust_kinetics = False
        args.infer_reversal_potential = False
        args.use_artefact_model = False
    elif args.fitting_case == '0b':
        args.adjust_kinetics = False
        args.infer_reversal_potential = True
        args.use_artefact_model = False
    elif args.fitting_case == '0c':
        args.adjust_kinetics = True
        args.infer_reversal_potential = True
        args.use_artefact_model = False
    elif args.fitting_case == 'I':
        args.adjust_kinetics = False
        args.infer_reversal_potential = False
        args.use_artefact_model = True
    elif args.case == 'II':
        args.adjust_kinetics = False
        args.infer_reversal_potential = False
        args.use_artefact_model = True

    if args.fontsize:
        matplotlib.rcParams.update({'font.size': args.fontsize})

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)

    infer_reversal_params = np.loadtxt(os.path.join('data', 'BeattieModel_roomtemp_staircase_params.csv')).flatten().astype(np.float64)

    param_labels = make_model_of_class(args.model_class).get_parameter_labels()

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    axs = setup_grid(fig)

    for ax in axs:
        spines = ['top', 'right']
        ax.spines[spines].set_visible(False)

    current_ax, protocol_ax, rank_ax, scatter_ax, baseline_profile_ax = axs

    if not args.wells:
        args.wells = list(params_df.well.unique())
    if not args.protocols:
        args.protocols = list(params_df.protocol.unique())
    if not args.sweeps:
        args.sweeps = list(params_df.sweep.unique())

    # Do all of our plots
    for well in args.wells:
        for protocol in args.protocols:
            for sweep in args.sweeps:
                do_scatter_plot(scatter_ax, params_df, well, protocol,
                                sweep)
                fig.savefig(os.path.join(output_dir, f"{well}_{protocol}_sweep{sweep}"))
                do_profile_plots(baseline_profile_ax, params_df, protocol, well, sweep)
                fig.savefig(os.path.join(output_dir, f"{well}_{protocol}_sweep{sweep}"))
                do_trace_plots(current_ax, protocol_ax, protocol, well, sweep, params_df)
                fig.savefig(os.path.join(output_dir, f"{well}_{protocol}_sweep{sweep}"))
                do_rank_plot(rank_ax, params_df, protocol, well, sweep)
                fig.savefig(os.path.join(output_dir, f"{well}_{protocol}_sweep{sweep}"))
                for ax in axs:
                    ax.cla()

def do_rank_plot(rank_ax, params_df, protocol, well, sweep):
    params_df = params_df[(params_df.well == well)
                          & (params_df.protocol == protocol)
                          & (params_df.sweep == sweep)].copy()

    params_df.score = params_df.score.astype(np.float64)

    params_df = params_df[np.isfinite(params_df.score.values)]

    scores = list(sorted(list(params_df.score.unique().flatten().astype(np.float64))))

    scores = np.array(scores)
    ranks = np.array(list(range(len(scores))))

    times_fname = os.path.join(args.data_dir,
                               f"{args.experiment_name}-{protocol}-times.csv")
    times = np.loadtxt(times_fname).flatten()

    trace, vp = get_data(well, protocol, args.data_dir,
                         args.experiment_name, sweep=sweep)
    desc = vp.get_all_sections()

    prot_func = make_voltage_function_from_description(desc)
    voltages = np.array([prot_func(t) for t in times])

    spike_times, _ = detect_spikes(times, voltages, window_size=0)
    _, _, indices = remove_spikes(times, voltages, spike_times,
                                  args.removal_duration)

    # Convert score to RMSE
    scores = np.sqrt(scores/len(indices))

    # Highlight 25% best results
    lq = np.quantile(scores, .25)
    highlight_indices = np.argwhere((scores <= lq) & (scores != scores.min()))

    other_indices = np.argwhere(scores > lq)

    rank_ax.scatter(np.array(ranks)[other_indices],
                    np.array(scores)[other_indices],
                    color=_colours[1], marker='+')

    rank_ax.scatter(np.array(ranks)[highlight_indices],
                    np.array(scores)[highlight_indices], color=_colours[0], marker='x')

    rank_ax.scatter([0], [scores.min()], color='gold', marker='s')

    rank_ax.set_yscale('log')
    rank_ax.set_ylabel('RMSE (pA)')
    rank_ax.set_xlabel('Rank')


def do_trace_plots(current_ax, protocol_ax, protocol, well, sweep, params_df):
    # Load trace
    data_fname = os.path.join(args.data_dir,
                              f"{args.experiment_name}-{protocol}-{well}-sweep{sweep}-subtracted.csv")
    times_fname = os.path.join(args.data_dir,
                               f"{args.experiment_name}-{protocol}-times.csv")
    times = np.loadtxt(times_fname).flatten()
    trace = np.loadtxt(data_fname).flatten()

    current, vp = get_data(well, protocol, args.data_dir,
                                args.experiment_name, sweep=sweep)
    desc = vp.get_all_sections()
    desc = np.vstack((desc, [[desc[-1, 1], np.inf, -80.0, -80.0]]))
    prot_func = make_voltage_function_from_description(desc)

    protocol_ax.plot(times*1e-3, [prot_func(t) for t in times])
    current_ax.plot(times*1e-3, trace, color='grey', alpha=.5)

    best_params = get_best_params(params_df)

    row = best_params.set_index(['well', 'protocol', 'sweep']).loc[(well, protocol, sweep)].copy()

    param_labels = make_model_of_class(args.model_class).get_parameter_labels()
    params = row[param_labels].values.flatten().astype(np.float64)

    if args.adjust_kinetics:
        params = adjust_parameters(params)

    if not args.infer_reversal_potential:
        E_rev = args.E_rev
    elif not use_artefact_model:
        E_rev = infer_reversal_potential(desc, trace, times,
                                         voltages=voltages)
    else:
        assert(False)


    m_model = make_model_of_class(args.model_class, voltage=prot_func, protocol_description=desc,
                                  times=times,
                                  E_rev=E_rev)

    if args.use_artefact_model:
        m_model = ArtefactModel(m_model)

    pred = m_model.make_hybrid_solver_current(hybrid=False, strict=False)(params)

    if not np.all(np.isfinite(pred)):
        assert(False)

    current_ax.plot(times*1e-3, pred)
    current_ax.set_xlabel('')
    current_ax.set_ylabel(r'$I_\text{subtracted}$ (pA)')
    protocol_ax.set_ylabel(r'$V_\text{m}$ (mV)')
    protocol_ax.set_xlabel('$t$ (ms)')


def do_scatter_plot(scatter_ax, params_df, well, protocol, sweep):
    param_labels = make_model_of_class(args.model_class).get_parameter_labels()
    params_df = params_df[(params_df.well == well)
                          & (params_df.protocol == protocol)
                          & (params_df.sweep == sweep)].copy()
    params_df = params_df[np.isfinite(params_df.score.values)]

    if len(params_df.index == 0):
        return

    best_params = get_best_params(params_df)

    params_df = params_df[params_df.score != params_df.score.min()]

    row = best_params.set_index(['well', 'protocol', 'sweep']).loc[(well, protocol, sweep)].copy()
    param_labels = make_model_of_class(args.model_class).get_parameter_labels()
    best_params = row[param_labels].values


    scores = params_df.score.values
    lq = np.quantile(scores, .25)
    highlight_indices = np.argwhere((scores <= lq) & (scores != scores.min()))

    other_indices = np.argwhere(scores > lq)

    scatter_ax.scatter(params_df[param_labels[0]].values[other_indices],
                       params_df[param_labels[1]].values[other_indices],
                       color=_colours[1], marker='+')

    scatter_ax.scatter(params_df[param_labels[0]].values[highlight_indices],
                       params_df[param_labels[1]].values[highlight_indices],
                       color=_colours[0], marker='x')

    scatter_ax.scatter([best_params[0]], [best_params[1]], color='gold', marker='s')

    inset_ax = inset_axes(scatter_ax,
                          width="35%", # width = 30% of parent_bbox
                          height="25%", # height : 1 inch
    )

    xlims = inset_ax.get_xlim()
    xlims = [xlims[0] - (xlims[1] - xlims[0]) * 0.2,
             xlims[1] + (xlims[1] - xlims[0]) * 0.2]

    ylims = inset_ax.get_ylim()
    ylims = [ylims[0] - (ylims[1] - ylims[0]) * 0.2,
             ylims[1] + (ylims[1] - ylims[0]) * 0.2]

    # inset_ax.set_xlim(xlims)
    # inset_ax.set_ylim(ylims)

    scatter_ax.set_yscale('log')
    scatter_ax.set_xscale('log')

    scatter_ax.set_xlabel(r'$p_1$')
    scatter_ax.set_xlabel(r'$p_2$')

    inset_ax.set_yscale('log')
    inset_ax.set_xscale('log')

    lq = np.quantile(scores, .25)
    highlight_indices = np.argwhere(scores < lq)

    inset_ax.scatter(params_df[param_labels[0]].values[highlight_indices],
                     params_df[param_labels[1]].values[highlight_indices],
                     color=_colours[0], marker='x')

    mark_inset(scatter_ax, inset_ax, 2, 4)

    inset_ax.scatter([best_params[0]], [best_params[1]], color='gold', marker='s')
    inset_ax.set_xticks([], minor=True)
    inset_ax.set_yticks([], minor=True)



def do_profile_plots(baseline_profile_ax, params_df, protocol, well, sweep):

    times_fname = os.path.join(args.data_dir,
                               f"{args.experiment_name}-{protocol}-times.csv")
    trace, vp = get_data(well, protocol, args.data_dir,
                         args.experiment_name, sweep=sweep)

    trace = trace.flatten()

    desc = vp.get_all_sections()
    # Temporary solver hack
    desc = np.vstack((desc, [[desc[-1, 1], np.inf, -80.0, -80.0]]))

    prot_func = make_voltage_function_from_description(desc)

    times = np.loadtxt(times_fname).flatten().astype(np.float64)
    voltages = np.array([prot_func(t) for t in times])

    assert(np.all(np.isfinite(voltages)))

    spike_times, _ = detect_spikes(times, voltages, window_size=0)
    _, _, indices = remove_spikes(times, voltages, spike_times,
                                  args.removal_duration)

    best_params = get_best_params(params_df)

    row = best_params.set_index(['well', 'protocol', 'sweep']).loc[(well, protocol, sweep)]

    param_labels = make_model_of_class(args.model_class).get_parameter_labels()
    params = row[param_labels].values.flatten()

    if args.adjust_kinetics:
        # TODO Fix
        params = adjust_parameters(args.model_class,
                                   params_df, E_rev_df,
                                   args.E_rev)

    if not args.infer_reversal_potential:
        E_rev = args.E_rev
    elif not use_artefact_model:
        E_rev = infer_reversal_potential(desc, trace, times,
                                         voltages=voltages)
    else:
        assert(False)


    if args.default_parameters_file:
        default_parameters = np.loadtxt(os.path.join(args.default_parameters_file))
        m_model = make_model_of_class(args.model_class, voltage=prot_func,
                                      protocol_description=desc,
                                      default_parameters=default_parameters,
                                      times=times, E_rev=E_rev)
    else:
        m_model = make_model_of_class(args.model_class, voltage=prot_func,
                                      protocol_description=desc,
                                      times=times, E_rev=E_rev)

    if args.use_artefact_model:
        m_model = ArtefactModel(m_model)

    default_params = m_model.get_default_parameters()
    solver = m_model.make_hybrid_solver_current(hybrid=False)

    @njit
    def compute_rmse(p):
        if np.any(p <= 0):
            return np.nan
        return np.sqrt(np.mean((solver(p)[indices] - trace[indices])**2))

    plot_var = np.linspace(-0.05, 1.05, 250)

    params = [params + (default_params - params) * l for l in plot_var]
    scores = [compute_rmse(p.flatten()) for p in params]

    baseline_profile_ax.plot(plot_var, scores)

    baseline_profile_ax.set_ylabel('RMSE (pA)')

    baseline_profile_ax.axhline(min(*scores), color='grey', linestyle='--')
    baseline_profile_ax.axvline(0, color='grey')
    baseline_profile_ax.axvline(1.0, color='grey')


def setup_grid(fig):
    no_columns = 2
    no_rows = 4
    gs = GridSpec(no_rows, no_columns, figure=fig, height_ratios=[.5, .5, 1, 1])

    current_ax = fig.add_subplot(gs[0, :])
    protocol_ax = fig.add_subplot(gs[1, :])

    baseline_profile_ax = fig.add_subplot(gs[2, :])

    rank_ax = fig.add_subplot(gs[3, 0])
    scatter_ax = fig.add_subplot(gs[3, 1])


    axs = current_ax, protocol_ax, rank_ax, scatter_ax, baseline_profile_ax

    return axs


if __name__ == "__main__":
    main()
