# Example command: python3 scripts/thesis/chapter4/optimisation_results.py ~/data/25112022_MW_FF_processed/traces 0a ~/data/sydney_fitting/25112022MW/Case0a/model3/combine_fitting_results/combined_fitting_results.csv model3 -w B09 --experiment_name 25112022_MW --sweep 1 --output tmp


import argparse
import logging
import os

import matplotlib
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import FormatStrFormatter

from matplotlib import rc

import markovmodels
from markovmodels.model_generation import make_model_of_class
from markovmodels.fitting import get_best_params, infer_reversal_potential, make_prediction, adjust_kinetics
from markovmodels.ArtefactModel import ArtefactModel, no_artefact_parameters
from markovmodels.utilities import setup_output_directory, get_data, get_all_wells_in_directory
from markovmodels.voltage_protocols import get_protocol_list, make_voltage_function_from_description
from markovmodels.voltage_protocols import remove_spikes, detect_spikes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


mpl.rcParams['axes.formatter.useoffset'] = True

# rc('text', usetex=True)
# rc('figure', dpi=400, facecolor=[0]*4)
# rc('axes', facecolor=[0]*4)
# rc('savefig', facecolor=[0]*4)
rc('figure', autolayout=True)

_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

multiprocessing_kws = {'maxtasksperchild': 1}

relabel_states = {
    'model2': {
        'O': 'O',
        'I': 'I',
        'C': 'C',
    },
    'model3': {
        'O1_I': 'I',
        'O1_O2': 'O',
        'C_I': 'IC',
        'C_O2': 'C',
    },
    'model10': {
        'O_I': 'I',
        'O_O2': 'O',
        'C1_I': 'IC',
        'C1_O2': 'C',
        'C2_I': 'IC',
        'C2_O2': 'C',
    }
}


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', help='directory where data is stored')
    parser.add_argument('fitting_case', type=str)
    parser.add_argument('fitting_results', type=str)
    parser.add_argument('subtraction_df', type=str)
    parser.add_argument('model_class')
    parser.add_argument('--data_label', default='')
    parser.add_argument('--plot_wip', action='store_true')
    parser.add_argument('--no_cpus', '-c', type=int, default=1)
    parser.add_argument('--E_rev', type=float, default=-91.71)
    parser.add_argument('--default_parameters_file')
    parser.add_argument('--infer_reversal_potential', action='store_true')
    parser.add_argument('--removal_duration', type=float, default=5.0)
    parser.add_argument('--experiment_name', '-e', default='newtonrun4')
    parser.add_argument('--protocols', type=str, nargs='+')
    parser.add_argument('-w', '--wells', type=str, nargs='+')
    parser.add_argument('-s', '--sweeps', type=int, nargs='+')
    parser.add_argument('--figsize', '-f', nargs=2, type=float, default=[5.3, 7.5])
    parser.add_argument('--fig_title', '-t', default='')
    parser.add_argument('--nolegend', action='store_true')
    parser.add_argument('--dpi', '-d', default=500, type=int)
    parser.add_argument('--fontsize', type=int, default=8)
    parser.add_argument('--show_uncertainty', action='store_true')
    parser.add_argument('--shared_plot_limits', action='store_true')
    parser.add_argument('--no_voltage', action='store_true')
    parser.add_argument('--file_format', default='')
    parser.add_argument('--reversal', default=-91.71, type=float)
    parser.add_argument('--cutoff_threshold', default=1.01, type=float)
    parser.add_argument('--output')

    global args
    args = parser.parse_args()

    global cutoff_threshold
    cutoff_threshold = args.cutoff_threshold

    output_dir = setup_output_directory(args.output, 'chapter_4_optimisation_results')

    fitting_results_fname = args.fitting_results
    params_df = pd.read_csv(fitting_results_fname)
    params_df.sweep = [max(0, sweep) for sweep in params_df.sweep]

    params_df['protocol'] = ['staircaseramp1_2' if protocol ==
                              'staircaseramp2' else protocol for
                              protocol in params_df.protocol]

    if len(args.fitting_case) > 4:
        if args.fitting_case[:4] == 'Case':
            args.fitting_case = args.fitting_case[4:]

    args.data_label = ''
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
    elif args.fitting_case == '0d':
        args.adjust_kinetics = True
        args.infer_reversal_potential = True
        args.data_label = 'before'
        args.use_artefact_model = False
    elif args.fitting_case == 'I':
        args.adjust_kinetics = False
        args.infer_reversal_potential = False
        args.use_artefact_model = True
        args.data_label = 'before'
    elif args.fitting_case == 'II':
        args.adjust_kinetics = False
        args.infer_reversal_potential = False
        args.use_artefact_model = True
        args.data_label = 'before'

    subtraction_df = pd.read_csv(args.subtraction_df)

    if args.fontsize:
        matplotlib.rcParams.update({'font.size': args.fontsize})

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)

    infer_reversal_params = np.loadtxt(os.path.join('data', 'BeattieModel_roomtemp_staircase_params.csv')).flatten().astype(np.float64)

    param_labels = make_model_of_class(args.model_class).get_parameter_labels()

    if not args.wells:
        args.wells = list(params_df.well.unique())
    if not args.protocols:
        args.protocols = list(params_df.protocol.unique())
    if not args.sweeps:
        args.sweeps = list(params_df.sweep.unique())

    tasks = []
    for well in args.wells:
        for protocol in args.protocols:
            for sweep in args.sweeps:
                tasks.append([well, protocol, sweep, params_df.copy(), args, output_dir])

    with multiprocessing.Pool(min(len(tasks), args.no_cpus), **multiprocessing_kws) as pool:
        pool.starmap(map_func, tasks)

    best_params = get_best_params(params_df)

    best_params.to_csv(os.path.join(output_dir, "best_params_df.csv"))

    opt_results_df = []
    # Iterate over (well, protocol, sweep) combinations
    for (well, protocol, sweep), _ in best_params.set_index(['well', 'protocol', 'sweep']).sort_index().iterrows():

        sub_df = params_df[(params_df.protocol == protocol)
                           & (params_df.well == well)
                           & (params_df.sweep == sweep)]

        times_fname = os.path.join(args.data_dir,
                               f"{args.experiment_name}-{protocol}-times.csv")
        sub_df = sub_df[sub_df.protocol == protocol].copy()
        no_obs = np.loadtxt(times_fname).flatten().shape[0]

        sub_df['RMSE'] = np.sqrt(sub_df.score / no_obs)

        min_score = sub_df['RMSE'].min()

        cutoff = min_score * cutoff_threshold

        success_rate = 2 * float(len(sub_df[sub_df.RMSE < cutoff].index)) / len(sub_df.index)

        this_row = {
            'well': well,
            'protocol': protocol,
            'sweep': sweep,
            'opt_success_rate': success_rate
            }

        for param in param_labels:
            this_row[f"{param}_std"] = sub_df[sub_df.RMSE < cutoff][param].values.std()
            this_row[f"{param}_best"] = sub_df[sub_df.RMSE == sub_df.RMSE.min()][param].values[0]

        opt_results_df.append(this_row)

    opt_results_df = pd.DataFrame.from_records(opt_results_df)
    opt_results_df['model'] = args.model_class
    opt_results_df['case'] = args.fitting_case

    opt_results_df.to_csv(os.path.join(output_dir, 'success_rates.csv'))


def map_func(well, protocol, sweep, params_df, args, output_dir):
    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    axs = setup_grid(fig)
    for ax in axs:
        spines = ['top', 'right']
        ax.spines[spines].set_visible(False)


    if (well, protocol, sweep) not in params_df.set_index(['well', 'protocol', 'sweep']).sort_index().index:
        return


    occupations_ax, current_ax, protocol_ax, rank_ax, scatter_ax, baseline_profile_ax = axs

    title_font_size = args.fontsize

    occupations_ax.set_title('b', fontweight='bold', fontsize=title_font_size,
                         loc='left')
    current_ax.set_title('c', fontweight='bold', fontsize=title_font_size,
                         loc='left')
    protocol_ax.set_title('a', fontweight='bold', fontsize=title_font_size,
                          loc='left')
    scatter_ax.set_title('d', fontweight='bold', fontsize=title_font_size,
                         loc='left')
    rank_ax.set_title('e', fontweight='bold', fontsize=title_font_size,
                      loc='left')
    baseline_profile_ax.set_title('f', fontweight='bold',
                                  fontsize=title_font_size,
                                  loc='left')
    do_rank_plot(rank_ax, params_df, protocol, well, sweep, args)
    if args.plot_wip:
        fig.savefig(os.path.join(output_dir, f"{well}_{protocol}_sweep{sweep}.pdf"))
    do_trace_plots(current_ax, protocol_ax, occupations_ax,
                   protocol, well, sweep, params_df, args)
    if args.plot_wip:
        fig.savefig(os.path.join(output_dir, f"{well}_{protocol}_sweep{sweep}.pdf"))
    do_scatter_plot(scatter_ax, params_df, well, protocol,
                    sweep, args)
    if args.plot_wip:
        fig.savefig(os.path.join(output_dir, f"{well}_{protocol}_sweep{sweep}.pdf"))
    do_profile_plots(baseline_profile_ax, params_df, protocol, well, sweep, args)

    # Make sure ticks and limits match for top 3 subfigures
    occupations_ax.set_xticks(current_ax.get_xticks())
    occupations_ax.set_xlim(current_ax.get_xlim())

    occupations_ax.set_xticklabels([])
    protocol_ax.set_xticklabels([])
    protocol_ax.set_xticks(current_ax.get_xticks())
    protocol_ax.set_xlim(current_ax.get_xlim())

    # Plot everything
    fig.align_ylabels([occupations_ax, current_ax, protocol_ax, rank_ax, scatter_ax])

    fig.savefig(os.path.join(output_dir, f"{well}_{protocol}_sweep{sweep}.pdf"))
    plt.close(fig)


def do_rank_plot(rank_ax, params_df, protocol, well, sweep, args):
    params_df = params_df[(params_df.well == well)
                          & (params_df.protocol == protocol)
                          & (params_df.sweep == sweep)].copy()
    params_df.score = params_df.score.astype(np.float64)
    params_df = params_df[np.isfinite(params_df.score.values)]

    scores = list(sorted(list(params_df.score.unique().flatten().astype(np.float64))))

    # 'RMSE' in the data is actually just MSE due to bug
    # scores = np.sqrt(scores / n_data)
    scores = np.sqrt(scores)

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
    n_data = len(indices)

    ranks = np.array(list(range(len(scores))))

    # Highlight 25% best results
    cutoff = scores.min() * cutoff_threshold
    highlight_indices = np.argwhere((scores <= cutoff) & (scores != scores.min()))

    other_indices = np.argwhere(scores > cutoff)

    rank_ax.scatter(np.array(ranks)[other_indices],
                    np.array(scores)[other_indices],
                    color=_colours[1], marker='+')

    rank_ax.scatter(np.array(ranks)[highlight_indices],
                    np.array(scores)[highlight_indices], color=_colours[0], marker='x')

    rank_ax.scatter([0], [scores.min()], color='gold', marker='s')

    rank_ax.set_yscale('log')
    rank_ax.set_ylabel('RMSE (pA)')
    rank_ax.set_xlabel('Rank')


def do_trace_plots(current_ax, protocol_ax, occupations_ax,
                   protocol, well, sweep, params_df, args):
    # Load trace
    data_fname = os.path.join(args.data_dir,
                              f"{args.experiment_name}-{protocol}-{well}-sweep{sweep}-subtracted.csv")

    subtraction_df = pd.read_csv(args.subtraction_df)

    times_fname = os.path.join(args.data_dir,
                               f"{args.experiment_name}-{protocol}-times.csv")
    times = np.loadtxt(times_fname).flatten()

    trace, vp = get_data(well, protocol, args.data_dir, args.experiment_name,
                           sweep=sweep, label=args.data_label)

    desc = vp.get_all_sections()
    desc = np.vstack((desc, [[desc[-1, 1], np.inf, -80.0, -80.0]]))

    global protocol_dict
    protocol_dict = {protocol: (desc, times)}

    prot_func = make_voltage_function_from_description(vp.get_all_sections())

    voltages = np.array([prot_func(t) for t in times])

    pred, states = make_prediction(args.model_class, args, well, protocol, sweep,
                                   protocol, sweep, params_df, subtraction_df,
                                   args.fitting_case, args.reversal, protocol_dict,
                                   trace, voltages, label=args.data_label,
                                   return_states=True
                                   )

    model = make_model_of_class(args.model_class)
    if args.fitting_case in ['I', 'II']:
        model = ArtefactModel(model)

    states, state_labels = model.compute_all_states(states)

    s_state_labels = sorted(state_labels)
    reorder_indices = [s_state_labels.index(s) for s in state_labels]
    states = states[:, reorder_indices].copy()

    # Hacky way of ensuring that the O state is at the bottom
    open_state_labels = ['O_O2', 'O1_O2', 'O']

    open_state_label = 'O'
    for lab in open_state_labels:
        if lab in state_labels:
            open_state_label = lab
            break

    open_index = state_labels.index(open_state_label)
    state_labels.remove(open_state_label)
    state_labels = [open_state_label] + state_labels

    relabel_states_indices = [open_index] +\
        [i for i in range(len(state_labels)) if i != open_index]

    states = states[:, relabel_states_indices]

    if args.model_class in relabel_states:
        state_labels = [relabel_states[args.model_class][s] for s in state_labels]

    colours = sns.husl_palette(len(state_labels))


    culm_states = np.full(states.shape[0], 0.0)
    for i in range(states.shape[1]):
        colour = colours[i]
        label = state_labels[i]

        occupations_ax.plot(times*1e-3, culm_states + states[:, i].flatten(),
                            color='grey', lw=.3)

        occupations_ax.fill_between(times*1e-3, culm_states,
                                    culm_states + states[:, i].flatten(),
                                    color=colour,
                                    label=label)

        culm_states += states[:, i].flatten()

    occupations_ax.legend(fontsize=8, ncol=states.shape[1], loc='lower center',
                          bbox_to_anchor=[.5, 1.0])

    occupations_ax.set_ylim([0, 1])

    current_ax.plot(times*1e-3, trace, alpha=.5, lw=.5, color='red')
    current_ax.plot(times*1e-3, pred, alpha=.5, lw=.8)
    current_ax.set_xlabel('')

    if args.fitting_case in ['0a', '0b', '0c']:
        current_ax.set_ylabel(r'$I_\mathrm{Kr}$ (pA)')
    else:
        current_ax.set_ylabel(r'$I_\mathrm{obs}$ (pA)')

    occupations_ax.set_ylabel(r'$\mathbf{x}(t)$')
    protocol_ax.plot(times*1e-3, voltages, color='black', label=r'$V$ (mV)')
    if args.fitting_case in ['I', 'II']:
        protocol_ax.set_ylabel(r'$V$ (mV)')
        pred, states = make_prediction(args.model_class, args, well, protocol, sweep,
                                   protocol, sweep, params_df, subtraction_df,
                                   args.fitting_case, args.reversal, protocol_dict,
                                   trace, voltages, label=args.data_label,
                                   return_states=True
                                   )
        Vm = states[:, -1]
        protocol_ax.plot(times*1e-3, Vm, label=r'$V_\mathrm{m}$')
    elif args.fitting_case == '0d':
        E_obs = infer_reversal_potential(desc, trace, times,
                                         voltages=voltages)
        V_off = args.reversal - E_obs
        Vm = voltages + V_off
        protocol_ax.plot(times*1e-3, Vm, label=r'$V_\mathrm{m}$')

    else:
        protocol_ax.set_ylabel(r'$V_\text{cmd}$ (mV)')
    protocol_ax.set_xlabel('$t$ (ms)')


def do_scatter_plot(scatter_ax, params_df, well, protocol, sweep, args):

    subtraction_df = pd.read_csv(args.subtraction_df)
    if args.adjust_kinetics:
        params_df = adjust_kinetics(args.model_class, params_df,
                                    subtraction_df, args.reversal, args.reversal).copy()

    param_labels = make_model_of_class(args.model_class).get_parameter_labels()
    params_df = params_df[(params_df.well == well)
                          & (params_df.protocol == protocol)
                          & (params_df.sweep == sweep)].copy()
    params_df = params_df[np.isfinite(params_df.score.values)]

    if len(params_df.index) == 0:
        return

    best_params = get_best_params(params_df)

    params_df = params_df[params_df.score != params_df.score.min()]

    row = best_params.set_index(['well', 'protocol', 'sweep']).loc[(well, protocol, sweep)].copy()
    param_labels = make_model_of_class(args.model_class).get_parameter_labels()
    best_params = row[param_labels].values


    scores = params_df.score.values

    cutoff = scores.min() * cutoff_threshold
    highlight_indices = np.argwhere((scores <= cutoff) & (scores != scores.min()))

    other_indices = np.argwhere(scores > cutoff)

    scatter_ax.scatter([best_params[0]], [best_params[1]], color='gold', marker='s', alpha=.5)
    scatter_ax.scatter(params_df[param_labels[0]].values[other_indices],
                       params_df[param_labels[1]].values[other_indices],
                       color=_colours[1], marker='+')

    scatter_ax.scatter(params_df[param_labels[0]].values[highlight_indices],
                       params_df[param_labels[1]].values[highlight_indices],
                       color=_colours[0], marker='x')


    scatter_ax.set_yscale('log')
    scatter_ax.set_xscale('log')

    # Limits for inset
    if highlight_indices.flatten().shape[0] > 0:
        xlims = [params_df[param_labels[0]].values[highlight_indices].min(),
                 params_df[param_labels[0]].values[highlight_indices].max()]

        ylims = [params_df[param_labels[1]].values[highlight_indices].min(),
                 params_df[param_labels[1]].values[highlight_indices].max()]


        if xlims[0] != xlims[1] and ylims[0] != ylims[1]:
            inset_ax = inset_axes(scatter_ax,
                                width="40%",
                                height="40%",
            )

            scatter_ax.set_xlabel(r'$p_1$ (ms$^{-1}$)')
            scatter_ax.set_ylabel(r'$p_2$ (mv$^{-1}$)')

            inset_ax.scatter(params_df[param_labels[0]].values[highlight_indices],
                            params_df[param_labels[1]].values[highlight_indices],
                            color=_colours[0], marker='x')

            inset_ax.scatter([best_params[0]], [best_params[1]], color='gold', marker='s')
            # inset_ax.xaxis.set_major_formatter(FormatStrFormatter('%.3E'))
            # inset_ax.yaxis.set_major_formatter(FormatStrFormatter('%.3E'))
            inset_ax.tick_params(axis='x', labelrotation=90)
            inset_ax.tick_params(axis='y')

            # xlims[0] -= 0.05 * (xlims[1] - xlims[0])
            # xlims[1] += 0.05 * (xlims[1] - xlims[0])

            # ylims[0] -= 0.05 * (ylims[1] - ylims[0])
            # ylims[1] += 0.05 * (ylims[1] - ylims[0])

            # inset_ax.set_xscale('log')
            # inset_ax.set_yscale('log')

            xticks = inset_ax.get_xticks()
            xticks = [xlims[0], xlims[-1]]

            yticks = inset_ax.get_yticks()
            yticks = [ylims[0], ylims[-1]]

            xspread = xlims[1] - xlims[0]
            xlims[0] -= xspread * 0.1
            xlims[1] += xspread * 0.1

            yspread = ylims[1] - ylims[0]
            ylims[0] -= yspread * 0.1
            ylims[1] += yspread * 0.1
            if len(np.unique(ylims)) == 2:
                inset_ax.set_xlim(xlims)
            if len(np.unique(ylims)) == 2:
                inset_ax.set_ylim(ylims)

            inset_ax.set_xticks(xticks)
            inset_ax.set_yticks(yticks)

            inset_ax.tick_params(axis='x', labelrotation=0)

            mark_inset(scatter_ax, inset_ax, 2, 3, alpha=.25)

    else:
        logging.warning(f"no highlited indices for {well} {protocol} sweep{sweep}")

def do_profile_plots(baseline_profile_ax, params_df, protocol, well, sweep, args):

    subtraction_df = pd.read_csv(args.subtraction_df)
    times_fname = os.path.join(args.data_dir,
                               f"{args.experiment_name}-{protocol}-times.csv")
    trace, vp = get_data(well, protocol, args.data_dir,
                         args.experiment_name, sweep=sweep,
                         label=args.data_label)

    trace = trace.flatten()
    desc = vp.get_all_sections()

    # Temporary solver hack
    desc = np.vstack((desc, [[desc[-1, 1], np.inf, -80.0, -80.0]]))
    prot_func = make_voltage_function_from_description(desc)

    times = np.loadtxt(times_fname).flatten().astype(np.float64)
    voltages = np.array([prot_func(t, protocol_description=desc) for t in times])

    assert(np.all(np.isfinite(voltages)))

    spike_times, _ = detect_spikes(times, voltages, window_size=0)
    _, _, indices = remove_spikes(times, voltages, spike_times,
                                  args.removal_duration)

    row = subtraction_df[(subtraction_df.well == well) & (subtraction_df.protocol == protocol)
                            & (subtraction_df.sweep == sweep)].iloc[0]
    gleak, Eleak = row[['gleak_before', 'E_leak_before']].values.flatten().astype(np.float64)
    gleak = float(gleak)
    Eleak = float(Eleak)
    I_leak = gleak * (voltages - Eleak)

    if not args.infer_reversal_potential:
        E_rev = args.E_rev
    elif not args.use_artefact_model:
        E_rev = subtraction_df[(subtraction_df.well == well) & (subtraction_df.protocol == protocol)
                               & (subtraction_df.sweep == sweep)].iloc[0]['E_rev']
        E_rev = float(E_rev)
    else:
        assert(False)

    m_model = make_model_of_class(args.model_class, E_rev=E_rev)

    if args.use_artefact_model:
        model = ArtefactModel(m_model)
    else:
        model = m_model

    default_params = model.get_default_parameters()

    best_params = get_best_params(params_df)
    row = best_params.set_index(['well', 'protocol', 'sweep']).loc[(well, protocol, sweep)]
    param_labels = model.get_parameter_labels()
    params = row[param_labels].values.flatten().astype(np.float64)

    if args.default_parameters_file:
        default_parameters = np.loadtxt(os.path.join(args.default_parameters_file))
        m_model = make_model_of_class(args.model_class, voltage=prot_func,
                                      protocol_description=desc,
                                      default_parameters=default_parameters,
                                      times=times, E_rev=E_rev)
    else:
        if args.use_artefact_model:
            channel_model_params = params[:-no_artefact_parameters].copy()
        else:
            channel_model_params = params.copy()

        m_model = make_model_of_class(args.model_class, voltage=prot_func,
                                      protocol_description=desc,
                                      default_parameters=channel_model_params,
                                      times=times, E_rev=E_rev)

    if args.use_artefact_model:
        model = ArtefactModel(m_model)
    else:
        model = m_model

    default_params[m_model.GKr_index] = params[m_model.GKr_index]

    if args.use_artefact_model:
        default_params[-no_artefact_parameters:] = params[-no_artefact_parameters:]

    if args.fitting_case in ['0a', '0b', '0d']:
        solver = model.make_hybrid_solver_current(hybrid=False,
                                                  strict=False,
                                                  njitted=False)
    else:
        solver = model.make_hybrid_solver_current(hybrid=False,
                                                  strict=False,
                                                  njitted=False,
                                                  return_var='I_out')

    def compute_rmse(p):
        y = solver(p.flatten(), times=times, protocol_description=desc,
                   E_rev=E_rev)
        if args.fitting_case == '0d':
            y = y.flatten() + I_leak.flatten()

        return np.sqrt(np.mean((y[indices] - trace[indices])**2))

    plot_var = np.linspace(-0.05, 1.05, 100)

    params = [params + (default_params - params) * l for l in plot_var]
    scores = [compute_rmse(p.flatten()) for p in params]

    baseline_profile_ax.plot(plot_var, scores)

    baseline_profile_ax.set_ylabel('RMSE (pA)')

    baseline_profile_ax.axhline(min(*scores), color='grey', linestyle='--')
    baseline_profile_ax.axvline(0, color='grey')
    baseline_profile_ax.axvline(1.0, color='grey')

    profile_plotted = np.any(np.isfinite(scores))
    if profile_plotted:
        baseline_profile_ax.set_xlabel(r'$\lambda$')
        baseline_profile_ax.set_yscale('log')


def setup_grid(fig):
    no_columns = 2
    no_rows = 5
    gs = GridSpec(no_rows, no_columns, figure=fig, height_ratios=[.5, .5, .5, 1, 1])

    protocol_ax = fig.add_subplot(gs[0, :])
    occupations_ax = fig.add_subplot(gs[1, :])
    current_ax = fig.add_subplot(gs[2, :])
    scatter_ax = fig.add_subplot(gs[3, :])
    baseline_profile_ax = fig.add_subplot(gs[4, 1])
    rank_ax = fig.add_subplot(gs[4, 0])

    axs = occupations_ax, current_ax, protocol_ax, rank_ax, scatter_ax, baseline_profile_ax

    return axs


if __name__ == "__main__":
    main()
