# Example command: python3 scripts/thesis/chapter4/optimisation_results.py ~/data/25112022_MW_FF_processed/traces 0a ~/data/sydney_fitting/25112022MW/Case0a/model3/combine_fitting_results/combined_fitting_results.csv model3 -w B09 --experiment_name 25112022_MW --sweep 1 --output tmp


import argparse
import os

import matplotlib
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
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
from markovmodels.fitting import get_best_params, infer_reversal_potential, make_prediction
from markovmodels.ArtefactModel import ArtefactModel, no_artefact_parameters
from markovmodels.utilities import setup_output_directory, get_data, get_all_wells_in_directory
from markovmodels.voltage_protocols import get_protocol_list, get_ramp_protocol_from_json, make_voltage_function_from_description
from markovmodels.voltage_protocols import remove_spikes, detect_spikes, get_ramp_protocol_from_csv
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

cutoff_threshold = 1.005

mpl.rcParams['axes.formatter.useoffset'] = True
plt.rcParams["axes.formatter.use_mathtext"] = True

# Threshold to add offset
plt.rcParams["axes.formatter.offset_threshold"] = 2


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir')
    parser.add_argument('--output', '-o')
    parser.add_argument('--figsize', '-f', nargs=2, type=float, default=[5.7, 8])
    parser.add_argument('--reversal', default=-91.71, type=float)
    parser.add_argument('--experiment_name', '-e', default='25112022_MW')

    global args
    args = parser.parse_args()

    output_dir = setup_output_directory(args.output, 'artefact_reversal_example')

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    protocol_ax_V, protocol_ax_I, voltage_ax, current_ax, iv_ax = setup_plot_grid(fig)

    voltage_func, desc = get_ramp_protocol_from_json('staircaseramp1',
                                                            os.path.join(args.data_dir,
                                                                         'protocols'),
                                                            args.experiment_name)

    gleak = 5e1
    Eleak = 0.0

    artefact_params = {
        'E_Kr': args.reversal,
        'gleak': gleak,
        'Eleak': Eleak,
        'gleak_leftover': 0.0,
        'Eleak_leftover': 0.0,
        'V_off': 0,
        'R_series': 10e-3,
        'C_m': 5e-3,
    }

    tstart = 0.0
    tend = max([v for v in desc[:, 1] if v != np.inf])

    times = np.arange(tstart, tend, .5)
    desc = np.vstack((desc, [[desc[-1, 1], np.inf, -80.0, -80.0]]))
    Vcmd = np.array([voltage_func(t, protocol_description=desc) for t in times])

    protocol_ax_V.plot(times, Vcmd, color='black')
    protocol_ax_V.axvspan(ramp_start, ramp_end, alpha=.5, color='grey')

    reversal_ramp_window = [line for line in desc if line[2] != line[3]][-1][:2]
    istart = np.argmax(times > reversal_ramp_window[0])
    iend = np.argmax(times > reversal_ramp_window[1])

    default_parameters = np.loadtxt(os.path.join('data',
                                                 'Beattie_Sinusoidal_params.csv'),
                                                 delimiter=',').flatten()

    print(default_parameters)

    # Use staircaseramp protocol as example
    model = make_model_of_class('BeattieModel', times=times,
                                voltage=voltage_func,
                                protocol_description=desc,
                                default_parameters=default_parameters)

    a_model = ArtefactModel(model)

    p = a_model.get_default_parameters()
    p[8] = 20.0

    p[-no_artefact_parameters:] = list(artefact_params.values())
    p[-no_artefact_parameters - 1] = p[-no_artefact_parameters - 1]

    a_solver_states = a_model.make_hybrid_solver_states(hybrid=False)
    Vm = a_solver_states(p)[:, -1].flatten()
    I = a_model.SimulateForwardModel(p, return_var='I_out')
    voltage_ax.plot(times[istart:iend], Vm[istart:iend] - Vcmd[istart:iend],
                    label=r'$V_\text{m} - V_\text{cmd}$')
    current_ax.plot(times[istart:iend], I[istart:iend],
                    label=r'$I_\text{out}$')
    # Set conductance to 0
    p[8] = 0.0
    Vm2 = a_solver_states(p)[:, -1].flatten()
    I2 = a_model.SimulateForwardModel(p, return_var='I_out')
    voltage_ax.plot(times[istart:iend], Vm2[istart:iend] - Vcmd[istart:iend],
                    label=r'$\tilde V_\text{m} - V_\text{cmd}$')
    # voltage_ax.plot(times[istart:iend], Vm2[istart:iend],
    #                 label=r'$V_\text{cmd}$')
    current_ax.plot(times[istart:iend], I2[istart:iend],
                    label=r'$\tilde I_\text{out}$')

    protocol_ax_I.axvspan(ramp_start, ramp_end, alpha=.5, color='grey')
    protocol_ax_I.plot(times, I, label=r'$I_\text{out}$')
    protocol_ax_I.plot(times, I2, label=r'$\tilde I_\text{out}$')

    voltage_ax.legend()
    current_ax.legend()

    iv_ax.plot(Vm[istart:iend], I[istart:iend], label='Artefact Model')
    iv_ax.plot(Vm2[istart:iend], I2[istart:iend], label=r'Artefact Model with $x_\text{O} = 0$')
    iv_ax.legend()

    fig.savefig(os.path.join(output_dir, 'artefact_reversal_example.pdf'))

    fig.clf()

    axs = fig.subplots(2)
    axs[0].plot(times, I - gleak*(Vcmd - Eleak))
    axs[0].plot(times, I2 - gleak*(Vcmd - Eleak))

    axs[1].plot(times, Vm)
    axs[1].plot(times, Vm2)
    axs[1].plot(times, Vcmd)

    fig.savefig(os.path.join(output_dir, 'currents.pdf'))



def setup_plot_grid(fig):
    no_rows = 4
    no_columns = 1
    gs = GridSpec(no_rows, no_columns, figure=fig,
                  height_ratios=[.25, 1, 1, 1.75])

    protocol_ax_V = fig.add_subplot(gs[0, 0])
    protocol_ax_I = fig.add_subplot(gs[0, 0])
    voltage_ax = fig.add_subplot(gs[1, 0])
    current_ax = fig.add_subplot(gs[2, 0])
    iv_ax = fig.add_subplot(gs[3, 0])

    for ax in (protocol_ax, protocol_ax_I, voltage_ax, current_ax, iv_ax):
        ax.cla()
        ax.spines[['top', 'right']].set_visible(False)

    return protocol_ax_V, protocol_ax_I, voltage_ax, current_ax, iv_ax


if __name__ == "__main__":
    main()
