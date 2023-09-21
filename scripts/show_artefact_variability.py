import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from markovmodels.utilities import setup_output_directory
from markovmodels.model_generation import make_model_of_class
from markovmodels.voltage_protocols import get_ramp_protocol_from_csv
from markovmodels.ArtefactModel import ArtefactModel


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('subtraction_summary_file')
    parser.add_argument('--figsize', default=[12, 9], nargs=2, type=int)
    parser.add_argument('--output')
    parser.add_argument('--leak_subtraction_file')
    parser.add_argument('--qc_estimates_file')
    parser.add_argument('--selection_file')

    global args
    args = parser.parse_args()

    global output_dir
    output_dir = setup_output_directory(args.output, 'show_artefact_variability')

    use_literature_range()

    if args.qc_estimates_file:
        qc_estimates_df = pd.read_csv(args.qc_estimates_file)
        leak_subtraction_df = pd.read_csv(args.leak_subtraction_file)
        if args.selection_file:
            with open(args.selection_file) as fin:
                selected_wells = fin.read().splitlines()
        else:
            selected_wells = None
        use_qc_estimates(qc_estimates_df, leak_subtraction_df, selected_wells)


def use_qc_estimates(qc_df, leak_df, passed_wells):
    qc_df = qc_df[qc_df.drug == 'before']

    leak_df = leak_df[leak_df.well.isin(passed_wells)]

    leak_df = leak_df.set_index(['well', 'protocol', 'sweep'])

    model_class = 'model3'
    protocol = 'staircaseramp1'
    voltage_func, times, desc = get_ramp_protocol_from_csv(protocol)
    c_model = make_model_of_class(model_class, times, voltage=voltage_func,
                                  protocol_description=desc,
                                  tolerances=(1e-7, 1e-7))

    artefact_model = ArtefactModel(c_model)
    a_solver = artefact_model.make_hybrid_solver_states(hybrid=False, njitted=False)
    a_solver_i = artefact_model.make_forward_solver_current(njitted=False)

    def get_Vm(p):
        return a_solver(p)[:, -1]

    fig = plt.figure(figsize=args.figsize)
    axs = fig.subplots(2)

    axs[0].plot(times*1e-3, c_model.SimulateForwardModel())
    ylims_1 = axs[0].get_ylim()

    axs[1].plot(times*1e-3, [voltage_func(t) for t in times])

    for index, row in qc_df.iterrows():
        if row['protocol'] not in ['staircaseramp1', 'staircaseramp2']:
            continue

        well, protocol, sweep, R_s, C_m = row[['well', 'protocol', 'sweep',
                                               'Rseal', 'Cm']]
        if well not in passed_wells:
            continue

        # Convert from base units
        R_s = R_s * 1e-12
        C_m = C_m * 1e9

        try:
            g_leak, E_leak = leak_df.loc[well,
                                         protocol,
                                         sweep][['pre-drug leak conductance',
                                                 'pre-drug leak reversal']]
        except KeyError:
            continue

        p = artefact_model.get_default_parameters()
        p[-7:] = .0, .0, .0, .0, .0, C_m, R_s

        print(p)
        sol = a_solver(p)

        print(sol)

        axs[0].plot(times*1e-3, a_solver_i(p),
                    color='grey', alpha=.5)
        axs[1].plot(times*1e-3, sol[:, -1],
                    color='grey', alpha=.5)

    axs[0].set_ylim(ylims_1)
    fig.savefig(os.path.join(output_dir,
                             'artefact_variability_from_qc_estimates'))
    fig.clf()


def use_literature_range():
    model_class = 'model3'
    protocol = 'staircaseramp1'
    voltage_func, times, desc = get_ramp_protocol_from_csv(protocol)
    c_model = make_model_of_class(model_class, times, voltage=voltage_func,
                                  protocol_description=desc,
                                  tolerances=(1e-7, 1e-7))

    artefact_model = ArtefactModel(c_model)

    _p = artefact_model.get_default_parameters()
    p = _p.copy()

    fig = plt.figure(figsize=args.figsize)
    axs = fig.subplots(2, 2)

    axs[0, 0].plot(times*1e-3, c_model.SimulateForwardModel(), color='grey')
    axs[0, 0].set_title('no artefacts')
    axs[0, 1].plot(times*1e-3, c_model.SimulateForwardModel(), color='grey')
    axs[1, 0].plot(times*1e-3, c_model.SimulateForwardModel(), color='grey')
    axs[1, 1].plot(times*1e-3, c_model.SimulateForwardModel(), color='grey')

    axs[0, 1].plot(times*1e-3, artefact_model.SimulateForwardModel())
    axs[0, 1].set_title(f"C_m={p[-2]}, R_s={p[-1]}")

    p = _p.copy()
    p[-1] = 20e-3 #MOhm
    p[-2] = 20e-3
    axs[1, 0].plot(times*1e-3, artefact_model.SimulateForwardModel(p))
    axs[1, 0].set_title(f"C_m={p[-2]}nF, R_s={p[-1]}GOhm")
    p[-7] = 2e-3 #gS

    axs[1, 1].plot(times*1e-3, artefact_model.SimulateForwardModel(p))
    axs[1, 1].set_title(f"C_m={p[-2]}nF, R_s={p[-1]}GOhm, " \
                        f"g_leak=2nS E_leak=0")

    fig.savefig(os.path.join(output_dir, "artefact_impact"))

    for ax in axs.flatten():
        ax.cla()

    voltages = np.array([voltage_func(t) for t in times])

    axs[0, 0].plot(times*1e-3, voltages, color='grey')
    axs[0, 0].set_title('no artefacts')
    axs[0, 1].plot(times*1e-3, voltages, color='grey')
    axs[1, 0].plot(times*1e-3, voltages, color='grey')
    axs[1, 1].plot(times*1e-3, voltages, color='grey')

    a_solver = artefact_model.make_hybrid_solver_states(hybrid=False)

    axs[0, 1].plot(times*1e-3, a_solver()[:, -1])
    axs[0, 1].set_title(f"C_m={p[-2]}nF, R_s={p[-1]}GOhm")

    p = _p.copy()
    p[-1] = 20e-3 #MOhm
    p[-2] = 20e-3
    axs[1, 0].plot(times*1e-3, a_solver(p)[:, -1])
    axs[1, 0].set_title(f"C_m={p[-2]}nF, R_s={p[-1]}MOhm")
    p[-7] = 2e-3 #gS

    axs[1, 1].plot(times*1e-3, a_solver(p)[:, -1])
    axs[1, 1].set_title(f"C_m={p[-2]}nF, R_s={p[-1]}MOhm, " \
                        f"g_leak=2nS E_leak=0")

    fig.savefig(os.path.join(output_dir, "artefact_impact_voltage"))


if __name__ == '__main__':
    main()
