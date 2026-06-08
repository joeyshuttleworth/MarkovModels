import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging

from markovmodels.utilities import setup_output_directory
from markovmodels.model_generation import make_model_of_class
from markovmodels.voltage_protocols import get_ramp_protocol_from_csv
from markovmodels.ArtefactModel import ArtefactModel, no_artefact_parameters


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('subtraction_summary_file')
    parser.add_argument('--figsize', default=[12, 9], nargs=2, type=int)
    parser.add_argument('--output', '-o')

    global args
    args = parser.parse_args()

    global output_dir
    output_dir = setup_output_directory(args.output, 'show_artefact_variability')

    use_literature_range()


def use_literature_range():
    model_class = 'model3'
    protocol = 'staircaseramp1'
    voltage_func, times, desc = get_ramp_protocol_from_csv(protocol)
    c_model = make_model_of_class(model_class, times, voltage=voltage_func,
                                  protocol_description=desc,
                                  tolerances=(1e-7, 1e-7))

    artefact_model = ArtefactModel(c_model)

    _p = artefact_model.get_default_parameters()

    # Convert current to pA
    _p[-8] *= 1e3

    p = _p.copy()

    c_p = p.copy()[:c_model.get_default_parameters().shape[0]]

    fig = plt.figure(figsize=args.figsize)
    axs = fig.subplots(2, 2)

    axs[0, 0].plot(times*1e-3, c_model.SimulateForwardModel(c_p), color='grey')
    axs[0, 0].set_title('no artefacts')
    axs[0, 1].plot(times*1e-3, c_model.SimulateForwardModel(c_p), color='grey')
    axs[1, 0].plot(times*1e-3, c_model.SimulateForwardModel(c_p), color='grey')
    axs[1, 1].plot(times*1e-3, c_model.SimulateForwardModel(c_p), color='grey')

    axs[0, 1].plot(times*1e-3, artefact_model.SimulateForwardModel(_p))
    axs[0, 1].set_title(f"C_m={p[-2]}, R_s={p[-1]}")

    p = _p.copy()
    p[-1] = 20e-3 #MOhm
    p[-2] = 20e-3
    p[-7] = 2 # uS
    axs[1, 0].plot(times*1e-3, artefact_model.SimulateForwardModel(p))
    axs[1, 0].set_title(f"C_m={p[-2]}nF, R_s={p[-1]}GOhm")

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

    axs[0, 1].plot(times*1e-3, a_solver(p)[:, -1])
    axs[0, 1].set_title(f"C_m={p[-2]}nF, R_s={p[-1]}GOhm")

    p = _p.copy()
    p[-1] = 20e-3 #GOhm
    p[-2] = 20e-3 # nF
    p[-7] = 2 # uS
    axs[1, 0].plot(times*1e-3, a_solver(p)[:, -1])
    axs[1, 0].set_title(f"C_m={p[-2]}nF, R_s={p[-1]}MOhm")

    axs[1, 1].plot(times*1e-3, a_solver(p)[:, -1])
    axs[1, 1].set_title(f"C_m={p[-2]}nF, R_s={p[-1]}MOhm, " \
                        f"g_leak=2nS E_leak=0")

    fig.savefig(os.path.join(output_dir, "artefact_impact_voltage"))


if __name__ == '__main__':
    main()
