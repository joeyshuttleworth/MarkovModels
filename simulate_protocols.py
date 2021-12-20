
from BeattieModel import BeattieModel
import argparse
import os
import common
import numpy as np
import logging
import matplotlib.pyplot as plt
import multiprocessing


def simulate_protocol(model, name, output_dir):
    fig = plt.figure(figsize=(14, 12))
    axs = fig.subplots(4)
    print(f"Plotting {name} to {output_dir}")

    model.set_tolerances(1e-7, 1e-9)

    current, S1 = model.SimulateForwardModelSensitivities()
    S1n = S1 * np.array(model.get_default_parameters())[None, :]

    param_labels = ['S(p' + str(i + 1) + ',t)' for i in range(model.n_params)]
    state_occupancies = model.GetStateVariables()

    axs[0].plot(model.times, model.GetVoltage(), label='generated voltage function')
    axs[0].plot()
    axs[0].grid(True)
    axs[0].set_xticklabels([])
    axs[0].set_ylabel('Voltage (mV)')

    axs[1].plot(model.times, current)
    axs[1].grid(True)
    axs[1].set_xticklabels([])
    axs[1].set_ylabel('Current (nA)')
    axs[2].plot(model.times, state_occupancies, label=model.state_labels + ['IC'])
    axs[2].legend(ncol=4)
    axs[2].grid(True)
    axs[2].set_xticklabels([])
    axs[2].set_ylabel('State occupancy')
    for i in range(model.n_params):
        axs[3].plot(model.times, S1n[:, i], label=param_labels[i])
    axs[3].legend(ncol=3)
    axs[3].grid(True)
    axs[3].set_xlabel('Time (ms)')
    axs[3].set_ylabel('Sensitivities')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{name}_sensitivities"))

    for ax in axs:
        ax.cla()
    plt.close(fig)

    # Now plot just the voltage and current
    fig = plt.figure(figsize=(14, 12))
    axs = fig.subplots(2)

    axs[0].set_title(name)

    axs[0].plot(model.times, model.GetVoltage())
    axs[1].plot(model.times, model.SimulateForwardModel())
    axs[1].set_xlabel('time /ms')
    axs[1].set_ylabel('current /nA')
    axs[0].set_ylabel('voltage / mV')

    fig.savefig(os.path.join(output_dir, f"{name}_protocol"))

    for ax in axs:
        ax.cla()
    plt.close(fig)

    print(f"{name} finished")


def main():
    parser = argparse.ArgumentParser(description="Plot output from different protocols")
    parser.add_argument("--protocols", "-p", default=[], type=str, nargs='+')

    args = parser.parse_args()

    output_dir = os.path.join('output', 'simulate_protocols')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tasks = []
    for fname in os.listdir(common.get_protocol_directory()):
        fname, ext = os.path.splitext(fname)
        if fname[0] == '#':
            continue
        print(fname, args.protocols)
        if fname in args.protocols or len(args.protocols) == 0:
            func(fname, ext, output_dir)


def func(protocol_name, ext, output_dir):
    if ext != ".csv":
        logging.warning(f"Using file with extension {ext}")

    protocol, t_start, t_end, t_step = common.get_protocol_from_csv(protocol_name)
    times = np.linspace(t_start, t_end, int((t_end - t_start) / t_step))
    model = BeattieModel(protocol, times, Erev=common.calculate_reversal_potential(298, K_out=130, K_in=5))

    model.default_parameters = np.array([0.00023215680795174809, 0.07422110165735675, 2.477501557744992e-05, 0.04414799725791213,
                                        0.11023652619943154, 0.015996823969951217, 0.015877336172564104, 0.027816696279347616, 49.70368237942998])
    simulate_protocol(model, protocol_name, output_dir)


if __name__ == "__main__":
    main()
