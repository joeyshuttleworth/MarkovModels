
from BeattieModel import BeattieModel
import argparse
import os
import common
import numpy as np
import logging
import matplotlib.pyplot as plt

def simulate_protocol(model, name, output_dir):
    print(f"Plotting {name}")
    current, S1 = model.SimulateForwardModelSensitivities()
    S1n = S1 * np.array(model.get_default_parameters())[None, :]

    param_labels = ['S(p' + str(i + 1) + ',t)' for i in range(model.n_params)]
    state_occupancies = model.GetStateVariables()

    fig = plt.figure(figsize=(14, 12))
    ax1 = fig.add_subplot(411)
    ax1.plot(model.times, model.GetVoltage())
    ax1.grid(True)
    ax1.set_xticklabels([])
    ax1.set_ylabel('Voltage (mV)')

    ax2 = fig.add_subplot(412)
    ax2.plot(model.times, model.SimulateForwardModel())
    ax2.grid(True)
    ax2.set_xticklabels([])
    ax2.set_ylabel('Current (nA)')
    ax3 = fig.add_subplot(413)
    ax3.plot(model.times, state_occupancies, label=model.state_labels + ['IC'])
    ax3.legend(ncol=4)
    ax3.grid(True)
    ax3.set_xticklabels([])
    ax3.set_ylabel('State occupancy')
    ax4 = fig.add_subplot(414)
    for i in range(model.n_params):
        ax4.plot(model.times, S1n[:, i], label=param_labels[i])
    ax4.legend(ncol=3)
    ax4.grid(True)
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Sensitivities')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_sensitivities"))

    S1n = S1 * model.get_default_parameters()
    H = np.dot(S1n.T, S1n)
    print(H)
    eigvals = np.linalg.eigvals(H)
    print('Eigenvalues of H:\n{}'.format(eigvals.real))

    # Plot the eigenvalues of H, shows the condition of H
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111)
    for i in eigvals:
        ax.axhline(y=i, xmin=0.25, xmax=0.75)
    ax.set_yscale('log')
    ax.set_xticks([])
    ax.grid(True)
    fig.savefig(os.path.join(output_dir, f"{name}_H_eigenvalues"))


def main():
    parser = argparse.ArgumentParser(description="Plot output from different protocols")
    output_dir = os.path.join('output', 'simulate_protocols')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for fname in os.listdir(common.get_protocol_directory()):
        fname, ext = os.path.splitext(fname)
        if fname[0] == '#':
            continue
        if ext != ".csv":
            logging.warning(f"Using file with extension {ext}")
        protocol, t_start, t_end, t_step = common.get_protocol(os.path.splitext(fname)[0])
        times = np.linspace(t_start, t_end, int((t_end-t_start)/t_step))
        model = BeattieModel(protocol, times)
        simulate_protocol(model, fname, output_dir)


if __name__ == "__main__":
    main()
