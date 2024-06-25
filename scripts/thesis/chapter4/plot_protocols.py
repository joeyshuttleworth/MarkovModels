import matplotlib.pyplot as plt
import argparse

import numpy as np
import pandas as pd
import os
from matplotlib.gridspec import GridSpec

from matplotlib import rc

from markovmodels.model_generation import make_model_of_class
from markovmodels.utilities import setup_output_directory
from markovmodels.voltage_protocols import get_ramp_protocol_from_json

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 8})
rc('text', usetex=True)
rc('figure', dpi=500)


def main():

    parser = argparse.ArgumentParser('--figsize')
    parser.add_argument('data_dir')
    parser.add_argument('chrono_file')
    parser.add_argument('--output', '-o')
    parser.add_argument('--experiment_name', default='25112022_MW')
    parser.add_argument('--figsize', nargs=2, type=float, default=[4.65, 6])
    parser.add_argument('--noise', default=0.03)

    global args
    args = parser.parse_args()

    with open(args.chrono_file, 'r') as fin:
        lines = fin.read().splitlines()
        protocol_order = [line.split(' ')[0] for line in lines]

        protocol_order = [p for p in protocol_order
                          if p not in ['staircaseramp2', 'staircaseramp1_2']]

    print(len(protocol_order), protocol_order)

    output_dir = setup_output_directory(args.output, 'plot_protocols')
    protocol_dir = os.path.join(args.data_dir, 'protocols')

    all_times = []
    for protocol in protocol_order:
        _, desc = get_ramp_protocol_from_json(protocol, protocol_dir,
                                              args.experiment_name)
        t_ends = desc[:, 3].flatten()
        t_end = t_ends[np.isfinite(t_ends)].max()
        all_times.append(t_end)

    relabel_dict = {prot: r'$d_{' f"{i+1}" r'}$'
                    for i, prot in enumerate(protocol_order)}
    print(relabel_dict)

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    plot_axes, label_axes = setup_grid(fig, no_protocols=len(protocol_order))

    for i, protocol in enumerate(protocol_order):
        voltage_func, desc = get_ramp_protocol_from_json(protocol, protocol_dir,
                                                         args.experiment_name)

        t_ends = desc[:, 3].flatten()
        t_end = t_ends[np.isfinite(t_ends)].max()
        times = np.arange(0, t_end, 0.1)

        voltages = np.array([voltage_func(t) for t in times])
        model = make_model_of_class('model3', voltage=voltage_func, protocol_description=desc,
                                    times=times)
        plot_axes[i].plot(times / t_end, voltages, color='black', lw=.5)

        current = model.SimulateForwardModel()
        observations = np.random.normal(current, args.noise)

        plot_axes[i].plot(times/t_end, observations, color='grey', lw=.5)

        plot_axes[i].set_xticks([0, 1])
        plot_axes[i].set_xticklabels(['', f"{t_end/3:.0f}"])

        if i % 2 == 0:
            # plot_axes[i].set_ylabel(r'$V_\text{m}$')
            pass
        else:
            plot_axes[i].set_ylabel('')
            plot_axes[i].set_yticks([])

        # label_axes[i].text(-0.5, 0.5, relabel_dict[protocol], size=11)
        label_axes[i].axis('off')

    plot_axes[-1].set_xticks([0, 1])
    # plot_axes[-1].set_xlabel(r'$t$ (ms)')

    for ax in plot_axes:
        ax.set_ylim([-120, 60])

    fig.savefig(os.path.join(output_dir, "protocols_figure.png"))


def setup_grid(fig, no_protocols):
    # No protocols 12
    no_columns = 2
    no_rows = no_protocols

    gs = GridSpec(no_rows, no_columns, figure=fig, width_ratios=[0.75, 1])

    plot_axs = []
    label_axs = []
    for i in range(no_rows):
        plot_axs.append(fig.add_subplot(gs[i, 1]))
        label_axs.append(fig.add_subplot(gs[i, 0]))

    for ax in plot_axs:
        ax.spines[['top', 'right']].set_visible(False)

    for ax in label_axs:
        ax.spines[['top', 'right']].set_visible(False)

    return label_axs, plot_axs


if __name__ == '__main__':
    main()
