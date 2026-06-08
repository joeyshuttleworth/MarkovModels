import matplotlib.pyplot as plt
import argparse

import numpy as np
import pandas as pd
import os
from matplotlib.gridspec import GridSpec

from matplotlib import rc

from markovmodels.utilities import setup_output_directory
from markovmodels.voltage_protocols import get_protocol_desc_from_json, make_voltage_function_from_description

rc('figure', dpi=500)
fontsize = 9
rc('font', **{'size': 9})
rc('xtick', labelsize=9)
rc('ytick', labelsize=9)

def main():

    parser = argparse.ArgumentParser('--figsize')
    parser.add_argument('data_dir')
    parser.add_argument('chrono_file')
    parser.add_argument('--output', '-o')
    parser.add_argument('--experiment_name', default='25112022_MW')
    parser.add_argument('--double_column', action='store_true')
    parser.add_argument('--figsize', nargs=2, type=float, default=[4.65, 7.5])
    parser.add_argument('--noise', default=0.03)

    global args
    args = parser.parse_args()

    with open(args.chrono_file, 'r') as fin:
        lines = fin.read().splitlines()
        protocol_order = [line.split(' ')[0] for line in lines]

        protocol_order = [p for p in protocol_order
                          if p not in ['staircaseramp2', 'staircaseramp1_2']]

    global output_dir
    output_dir = setup_output_directory(args.output, 'plot_protocols')

    if args.double_column:
        output_dir = setup_output_directory(args.output, 'plot_protocols_double_column')

    protocol_dir = os.path.join(args.data_dir, 'protocols')

    all_times = []
    for protocol in protocol_order:
        desc = get_protocol_desc_from_json(protocol, protocol_dir,
                                              args.experiment_name)
        t_ends = desc[:, 3].flatten()
        t_end = t_ends[np.isfinite(t_ends)].max()
        all_times.append(t_end)

    relabel_dict = {prot: r'$d_{' f"{i+1}" r'}$'
                    for i, prot in enumerate(protocol_order)}

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    label_axs, plot_axes = setup_grid(fig, no_protocols=len(protocol_order),
                                       double_column=args.double_column)

    voltage_func = None
    for i, protocol in enumerate(protocol_order):
        desc = get_protocol_desc_from_json(protocol, protocol_dir,
                                           args.experiment_name)
        if not voltage_func:
            voltage_func = make_voltage_function_from_description(desc)

        t_ends = desc[:, 1].flatten()
        t_end = t_ends[np.isfinite(t_ends)].max()
        times = np.arange(0, t_end, 2)

        voltages = np.array([voltage_func(t, protocol_description=desc) for t in times])
        plot_axes[i].plot(times / t_end, voltages, color='black', lw=.5)

        # plot_axes[i].set_title(relabel_dict[protocol])
        label_axs[i].text(0.5, 0.5, relabel_dict[protocol], fontsize=9)

        plot_axes[i].set_xticks([0, 1])
        plot_axes[i].set_xticklabels(['0.0', f"{t_end*1e-3:.0f}"], fontsize=8)

        plot_axes[i].set_ylabel(r'$V_\text{cmd}$', rotation=0)

        plot_axes[i].set_yticks([-100, 0])
        plot_axes[i].set_yticklabels([f"{int(lab)} mV" for lab in plot_axes[i].get_yticks()])

    plot_axes[-1].set_xticks([0, 1])
    plot_axes[-1].set_xlabel(r'$t$ (s)')

    if args.double_column:
        plot_axes[-2].set_xticks([0, 1])
        plot_axes[-2].set_xlabel(r'$t$ (s)')

        for ax in plot_axes[1::2]:
            ax.set_yticks([])
            ax.set_ylabel('')

    for ax in plot_axes:
        ax.set_ylim([-120, 60])

    fig_fname = os.path.join(output_dir, "protocols_figure.pdf")
    print(fig_fname)

    fig.savefig(fig_fname)


def setup_grid(fig, no_protocols, double_column=False):

    if not double_column:
        # No protocols 12
        assert(False)
        no_columns = 2
        no_rows = no_protocols

        gs = GridSpec(no_rows, no_columns, figure=fig, width_ratios=[1, 0.15])

        plot_axs = []
        for i in range(no_rows):
            plot_axs.append(fig.add_subplot(gs[i, 2]))

        for ax in plot_axs:
            ax.spines[['top', 'right']].set_visible(False)

    else:
        # No protocols
        assert(no_protocols % 2 == 0)
        no_columns = 4
        no_rows = int(no_protocols / 2)

        gs = GridSpec(no_rows, no_columns, figure=fig, width_ratios=[0.05, 1, 1, 0.05])

        plot_axs = []
        label_axs = []
        for i in range(no_rows):
            plot_axs.append(fig.add_subplot(gs[i, 1]))
            plot_axs.append(fig.add_subplot(gs[i, 2]))

        label_axs = []
        for i in range(no_rows):
            label_axs.append(fig.add_subplot(gs[i, 0]))
            label_axs.append(fig.add_subplot(gs[i, -1]))

        for ax in plot_axs:
            ax.spines[['top', 'right']].set_visible(False)

        for ax in label_axs:
            ax.spines[['top', 'right']].set_visible(False)
            ax.axis('off')

    return label_axs, plot_axs


if __name__ == '__main__':
    main()
