from MarkovModels import common
from MarkovModels.BeattieModel import BeattieModel

import matplotlib.pyplot as plt
import argparse

import numpy as np
import pandas as pd
import os
from matplotlib.gridspec import GridSpec

from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 8})
rc('text', usetex=True)
rc('figure', dpi=500)

protocols_list = sorted(['staircaseramp1', 'sis', 'spacefill19', 'hhbrute3gstep', 'wangbrute3gstep'])
protocols_list = ['longap'] + protocols_list

relabel_dict = {p: f"$d_{i+1}$" for i, p in enumerate([p for p in
                                                       protocols_list if p != 'longap'])}

relabel_dict['longap'] = '$d_0$'


def setup_axes(fig):
    gs = GridSpec(2, 5, figure=fig)
    voltage_axes = [fig.add_subplot(gs[1, i]) for i in range(5)]
    current_axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
    return voltage_axes, current_axes


def main():

    parser = argparse.ArgumentParser('--figsize')
    parser.add_argument('--output', '-o')
    parser.add_argument('--figsize', nargs=2, type=float, default=[4.5, 1.8])
    parser.add_argument('--noise', default=0.03)

    global args
    args = parser.parse_args()

    output_dir = common.setup_output_directory(args.output, 'plot_protocols')

    all_times = []
    for protocol in sorted(protocols_list):
        _, times, _ = common.get_ramp_protocol_from_csv(protocol)
        all_times.append(times)

    # t_max = max([max(row) for row in all_times])

    fig = plt.figure(figsize=args.figsize)
    current_axes, voltage_axes = setup_axes(fig)

    for i, protocol in enumerate(protocols_list[1:]):
        voltage_func, times, desc = common.get_ramp_protocol_from_csv(protocol)
        model = BeattieModel(voltage_func, times=times,
                             protocol_description=desc)
        voltages = np.array([voltage_func(t) for t in times])

        voltage_axes[i].plot(times*1e-3, voltages, color='black', lw=.5)

        voltage_axes[i].set_xlabel('')
        voltage_axes[i].set_yticks([])
        voltage_axes[i].set_xticks([])
        current_axes[i].set_yticks([])
        current_axes[i].set_ylim([-3, 5])

        voltage_axes[i].set_title(r'$d_{' f"{i+1}" "}$")

        current = model.SimulateForwardModel()
        observations = np.random.normal(current, args.noise)

        current_axes[i].plot(times*1e-3, observations, color='grey', lw=.3,
                             alpha=.8)
        for side in ['top', 'right']:
            current_axes[i].spines[side].set_visible(False)
            voltage_axes[i].spines[side].set_visible(False)

        if i != 1:
            current_axes[i].set_ylabel('')
            current_axes[i].set_yticks([])

    voltage_axes[0].set_ylabel(r'$V$ (mV)', fontsize=8)
    voltage_axes[0].set_yticks([-80, 0])
    current_axes[0].set_ylabel(r'$I_\textrm{Kr}$ (nA)', fontsize=8)
    current_axes[0].set_yticks([-2, 2])

    current_axes[2].set_xlabel('$t$ (s)', loc='right')

    # x, y = current_axes[2].xaxis.get_label().get_position()
    # current_axes[2].xaxis.get_label().set_position([y, x+0.5])

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "synthetic_voltage_plots.png"))


if __name__ == '__main__':
    main()
