from markovmodels import common
from markovmodels.BeattieModel import BeattieModel

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
    gs = GridSpec(6, 3, figure=fig, width_ratios=[.05, 1, 1])
    axes = [fig.add_subplot(cell) for cell in gs]
    return axes


def main():

    parser = argparse.ArgumentParser('--figsize')
    parser.add_argument('--output', '-o')
    parser.add_argument('--figsize', nargs=2, type=float, default=[4.65, 6])
    parser.add_argument('--noise', default=0.03)

    global args
    args = parser.parse_args()

    output_dir = common.setup_output_directory(args.output, 'plot_protocols')

    all_times = []
    for protocol in sorted(protocols_list):
        _, times, _ = common.get_ramp_protocol_from_csv(protocol)
        all_times.append(times)

    t_max = max([max(row) for row in all_times])

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    axes = setup_axes(fig)

    axes[1].set_title(r'$V$ (mV)')
    axes[2].set_title(r'$I_{\textrm{Kr}}$ (nA)')

    for i, protocol in enumerate(protocols_list):
        voltage_func, times, desc = common.get_ramp_protocol_from_csv(protocol)
        model = BeattieModel(voltage_func, times=times,
                             protocol_description=desc)
        voltages = np.array([voltage_func(t) for t in times])

        axes[i*3 + 1].plot(times*1e-3, voltages, color='black', lw=.5)

        current = model.SimulateForwardModel()
        observations = np.random.normal(current, args.noise)

        axes[i*3 + 2].plot(times*1e-3, observations, color='grey', lw=.5)
        # axes[i*3 + 2].yaxis.tick_right()
        # axes[i*3 + 2].yaxis.set_label_position('right')

        # axes[i*3 + 2].set_ylabel(r'$I_{\textrm{Kr}}$ / nA')
        # axes[i*3 + 1].set_ylabel(r'$V_{\textrm{m}}$ / mV')

        axes[i*3].text(-0.5, 0.5, relabel_dict[protocol], size=11)

        axes[i*3].axis('off')

        # axes[i*3 + 1].set_xlim([0, t_max])
        # axes[i*3 + 2].set_xlim([0, t_max])

        # if i < len(protocols_list) - 1:
        #     axes[i*3 + 1].set_xticks([])
        #     axes[i*3 + 2].set_xticks([])

        for side in ['top', 'right']:
            axes[i*3 + 1].spines[side].set_visible(False)
            axes[i*3 + 2].spines[side].set_visible(False)

    axes[5*3 + 1].set_xlabel('$t$ (s)')
    axes[5*3 + 2].set_xlabel('$t$ (s)')

    fig.savefig(os.path.join(output_dir, "Fig3.png"))

if __name__ == '__main__':
    main()
