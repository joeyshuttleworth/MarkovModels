#!/usr/bin/env python3

from MarkovModels import common
import os
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 8})
rc('text', usetex=True)
rc('figure', dpi=600)


def create_axes(fig):
    axs = fig.subplots(3, 4)
    return [ax for row in axs for ax in row]


protocol_order = [
    'staircaseramp1',
    'sis',
    'spacefill19',
    'hhbrute3gstep',
    'wangbrute3gstep',
    'longap',
    'rtov',
    'spacefill10',
    'spacefill26',
    'wangsobol3step',
    'rvot',
    'hhsobol3step',
]


def main():
    description = ""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output_dir", "-o", help="Directory to output plots to.\
    By default a new directory will be generated", default=None)
    parser.add_argument("--figsize", default=(4, 2.6), nargs=2, type=float)

    global args
    args = parser.parse_args()

    global output_dir
    output_dir = common.setup_output_directory(args.output_dir, 'voltage_plots')

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)

    axs = create_axes(fig)

    for i, protocol in enumerate(protocol_order):
        prot_func, times, desc = common.get_ramp_protocol_from_csv(protocol)
        axs[i].plot(times/1e3, [prot_func(t) for t in times], color='black',
                    lw=.5)
        axs[i].set_title(r'$d_{' f"{i}" r'}$')
        axs[i].set_yticks([])
        axs[i].spines.right.set_visible(False)
        axs[i].spines.top.set_visible(False)

    for ax in axs[-4:]:
        ax.set_xlabel(r'$t$ (s)')

    for ax in axs[0::4]:
        ax.set_yticks([-120, 0])


    axs[4].set_ylabel(r'$V$ (mV)')

    fig.savefig(os.path.join(output_dir, "voltage_plots.pdf"))


if __name__ == "__main__":
    main()
