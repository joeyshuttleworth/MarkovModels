#!/usr/bin/env python3

from MarkovModels.BeattieModel import BeattieModel
import argparse
import os
from MarkovModels import common
import numpy as np
import logging
import matplotlib.pyplot as plt
import pandas as pd

global params
params = np.array([2.07E-3, 7.17E-2, 3.44E-5, 6.18E-2, 4.18E-1, 2.58E-2,
                   4.75E-2, 2.51E-2, 3.33E-2])


def simulate_protocol(model, name, output_dir, params=None):
    if args.poster_plot:
        figsize = (6, 6)
        plt.axis('off')
        transparent = True
    else:
        figsize = (12, 9)
        transparent = False

    print(f"Plotting {name} to {output_dir}")

    # Now plot just the voltage and current
    fig = plt.figure(figsize=figsize, dpi=1000)
    axs = fig.subplots(2)

    current = model.SimulateForwardModel()
    observations = current + np.random.normal(0, args.noise, current.shape)

    axs[0].plot(model.times, model.GetVoltage(), color='black')
    axs[1].plot(model.times, observations, color='grey', alpha=0.5)
    axs[1].plot(model.times, current, color='green')

    if not args.poster_plot:
        axs[0].set_title(name)
        axs[1].set_xlabel('time /ms')
        axs[1].set_ylabel('current /nA')
        axs[0].set_ylabel('voltage / mV')

    bbox_inches = 0 if args.poster_plot else None

    for ax in axs:
        for side in ['top', 'right', 'bottom', 'left']:
            ax.spines[side].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    fig.savefig(os.path.join(output_dir, f"{name}_protocol"),
                bbox_inches=bbox_inches, transparent=transparent)

    for ax in axs:
        ax.cla()
    plt.close(fig)

    print(f"{name} finished")


def main():
    parser = argparse.ArgumentParser(description="Plot output from different protocols")
    parser.add_argument("--protocols", "-p", default=[], type=str, nargs='+')
    parser.add_argument("--parameter_file", type=str, default=None)
    parser.add_argument("-s", "--noise", default=0, type=float)
    parser.add_argument('--poster_plot', default=False, action='store_true')

    global args
    args = parser.parse_args()

    if args.parameter_file is not None:
        params = pd.read_csv(args.parameter_file).values.flatten()

    output_dir = common.setup_output_directory(None, 'simulate_protocols')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for fname in os.listdir(common.get_protocol_directory()):
        fname, ext = os.path.splitext(fname)
        if fname[0] == '#':
            continue
        if fname in args.protocols or len(args.protocols) == 0:
            func(fname, ext, output_dir)


def func(protocol_name, ext, output_dir):
    if ext != ".csv":
        logging.warning(f"Using file with extension {ext}")

    protocol, t_start, t_end, t_step, protocol_description = common.get_ramp_protocol_from_csv(protocol_name)
    times = np.linspace(t_start, t_end, int((t_end - t_start) / t_step))
    model = BeattieModel(protocol, times,
                         protocol_description=protocol_description, parameters=params)

    simulate_protocol(model, protocol_name, output_dir)


if __name__ == "__main__":
    main()
