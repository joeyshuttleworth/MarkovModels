#!/usr/bin/env python3

from MarkovModels.BeattieModel import BeattieModel
import argparse
import os
from MarkovModels import common
import numpy as np
import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import string

from matplotlib import gridspec
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def simulate_protocol(models, name, output_dir, params=None, limits=None):
    print(f"Plotting {name} to {output_dir}")

    # Now plot just the voltage and current

    if not args.separate_voltage:
        fig = plt.figure(figsize=args.figsize, dpi=250)
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        axs = [fig.add_subplot(ax) for ax in gs]

    else:
        fig = plt.figure(figsize=args.figsize, dpi=250)
        voltage_fig = plt.figure(figsize=args.figsize, dpi=250)
        axs = [fig.subplots(), voltage_fig.subplots()]

    axs[1].plot(models[0].times*1e-3, models[0].GetVoltage(), color='black', lw=args.linewidth)

    for model in models:
        times = model.times / 1000

        # if model.get_model_name() == 'KempModel':
        #     model.default_parameters =

        current = model.SimulateForwardModel()
        if args.normalise_current:
            current = current / np.max(current)

        if args.noise:
            observations = current + np.random.normal(0, args.noise, current.shape)
            axs[0].plot(times, observations, color='grey', alpha=0.5,
                        lw=args.linewidth)

        axs[0].plot(times, current, label=model.get_model_name(), lw=args.linewidth)

    bbox_inches = 0 if args.poster_plot else None

    if not args.poster_plot:
        # axs[0].set_title(name)
        # axs[1].set_xlabel('time /s')
        pass

    else:
        for ax in axs:
            for side in ['top', 'right', 'bottom', 'left']:
                ax.spines[side].set_visible(False)
                ax.get_xaxis().set_ticks([])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])

    if not args.no_legend:
        axs[0].legend()

    if limits:
        # axs[0].set_xlim([limits['tmin'], limits['tmax']])
        # axs[1].set_xlim([limits['tmin'], limits['tmax']])

        axs[0].set_ylim([-0.5, 1.5])
        axs[1].set_ylim([-120, 40])

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{name}.{args.output_format}"),
                bbox_inches=bbox_inches)

    if args.separate_voltage:
        voltage_fig.tight_layout()
        voltage_fig.savefig(os.path.join(output_dir, f"{name}_voltage.{args.output_format}"),
                            bbox_inches=bbox_inches)

    for ax in axs:
        ax.cla()
    plt.close(fig)

    print(f"{name} finished")


def main():
    parser = argparse.ArgumentParser(description="Plot output from different protocols")
    parser.add_argument("--protocols", "-p", default=[], type=str, nargs='+')
    parser.add_argument("--parameter_file", type=str, default=None)
    parser.add_argument("--models", "-m", nargs='+', default=['Beattie'])
    parser.add_argument("--figsize", nargs=2, type=int)
    parser.add_argument("-s", "--noise", default=0, type=float)
    parser.add_argument('--poster_plot', default=False, action='store_true')
    parser.add_argument('--normalise_current', default=False, action='store_true')
    parser.add_argument('--output_format', default='png')
    parser.add_argument('--separate_voltage', action='store_true')
    parser.add_argument('--share_limits', action='store_true')
    parser.add_argument('--fontsize', type=int, default=10)
    parser.add_argument('--no_legend', action='store_true')
    parser.add_argument('--removal_duration', type=float, default=.5)
    parser.add_argument('--linewidth', '-l', type=float)

    global args
    args = parser.parse_args()

    global params
    if args.parameter_file is not None:
        params = pd.read_csv(args.parameter_file).values.flatten()
    else:
        params = None

    output_dir = common.setup_output_directory(None, 'simulate_protocols')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_classes = [common.get_model_class(name) for name in args.models]

    protocols, times, voltages, currents = [], [], [], []
    for fname in os.listdir(common.get_protocol_directory()):
        fname, ext = os.path.splitext(fname)
        if fname[0] == '#':
            continue
        if fname in args.protocols or len(args.protocols) == 0:
            _times, _voltages, _currents = func(fname, ext, output_dir, model_classes)
        else:
            continue
        times.append(_times)
        voltages.append(_voltages)
        currents.append(_currents)
        protocols.append(fname)

    print(times)
    tmin, tmax = np.quantile(np.concatenate(times), [0.0, 1])
    vmin, vmax = np.quantile(np.concatenate(voltages), [0.01, .99])
    cmin, cmax = np.quantile(np.concatenate(currents), [0.05, .95])

    limits = {
        'tmin': tmin,
        'tmax': tmax,
        'vmin': vmin,
        'vmax': vmax,
        'cmin': cmin,
        'cmax': cmax,
    }

    for protocol in protocols:
        func(protocol, '.csv', output_dir, model_classes, limits=limits, plot=True)

    protocol_chrono_order = ['staircaseramp1',
                            'sis',
                            'rtovmaxdiff',
                            'rvotmaxdiff',
                            'spacefill10',
                            'spacefill19',
                            'spacefill26',
                            'longap',
                            'hhbrute3gstep',
                            'hhsobol3step',
                            'wangbrute3gstep',
                            'wangsobol3step',
                            'staircaseramp2']

    relabel_dict = dict(zip(protocol_chrono_order,
                            string.ascii_uppercase[:len(protocol_chrono_order)]))

    # Plot D-optimalities for each protocol
    dfs = []
    for protocol in protocol_chrono_order:
        alphabet_label = relabel_dict[protocol]

        for model_class_name in args.models:

            fig = plt.figure(figsize=args.figsize)
            ax = fig.subplots()
            model_class = common.get_model_class(model_class_name)
            # compute D-optimality
            voltage_func, t_start, t_end, t_step, protocol_desc = common.get_ramp_protocol_from_csv(protocol)

            times = np.linspace(t_start, t_end, int((t_end - t_start)/t_step))

            voltages = np.array([voltage_func(t) for t in times])

            model = model_class(voltage=voltage_func,
                                protocol_description=protocol_desc,
                                times=times)

            spike_times, spike_indices = common.detect_spikes(times, voltages,
                                                              window_size=0)

            indices = common.remove_indices(list(range(len(times))),
                                            [(spike,
                                              int(spike + args.removal_duration / t_step))
                                             for spike in spike_indices])

            print(indices)

            _, S1 = model.SimulateForwardModelSensitivities()

            S1 = S1 / model.get_default_parameters()[None, :]

            ax.plot(times/1e3, S1[:, :-1], label=[f"${param}$ sensitivty" for param in
                                                  model.parameter_labels[:-1]])
            fig.savefig(os.path.join(output_dir, f"sensitivities_plot_{model_class_name}_{protocol}.png"))
            plt.close(fig)

            H = S1[indices, :].T @ S1[indices, :]
            H_inv = np.linalg.inv(H)

            D_opti = np.linalg.det(H_inv)
            min_eig = min(abs(np.linalg.eigh(H)[0]))

            dfs.append(pd.DataFrame([(model_class_name, alphabet_label, D_opti, min_eig)],
                                    columns=('model', 'protocol', 'D optimality', 'min absolute eigenvalue')))
            print(dfs)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(output_dir, f"D_optimalities.csv"))

    fig = plt.figure(figsize=args.figsize)
    axs = fig.subplots(ncols=len(args.models), sharey=True)
    print(axs)

    for i, model_name in enumerate(args.models):
        sub_df = df[df.model == model_name]

        sns.stripplot(data=sub_df, hue='protocol', x='D optimality', ax=axs[i])

    fig.savefig(os.path.join(output_dir, "D_optimalities.pdf"))


def func(protocol_name, ext, output_dir, model_classes, limits=None, plot=False):
    if ext != ".csv":
        logging.warning(f"Using file with extension {ext}")

    protocol, t_start, t_end, t_step, protocol_description = common.get_ramp_protocol_from_csv(protocol_name)
    times = np.linspace(t_start, t_end, int((t_end - t_start) / t_step))

    models = [model_class(protocol, times,
                          protocol_description=protocol_description, parameters=params)

              for model_class in model_classes]
    times, voltages, currents = zip(*[(model.times, model.GetVoltage(),
                                      model.SimulateForwardModel())
                                      for model in models])
    times = np.concatenate(times)
    voltages = np.concatenate(voltages)
    currents = np.concatenate(currents)

    if plot:
        simulate_protocol(models, protocol_name, output_dir, limits=limits)

    return times, voltages, currents


if __name__ == "__main__":
    main()
