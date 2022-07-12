#!/usr/bin/env python3

from MarkovModels import common
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import regex as re
import pints
import pints.plot
import string

from matplotlib import gridspec
from matplotlib import rc


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

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


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('data_directory', help='directory where data is stored')
    parser.add_argument('--experiment_name', '-e', default='newtonrun4')
    parser.add_argument('--wells', '-w', type=str, nargs='+')
    parser.add_argument('--protocols', type=str, nargs='+')
    parser.add_argument('--true_param_file')
    parser.add_argument('--output', type=str)
    parser.add_argument('--figsize', '-f', nargs=2, type=float)
    parser.add_argument('--fontsize', type=int)
    parser.add_argument('--parameter_file')
    parser.add_argument('--model', default='Beattie')
    parser.add_argument('--nolegend', action='store_true')
    parser.add_argument('--dpi', '-d', default=500, type=int)
    parser.add_argument("-A", "--alphabet_labels", action='store_true')
    parser.add_argument('--ignore_protocols', '-i', nargs='+', default=[])
    parser.add_argument('--linewidth', '-l', type=float)
    parser.add_argument('--file_format', default='png')
    parser.add_argument('--burn_in', '-b', default=1000, type=int)

    global args
    args = parser.parse_args()

    output_dir = common.setup_output_directory(args.output, 'plot_mcmc_histograms')

    if args.true_param_file:
        default_params = np.genfromtxt(args.true_param_file, delimiter=',')

    if args.fontsize:
        plt.rcParams.update({'font.size': args.fontsize})

    if args.wells is None:
        args.wells = [letter + f"{number}" if number >= 10 else letter + f"0{number}" for letter in 'ABCDEFGHIJKLMNOP'
                      for number in range(1, 25)]

    if args.protocols is None:
        args.protocols = common.get_protocol_list()

    model_class = common.get_model_class(args.model)
    model_name  = model_class().get_model_name()

    if args.parameter_file and args.model:
        params_df = pd.read_csv(args.parameter_file)

    else:
        params_df = None

    if params_df is not None:
        args.protocols = [protocol for protocol in args.protocols if protocol
                          in params_df['protocol'].unique()]
        args.wells = [well for well in args.wells if well in params_df['well'].unique()]

    fig = plt.figure(figsize=args.figsize)
    ax = fig.subplots()

    cm = sns.husl_palette(len(args.protocols))

    if len(args.protocols) == 1:
        lw = 1
    else:
        lw = .5

    if args.protocols is None:
        args.protocols = [p for p in args.protocols if p not in
                          args.ignore_protocols]

    model = model_class()

    for well in args.wells:
        rstring = re.compile(f"^mcmc_{model_name}_{well}_([a-z|0-9]*).npy$")
        dfs = []

        for f in filter(rstring.match, os.listdir(args.data_directory)):
            protocol = rstring.search(f).group(1)

            print(protocol)

            if protocol not in args.protocols:
                continue

            chains = np.load(os.path.join(args.data_directory, f))

            trace_fig, trace_ax = pints.plot.trace(chains)
            trace_fig.savefig(os.path.join(output_dir, f"trace_{well}_{protocol}.png"))
            plt.close(trace_fig)

            print(well, protocol, pints.rhat(chains))

            rhats = pints.rhat(chains)
            print(rhats)
            print(chains.shape)
            chain = chains.reshape([-1, len(model.get_default_parameters())])
            print(chain.shape)

            chain = chain[args.burn_in:, :]

            if protocol:
                df = pd.DataFrame(chain, columns=model.get_parameter_labels())
                df['protocol'] = protocol

                dfs.append(df)

        if not dfs:
            continue

        df = pd.concat(dfs, ignore_index=True)

        print(df)
        if args.alphabet_labels:
            df = df[df.protocol.isin(relabel_dict)]
            df['protocol'] = pd.Categorical(df['protocol'], categories=protocol_chrono_order)
            print(df['protocol'])
            df = df.replace({
                'protocol': relabel_dict})

            df = df.sort_values('protocol')

        for i, param in enumerate(model.get_parameter_labels()):
            print('plotting hist')
            q1, q2 = df[param].quantile((0.1, 0.9))
            # sub_df = df[(df[param] > q1) & (df[param] < q2)]
            sub_df = df
            print(param)
            sns.histplot(data=sub_df[[param, 'protocol']], x=param, hue='protocol', bins=500)

            if args.true_param_file:
                plt.gca().axvline(default_params[i], label=f"true {param}", linestyle='--')

            plt.savefig(os.path.join(output_dir, f"{well}_{param}_mcmc_histograms.{args.file_format}"))

            fig.legend()

            plt.clf()

        for i, param in enumerate(model.get_parameter_labels()):
            sns.boxplot(data=df[[param, 'protocol']], x='protocol', y=param, linewidth=.5,
                        order=string.ascii_uppercase[:len(args.protocols)], showfliers=False)

            ax.set_xlim((max(0, ax.get_xlim()[0]), ax.get_xlim()[1]))

            if args.true_param_file:
                plt.gca().axhline(default_params[i], label=f"true {param}", linestyle='--')

            fig.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir,
                                     f"{well}_{param}_mcmc_boxplot.{args.file_format}"), dpi=args.dpi)

            fig.clf()

        # Plot individually
        for protocol in df.protocol.unique():
            sub_df = df[df.protocol == protocol].copy()
            sub_df['protocol'] = pd.Categorical(sub_df['protocol'], categories=(protocol,))

            for param in ['Gkr']:
                vals = sub_df[param].values

                IQR = np.quantile(vals, .75) - np.quantile(vals, .25)
                whis = 3

                plot_lims = [vals.mean() - 2*vals.std(), vals.mean() + 2*vals.std()]

                # vals = vals[(vals > plot_lims[0])
                #             & (vals < plot_lims[1])]

                sns.violinplot(y=vals, linewidth=.5, x=np.repeat(protocol, len(vals)))

                if args.true_param_file:
                    plt.gca().axhline(default_params[-1], label=f"true {param}", linestyle='--')

                fig.tight_layout()
                ax.ticklabel_format(style='plain', axis='y', useOffset=False)

                ax.set_ylim(plot_lims)
                fig.legend()
                fig.savefig(os.path.join(output_dir,
                                         f"{well}_{param}_mcmc_violin_{protocol}.{args.file_format}"), dpi=args.dpi)

                plt.clf()


if __name__ == "__main__":
    main()
