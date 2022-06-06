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
import string
from matplotlib import gridspec


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

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('data_directory', help='directory where data is stored')
    parser.add_argument('--experiment_name', '-e', default='newtonrun4')
    parser.add_argument('--wells', '-w', type=str, nargs='+')
    parser.add_argument('--protocols', type=str, nargs='+')
    parser.add_argument('--output', type=str)
    parser.add_argument('--figsize', '-f', nargs=2, type=float)
    parser.add_argument('--fontsize', type=int)
    parser.add_argument('--parameter_file')
    parser.add_argument('--model', default='Beattie')
    parser.add_argument('--nolegend', action='store_true')
    parser.add_argument('--dpi', '-d', default=500, type=int)
    parser.add_argument("-A", "--alphabet_labels", action='store_true')
    parser.add_argument('--ignore_protocols', '-i', nargs='+', default=[])

    global args
    args = parser.parse_args()

    output_dir = common.setup_output_directory(args.output, 'plot_mcmc_histograms')

    plt.rcParams.update({'font.size': args.fontsize})

    if args.wells is None:
        args.wells = [letter + f"{number:2d}" for letter in 'ABCDEFGHIJKLMNOP'
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

            if protocol not in args.protocols:
                continue

            chains = np.load(os.path.join(args.data_directory, f))

            print(well, protocol, pints.rhat(chains))

            # chain = chains.reshape((-1, model.get_no_parameters()))
            chain = chains[0, :, :]

            df = pd.DataFrame(chain, columns=model.get_parameter_labels())
            df['protocol'] = protocol

            dfs.append(df)

        if not dfs:
            continue

        df = pd.concat(dfs, ignore_index=True)

        if args.alphabet_labels:
            df['protocol'] = pd.Categorical(df['protocol'], categories=protocol_chrono_order)
            print(df['protocol'])
            relabel_dict = dict(zip(args.protocols,
                                    string.ascii_uppercase[:len(protocol_chrono_order)]))
            df = df.replace({
                'protocol': relabel_dict})

        for param in model.get_parameter_labels():
            sns.boxplot(data=df[[param, 'protocol']], x='protocol', y=param, linewidth=.5,
                        order=string.ascii_uppercase[:len(args.protocols)], showfliers=False)

            ax.set_xlim((max(0, ax.get_xlim()[0]), ax.get_xlim()[1]))
            fig.savefig(os.path.join(output_dir,
                                     f"{well}_{param}_mcmc_histograms.svg"), dpi=args.dpi)

            ax.cla()

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
                fig.tight_layout()
                ax.ticklabel_format(style='plain', axis='y', useOffset=False)

                ax.set_ylim(plot_lims)
                fig.savefig(os.path.join(output_dir,
                                         f"{well}_{param}_mcmc_histograms_{protocol}.svg"), dpi=args.dpi)

                ax.cla()


if __name__ == "__main__":
    main()
