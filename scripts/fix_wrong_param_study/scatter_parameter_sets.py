#!/usr/bin/env python3

import multiprocessing
import regex as re
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from markovmodels import common
from markovmodels.BeattieModel import BeattieModel
from markovmodels.ClosedOpenModel import ClosedOpenModel
from markovmodels.KempModel import KempModel
import argparse
import seaborn as sns
import os
import string
import re
import itertools

from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

_markers = itertools.cycle(['o', 'v', '^', '<', '>', 's', 'p', '*', 'h',
                            'H', 'D', 'd', 'P', 'X'])

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
    parser.add_argument('datafile')
    parser.add_argument('--model_class', default='BeattieModel')
    parser.add_argument('--repeats', type=int, default=16)
    parser.add_argument('--wells', '-w', type=str, default=[], nargs='+')
    parser.add_argument('--removal_duration', '-r', default=5, type=float)
    parser.add_argument('--experiment_name', default='newtonrun4', type=str)
    parser.add_argument('--no_chains', '-N', default=0, help='mcmc chains to run', type=int)
    parser.add_argument('--chain_length', '-l', default=500, help='mcmc chains to run', type=int)
    parser.add_argument('--figsize', '-f', type=int, nargs=2)
    parser.add_argument('--use_parameter_file')
    parser.add_argument('-i', '--ignore_protocols', nargs='+', default=[])
    parser.add_argument('-o', '--output_dir')
    parser.add_argument("-A", "--alphabet_labels", action='store_true')
    parser.add_argument("-F", "--file_format", default='png')
    parser.add_argument("-m", "--model", default='Beattie')
    parser.add_argument('--fix_param')
    parser.add_argument('--same_plot_lims', action='store_true', default=False)

    global args
    args = parser.parse_args()

    output_dir = common.setup_output_directory(args.output_dir,
                                               "scatter_parameter_sets")

    model_class = common.get_model_class(args.model_class)

    if args.use_parameter_file:
        true_parameters = pd.read_csv(args.use_parameter_file,
                                      header=None).values[0, :]
    else:
        true_parameters = model_class().get_default_parameters()

    if not args.fix_param:
        args.fix_param = model_class().get_parameter_labels()[-1]
        print(f"using {args.fix_param} as fixed parameter")

    df = pd.read_csv(args.datafile).sort_values(args.fix_param)

    for param_label in model_class().get_parameter_labels():
        df[param_label] = df[param_label].astype(np.float64)

    parameter_labels = model_class().get_parameter_labels()
    fig = plt.figure(figsize=args.figsize)
    ax = fig.subplots()

    df = df[~df.protocol.isin(args.ignore_protocols)]
    print(df.protocol.unique())

    markers = [next(_markers) for p in df.protocol.unique()]

    for i in range(int(len(parameter_labels) / 2)):
        lab1 = parameter_labels[2*i]
        lab2 = parameter_labels[2*i + 1]

        if args.same_plot_lims:
            xlims = np.quantile(df[lab1], [0, 1])
            ylims = np.quantile(df[lab2], [0, 1])
            print(xlims, ylims)

        for j, val in enumerate(df[args.fix_param].unique()):
            sub_df = df[df[args.fix_param] == val]

            # sub_df = sub_df.replace({
            #     'protocol': relabel_dict})

            sub_df = sub_df.sort_values(by='protocol')

            g = sns.scatterplot(data=sub_df, x=lab1, y=lab2, hue='protocol',
                                style='protocol', legend=True, markers=markers,
                                linewidth=0.1)
            ax = fig.gca()

            if i == 0:
                legend_params = [x.copy() for x in ax.get_legend_handles_labels()]

            # g.get_legend().remove()

            val1 = true_parameters[2*i]
            val2 = true_parameters[2*i + 1]

            ax.axvline(val1, label=f"true {lab1}", linestyle='dotted')
            ax.axhline(val2, label=f"true {lab2}", linestyle='dotted')

            if args.same_plot_lims:
                ax.set_xlim(*xlims)
                ax.set_ylim(*ylims)

            ax.set_title(f"{args.fix_param} = {val:.4f}")
            fig.savefig(os.path.join(output_dir, f"{lab1}_{lab2}_{j}.{args.file_format}"))
            fig.clf()

        # lastly, plot conductance on its own
        g = sns.stripplot(data=sub_df, y='protocol', hue='protocol',
                          x=parameter_labels[-1])
        ax = plt.gca()
        ax.legend()

        ax.axvline(true_parameters[-1], label=f"true {parameter_labels[-1]}", linestyle='--')
        g.get_legend().remove()
        fig.savefig(os.path.join(output_dir, f"gkr_{val:.4f}.{args.file_format}"))

        fig.clf()

        fig.legend(*legend_params, frameon=False, loc='center')
        fig.savefig(os.path.join(output_dir, f"legend.{args.file_format}"))

        fig.clf()


if __name__ == '__main__':
    main()
