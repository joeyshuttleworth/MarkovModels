#!/usr/bin/env python3

import multiprocessing
import regex as re
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from MarkovModels import common
from MarkovModels.BeattieModel import BeattieModel
from MarkovModels.ClosedOpenModel import ClosedOpenModel
from MarkovModels.KempModel import KempModel
import argparse
import seaborn as sns
import os
import string

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
    parser.add_argument('datafile')
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
    parser.add_argument("-F", "--file_format", default='pdf')
    parser.add_argument('--true_param_file')

    global args
    args = parser.parse_args()

    if args.true_param_file:
        default_params = np.genfromtxt(args.true_param_file, delimiter=',')

    df = pd.read_csv(args.datafile)

    print(df['protocol'])
    df = df[~df['protocol'].isin(args.ignore_protocols)]

    output_dir = common.setup_output_directory(args.output_dir, "scatter_parameter_sets")

    parameter_labels = [lab for lab in df.columns[1:] if lab not in ['error',
                                                                     'protocol', 'fitting_protocol', 'validation_protocol',
                                                                     'E_rev', 'score', 'well']]

    df = df.replace({
        'protocol': relabel_dict})

    df = df.sort_values(by='protocol')

    fig = plt.figure(figsize=args.figsize)
    ax = fig.subplots()
    fig.tight_layout()

    for i in range(int(len(parameter_labels) / 2)):
        lab1 = parameter_labels[2*i]
        lab2 = parameter_labels[2*i + 1]

        g = sns.scatterplot(data=df, x=lab1, y=lab2, hue='protocol')
        ax = fig.gca()

        if i == 0:
            legend_params = [x.copy() for x in ax.get_legend_handles_labels()]

        g.get_legend().remove()

        fig.tight_layout()

        if args.true_param_file:
            print('plotting lines', lab1, lab2)
            ax.axvline(default_params[2*i], label=f"true {lab1}", linestyle='--')
            ax.axhline(default_params[2*i+1], label=f"true {lab2}", linestyle='--')

        fig.savefig(os.path.join(output_dir, f"{lab1}_{lab2}.{args.file_format}"))
        fig.clf()

    # lastly, plot conductance on its own
    g = sns.stripplot(data=df, y='protocol', hue='protocol', x=parameter_labels[-1])
    ax = plt.gca()
    ax.axvline(default_params[-1], label=f"true {parameter_labels[-1]}", linestyle='--')
    g.get_legend().remove()
    fig.savefig(os.path.join(output_dir, f"gkr.{args.file_format}"))

    fig.clf()

    print(legend_params)

    fig.legend(*legend_params, frameon=False, loc='center')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"legend.{args.file_format}"))

    fig.clf()

    sns.pairplot(df[parameter_labels + ['protocol']], hue='protocol', aspect=args.figsize[0]/args.figsize[1], height=args.figsize[1])
    plt.savefig(os.path.join(output_dir, f"pairplot.{args.file_format}"))


if __name__ == '__main__':
    main()
