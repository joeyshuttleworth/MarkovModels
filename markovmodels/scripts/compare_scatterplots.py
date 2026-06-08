#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

import markovmodels

from markovmodels.fitting import infer_reversal_potential, get_best_params
from markovmodels.utilities import setup_output_directory
from markovmodels.model_generation import make_model_of_class


def create_axes(fig, no_rows):
    return [fig.subplots(no_rows)]


def main():
    description = ""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("input_files", nargs=2, help="CSV file listing model errors for each cell and protocol")
    parser.add_argument("--output_dir", "-o", help="Directory to output plots to.\
    By default a new directory will be generated", default=None)
    parser.add_argument("--figsize", default=(12, 10), nargs=2, type=float)
    parser.add_argument('--experiment_name', default='newtonrun4', type=str)
    parser.add_argument('--removal_duration', '-r', default=5, type=float)
    parser.add_argument('--reversal', type=float, default=np.nan)
    parser.add_argument('--solver_type', default='hybrid')
    parser.add_argument('--ignore_protocols', nargs='+', default=['longap'])
    parser.add_argument('--wells', '-w', nargs='+')
    parser.add_argument('--protocols', nargs='+')
    parser.add_argument('--sweeps', nargs='+')
    parser.add_argument('--legend', action='store_true')
    parser.add_argument('--adjust_kinetics', action='store_true')
    parser.add_argument('--labels', nargs=2, default=['1', '2'])
    parser.add_argument('--hue', default='label')
    parser.add_argument('--model', default='model3')

    global args
    args = parser.parse_args()

    dfs = []
    for label, input_file in zip(args.labels, args.input_files):
        df = pd.read_csv(input_file)
        df = get_best_params(df)
        df['label'] = label
        df['score'] = df['score'].astype(np.float64)
        df = df.sort_values('score')

        df = df.drop_duplicates(subset=['well', 'protocol', 'sweep'],
                                keep='first')

        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    df[~df.protocol.isin(args.ignore_protocols)]

    if args.sweeps:
        df = df[df.sweep.astype(str).isin(args.sweeps)]

    df[~df.protocol.isin(args.ignore_protocols)]

    if args.wells:
        df = df[df.well.isin(args.wells)]

    if args.protocols:
        df = df[df.protocol.isin(args.protocols)]

    print(df)

    print(list(df.well.unique()))

    param_labels = make_model_of_class(args.model).get_parameter_labels()
    df[param_labels] = df[param_labels].astype(np.float64)

    global output_dir
    output_dir = setup_output_directory(args.output_dir, 'compare_scatterplots')

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    ax = fig.subplots()

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    axes = create_axes(fig, 5)

    for i in range(4):
        ax1 = axes[0][i]
        ax2 = axes[0][i]
        sns.scatterplot(df, x=param_labels[i*2], y=param_labels[i*2+1],
                        hue=args.hue, legend=args.legend,
                        ax=ax1, alpha=.5)

        xmin = min(ax1.get_xlim()[0], ax2.get_xlim()[0])
        xmax = max(ax1.get_xlim()[1], ax2.get_xlim()[1])

        ax1.set_xlim((xmin, xmax))
        ax2.set_xlim((xmin, xmax))

        ymin = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
        ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])

        ax1.set_ylim((ymin, ymax))
        ax2.set_ylim((ymin, ymax))

    ax1 = axes[0][-1]
    sns.stripplot(df, x=param_labels[-1], hue=args.hue,
                  legend=args.legend, ax=ax1)

    xmin = min(ax1.get_xlim()[0], ax2.get_xlim()[0])
    xmax = max(ax1.get_xlim()[1], ax2.get_xlim()[1])
    ymin = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    xmax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])

    fig.savefig(os.path.join(output_dir, "scatterplot_figure"))

    for ax in axes[0]:
        ax.cla()

    for i in range(2):
        ax = axes[0][i]
        sns.scatterplot(df, x=param_labels[i*2], y=param_labels[i*2+2],
                        alpha=.5, hue=args.hue, legend=args.legend, ax=ax)

    for i in range(2):
        ax = axes[0][i+2]
        sns.scatterplot(df, x=param_labels[i*4+1], y=param_labels[i*4+3],
                        alpha=.5, hue=args.hue, legend=args.legend, ax=ax)

    for i in range(4):
        ax1 = axes[0][i]
        ax2 = ax1

        xmin = min(ax1.get_xlim()[0], ax2.get_xlim()[0])
        xmax = max(ax1.get_xlim()[1], ax2.get_xlim()[1])

        ax1.set_xlim((xmin, xmax))
        ax2.set_xlim((xmin, xmax))

        ymin = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
        ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])

        ax1.set_ylim((ymin, ymax))
        ax2.set_ylim((ymin, ymax))

    ax1 = axes[0][-1]
    sns.stripplot(df, x=param_labels[-1], hue=args.hue,
                  legend=args.legend, ax=ax1)

    fig.savefig(os.path.join(output_dir, "scatterplot_figure2"))


if __name__ == "__main__":
    main()
