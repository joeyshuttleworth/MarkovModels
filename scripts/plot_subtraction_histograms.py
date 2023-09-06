#!/usr/bin/env python3

from markovmodels import common
from markovmodels.BeattieModel import BeattieModel
import os
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
import regex as re


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("input_filepaths", help="", nargs='+')
    parser.add_argument("--experiment_names", help="", nargs='+')
    parser.add_argument("-o", "--output", help="Directory to output to", default=None)

    args = parser.parse_args()

    experiment_names = args.experiment_names if args.experiment_names else ['experiment %s' % (i + 1)
                                                                            for i in range(len(args.input_filepaths))]

    output_dir = common.setup_output_directory(args.output,
                                               "plot_subtraction_histograms")

    subtraction_dfs = [pd.read_csv(filepath) for filepath in args.input_filepaths]

    protocols = ['staircaseramp1', 'staircaseramp2']

    variables_to_plot = ['fitted_E_rev', 'R_leftover',
                         'pre-drug leak conductance', 'post-drug leak conductance',
                         'pre-drug leak reversal',
                         'post-drug leak reversal']

    fig = plt.figure(figsize=(12, 9))
    ax = fig.subplots()
    for var in variables_to_plot:
        for protocol in protocols:
            views = []
            for df, experiment_name in zip(subtraction_dfs, experiment_names):
                view = df[df.protocol == protocol].copy()
                view['experiment_name'] = experiment_name
                views.append(view)

            df = pd.concat(views, ignore_index=True)

            sns.histplot(data=df, x=var, ax=ax, hue='experiment_name',
                         label=experiment_name, stat='probability', common_norm=False)

            fig.savefig(os.path.join(output_dir, f"hist_{protocol}_{var}.png"))
            ax.cla()


if __name__ == "__main__":
    main()
