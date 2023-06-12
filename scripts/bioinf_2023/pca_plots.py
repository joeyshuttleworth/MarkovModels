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

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 11})
rc('text', usetex=True)
rc('figure', dpi=600)


def create_axes(fig):
    axs = fig.subplots(2, 2)
    return [axs[0, 0], axs[0, 1], axs[1, 0]], axs[1, 1]


def main():
    description = ""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("fitting_results", nargs='+')
    parser.add_argument("--output_dir", "-o", help="Directory to output plots to.\
    By default a new directory will be generated", default=None)
    parser.add_argument("--normalise_diagonal", action="store_true")
    parser.add_argument("--vmax", "-m", default=None, type=float)
    parser.add_argument("--share_limits", action='store_true')
    parser.add_argument("--model_classes", default=['Beattie'], nargs='+')
    parser.add_argument("--figsize", default=(15, 12), nargs=2, type=float)
    parser.add_argument('--experiment_name', default='newtonrun4', type=str)
    parser.add_argument('--removal_duration', '-r', default=5, type=float)
    parser.add_argument('--reversal', type=float, default=np.nan)
    parser.add_argument('--solver_type', default='hybrid')
    parser.add_argument('--wells', '-w', nargs='+')
    parser.add_argument('--protocols', nargs='+')
    parser.add_argument('--adjust_kinetics', action='store_true')
    parser.add_argument('--hue', default='well')

    global args
    args = parser.parse_args()

    global output_dir
    output_dir = common.setup_output_directory(args.output_dir, 'scatterplots')

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)

    scatter_axs, stripplot_ax = create_axes(fig)

    gkr_dfs = []
    for i, (model_class_name, results_fname) in enumerate(zip(args.model_classes,
                                                              args.fitting_results)):
        model_class = common.get_model_class(model_class_name)

        df = pd.read_csv(results_fname)

        if 'fitting_protocol' in df.columns:
            df['protocol']

        if 'fitting_sweep' in df.columns:
            df['sweep']

        param_labels = model_class().get_parameter_labels()
        df[r'$G_\textrm{Kr}$'] = df[param_labels[model_class().GKr_index]]

        df = df.drop_duplicates(subset=['well', 'protocol', 'sweep'],
                                keep='first')
        df = df[df.protocol != 'longap']

        if args.wells:
            df = df[df.well.isin(args.wells)]

        if args.protocols:
            df = df[df.protocol.isin(args.protocols)]

        df[param_labels] = df[param_labels].astype(np.float64)

        df = df.apply(replace_staircases, axis=1)

        pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=5))])
        res = pipeline.fit_transform(df[param_labels].values)

        df['pca1'] = res[:, 0]
        df['pca2'] = res[:, 1]

        if args.hue == 'well':
            sub_df = df[df.well == 'D09'].copy()
            sub_df['hue'] = (sub_df['protocol'] == 'staircaseramp1').copy()
            sns.scatterplot(data=sub_df, x='pca1', y='pca2', legend=False,
                            hue='hue', ax=scatter_axs[i])

            scatter_axs[i].scatter(data=df[df.well != 'D09'], x='pca1',
                                   y='pca2', color='grey', alpha=0, s=0)
        else:
            sns.scatterplot(data=df, x='pca1', y='pca2', legend=False,
                            hue='protocol')

        plt.close(fig)

        gkr_sub_df = df[[r'$G_\textrm{Kr}$', 'well', 'protocol']].copy()
        gkr_sub_df['model'] = model_class_name

        gkr_dfs.append(gkr_sub_df)

    gkr_df = pd.concat(gkr_dfs, ignore_index=True)
    if args.hue == 'well':
        sub_df = gkr_df[gkr_df.well == 'D09'].copy()
        sub_df['hue'] = (sub_df.protocol == 'staircaseramp1')
        sns.stripplot(sub_df, x='model',
                      y=r'$G_\textrm{Kr}$', hue='hue',
                      legend=False, ax=stripplot_ax)

    else:
        sns.stripplot(gkr_df, y='model', x=r'$G_\textrm{Kr}$', hue='protocol',
                      legend=False, ax=stripplot_ax)

    fig.savefig(os.path.join(output_dir, "pca_plots"))


def replace_staircases(row):
    if row['protocol'] == 'staircaseramp2':
        row['protocol'] = 'staircaseramp1'
    return row

if __name__ == "__main__":
    main()
