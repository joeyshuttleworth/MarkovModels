#!/usr/bin/env python3

from MarkovModels import common
import os
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
import string
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

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
    description = ""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("input_file", help="CSV file listing model errors for each cell and protocol")
    parser.add_argument("--output_dir", "-o", help="Directory to output plots to.\
    By default a new directory will be generated", default=None)
    parser.add_argument("--normalise_diagonal", action="store_true")
    parser.add_argument("-s", "--sort", action='store_true')
    parser.add_argument("-l", "--log_scale", action='store_true')
    parser.add_argument("-A", "--alphabet_labels", action='store_true')
    parser.add_argument("-i", "--ignore_protocols", nargs='+', default=[])
    parser.add_argument("--figsize", "-f", nargs=2, type=int)
    parser.add_argument("--fontsize", type=int, default=10)
    parser.add_argument("--dpi", "-d", type=int, default=600)
    parser.add_argument("--vmax", "-m", default=None, type=float)
    parser.add_argument("--vmin", default=None, type=float)
    parser.add_argument("--share_colourbar", action='store_true')
    parser.add_argument("--file_format", default="svg")
    parser.add_argument("--separate_cbar", action='store_true')
    parser.add_argument("--remove_cbar", action='store_true')

    args = parser.parse_args()

    rc('font', size=args.fontsize)
    df = pd.read_csv(args.input_file)
    df = df.drop_duplicates(subset=['well', 'fitting_protocol', 'validation_protocol'], keep='first')

    df = df[~df.fitting_protocol.isin(args.ignore_protocols)]
    df = df[~df.validation_protocol.isin(args.ignore_protocols)]

    if args.sort:
        protocols = df['fitting_protocol'].unique()

        # Rank protocols
        def score(protocol):
            return df[df.fitting_protocol == protocol]['RMSE'].sum()
        scores = [score(protocol) for protocol in protocols]

        order = protocols[np.argsort(scores)]
        score_df = pd.DataFrame(np.column_stack((protocols, scores)), columns=('protocol', 'score'))
        score_df['protocol'] = pd.Categorical(score_df['protocol'], order)

    else:
        order = protocol_chrono_order
        df = df[df.fitting_protocol.isin(protocol_chrono_order) &
                df.validation_protocol.isin(protocol_chrono_order)]

    print(df['fitting_protocol'])
    # Change order of protocols
    df['fitting_protocol'] = pd.Categorical(df['fitting_protocol'], categories=order)
    df['validation_protocol'] = pd.Categorical(df['validation_protocol'], categories=order)

    if args.alphabet_labels:
        df = df.replace({
            'validation_protocol': relabel_dict,
            'fitting_protocol': relabel_dict})

    fig = plt.figure(figsize=args.figsize)

    output_dir = common.setup_output_directory(args.output_dir, 'fitting_validation_heatmaps')
    protocol_list = df['fitting_protocol'].unique()

    if args.normalise_diagonal:
        assert False

    # if args.wells:
    #         df = df[df.well.isin(args.wells)]
    # if args.protocols:
    #     df = df[df.protocols.isin(protocols)]

    cbar_min = df['RMSE'].min()
    cbar_max = df['RMSE'].max()

    if args.log_scale:
        cbar_min = np.log10(cbar_min)
        cbar_max = np.log10(cbar_max)


    # Iterate over wells for heatmap
    for well in df['well'].unique():
        ax = fig.subplots()

        sub_df = df[df.well == well].copy()
        print(sub_df)

        if args.normalise_diagonal:
            index_df = sub_df.set_index(['fitting_protocol', 'validation_protocol'])
            diagonals = {protocol: index_df.loc[(protocol, protocol), 'RMSE'] for protocol in protocol_list}
            sub_df['normalised RMSE'] = [
                row['RMSE'] /
                diagonals[row['validation_protocol']] for _, row in sub_df.iterrows()]
            sub_df['log$_{10}$ normalised RMSE'] = np.log10(sub_df['normalised RMSE'])

            if args.log_scale:
                value_col = 'log$_{10}$ normalised RMSE'
            else:
                value_col = 'normalised RMSE'

        elif args.log_scale:
            sub_df['log RMSE'] = np.log10(sub_df['RMSE'])
            value_col = 'log RMSE'

        else:
            value_col = 'RMSE'

        pivot_df = sub_df.pivot(index='fitting_protocol', columns='validation_protocol',
                                values=value_col)

        for protocol in sub_df.fitting_protocol.unique():
            pivot_df.loc[protocol][protocol] = pivot_df.loc[protocol].values.min()

        pivot_df = pivot_df[np.isfinite(pivot_df)]

        cmap = sns.cm.mako_r
        # vals = pivot_df['values'].values
        # norm = plt.Normalize(vals.min(), vals.max())

        # if not args.share_colourbar:
        # vmax = min(args.vmax, np.max(pivot_df.values)) if args.vmax else None
        # vmin = None
        # else:
        #     vmax = cbar_max
        #     vmin = cbar_min

        hm = sns.heatmap(pivot_df, ax=ax, cbar_kws={'label': value_col}, vmin=args.vmin,
                         vmax=args.vmax, cmap=cmap, square=True, cbar=not args.remove_cbar)
        hm.set_yticklabels(hm.get_yticklabels(), rotation=0)
        hm.set_facecolor('black')

        # if not args.alphabet_labels:
        #     hm.set_xticks([])

        # ax.set_title(f"{well}")
        ax.set_ylabel("Fitting protocol")
        ax.set_xlabel("Validation protocol")

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{well}_fit_predict_heatmap.{args.file_format}"), dpi=args.dpi)
        fig.clf()


if __name__ == "__main__":
    main()
