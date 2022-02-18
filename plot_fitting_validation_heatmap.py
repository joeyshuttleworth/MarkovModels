#!/usr/bin/env python3

from MarkovModels import common
import os
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import matplotlib.pyplot as plt

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
    description = ""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("input_file", help="CSV file listing model errors for each cell and protocol")
    parser.add_argument("--output_dir", "-o", help="Directory to output plots to.\
    By default a new directory will be generated", default=None)
    parser.add_argument("--normalise_diagonal", action="store_true")

    parser.add_argument("--vmax", "-m", default=None, type=float)

    args = parser.parse_args()
    df = pd.read_csv(args.input_file)
    df = df.drop_duplicates(subset=['well', 'fitting_protocol', 'validation_protocol'], keep='first')

    # Set order of protocols
    df['fitting_protocol'] = pd.Categorical(df['fitting_protocol'], protocol_chrono_order)
    df['validation_protocol'] = pd.Categorical(df['validation_protocol'], protocol_chrono_order)


    fig = plt.figure(figsize=(14, 10))

    output_dir = common.setup_output_directory(args.output_dir)
    protocol_list = df['fitting_protocol'].unique()

    # Iterate over wells for heatmap
    for well in df['well'].unique():
        ax = fig.subplots()
        sub_df = df[df.well == well].copy()

        if args.normalise_diagonal:
            index_df = sub_df.set_index(['fitting_protocol', 'validation_protocol'])
            diagonals = {protocol: index_df.loc[(protocol, protocol), 'RMSE'] for protocol in protocol_list}
            sub_df['log normalised RMSE'] = [
                np.log(row['RMSE'] /
                       diagonals[row['validation_protocol']]) for _, row in sub_df.iterrows()]
            value_col = 'log normalised RMSE'
        else:
            df['log RMSE'] = np.log(df['RMSE'])
            value_col = 'log RMSE'

        pivot_df = sub_df.pivot(index='fitting_protocol', columns='validation_protocol',
                                values=value_col)

        # pivot_df = pivot_df[np.isfinite(pivot_df)]

        cmap = sns.cm.rocket_r

        vmax = min(args.vmax, np.max(pivot_df.values))
        sns.heatmap(pivot_df, ax=ax, cbar_kws={'label': value_col}, vmin=None, vmax=vmax, cmap=cmap)

        ax.set_title(f"well {well}")
        ax.set_ylabel("Fitting protocol")
        ax.set_xlabel("Validation protocol")

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{well}_fit_predict_heatmap.png"))
        fig.clf()


if __name__ == "__main__":
    main()
