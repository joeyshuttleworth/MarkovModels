#!/usr/bin/env python3

from MarkovModels import common
import os
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import matplotlib.pyplot as plt


def main():
    description = ""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("input_file", help="CSV file listing model errors for each cell and protocol")
    parser.add_argument("--output_dir", "-o", help="Directory to output plots to.\
    By default a new directory will be generated", default=None)
    parser.add_argument("--normalise_diagonal", action="store_true")

    parser.add_argument("--vmax", "-m", default=None)

    args = parser.parse_args()
    df = pd.read_csv(args.input_file)

    fig = plt.figure(figsize=(14, 10))

    output_dir = common.setup_output_directory(args.output_dir)

    protocol_list = df['fitting_protocol'].unqiue()

    # Iterate over wells for heatmap
    for well in df['well'].unique():
        ax = fig.subplots()
        df['log RMSE'] = np.log(df['RMSE'])
        sub_df = df[df.well == well]
        print(sub_df)
        pivot_df = sub_df.pivot(index='fitting_protocol', columns='validation_protocol',
                                values='log RMSE')
        index_df = df.set_index(('fitting_protocol', 'validation_protocol'))

        if args.normalise_diagonal:
            sub_df = sub_df.set_index(('fitting_protocol', 'validation_protocol'))
            diagonals = {protocol : sub_df[protocol, protocol]['RMSE'] for protocol in protocol_list}

        pivot_df['normalised log RMSE'] = np.log(pivot_df['RMSE'] /
                                                 diagonals[diagonals[pivot_df['fitting_protocol']]])

        pivot_df = pivot_df[np.isfinite(pivot_df)]

        cmap = sns.cm.rocket_r
        sns.heatmap(pivot_df, ax=ax, cbar_kws={'label': 'log RMSE'}, vmin=None, vmax=args.vmax, cmap=cmap)

        ax.set_title(f"well {well}")
        ax.set_ylabel("Fitting protocol")
        ax.set_xlabel("Validation protocol")

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{well}_fit_predict_heatmap.png"))
        fig.clf()


if __name__ == "__main__":
    main()
