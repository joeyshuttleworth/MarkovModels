#!/usr/bin/env python3

from MarkovModels import common
import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import pints
import pints.plot
import uuid
import argparse
import matplotlib.pyplot as plt

def main():
    description = ""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("input_file", help="CSV file listing model errors for each cell and protocol")
    parser.add_argument("--output_dir", "-o", help="Directory to output plots to. By default a new directory will be generated", default=None)

    args = parser.parse_args()
    df = pd.read_csv(args.input_file)

    fig = plt.figure(figsize=(14, 10))

    output_dir = common.setup_output_directory(args.output_dir)

    # Iterate over wells
    for well in df['well'].unique():
        ax = fig.subplots()
        sub_df = df[df.well == well].pivot(index='fitting_protocol', columns='validation_protocol',
                                           values='RMSE')
        sns.heatmap(sub_df, ax=ax, cbar_kws={'label': 'RMSE'}, vmin=0)

        ax.set_title(f"well {well}")

        fig.savefig(os.path.join(output_dir, f"{well}_fit_predict_heatmap.png"))
        fig.clf()

if __name__ == "__main__":
    main()
