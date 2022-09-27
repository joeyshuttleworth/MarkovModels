#!/usr/bin/env python3

from MarkovModels import common

from MarkovModels import common
from MarkovModels.BeattieModel import BeattieModel
import os
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
import regex as re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mcmc_dir", help="Directory containing MCMC chains")
    parser.add_argument('--output', '-o')

    args = parser.parse_args()

    output_dir = common.setup_output_directory(args.output, 'compare_protocol_mcmcs')

    regex = re.compile("^mcmc_([a-z|A-Z]*)_([A-Z][0-9][0-9])_([0-9|a-z|A-Z]*)\.npy$")

    all_chains, protocol_names, rows = [], [], []

    for i, fname in enumerate(filter(regex.match, os.listdir(args.mcmc_dir))):
        protocol_name = re.search(regex, fname).group(3)
        well = re.search(regex, fname).group(2)
        model_name = re.search(regex, fname).group(1)

        chains = np.load(os.path.join(args.mcmc_dir, fname))

        rows.append([model_name, well, protocol_name, i])

        print(chains)

        # Flatten chains together
        chains = chains.reshape((-1, chains.shape[-1]), order='C')

        print(chains.shape)

        all_chains.append(chains)

    df = pd.DataFrame(rows, columns=('model', 'well', 'protocol', 'chain_index'))

    print(df)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.subplots()

    # Iterate for each model and well

    for well in df['well'].unique():
        for model_name in df['model'].unique():
            df_view = df[(df.well == well) & (df.model == model_name)]

            model = common.get_model_class(model_name)()
            param_labels = model.get_parameter_labels()
            for i, param_label in enumerate(param_labels):
                for index, row in df_view.iterrows():
                    chain = all_chains[row['chain_index']]
                    print(chain)
                    sns.kdeplot(chain[:, i], ax=ax, label=row['protocol'])

                    ax.set_title(param_label)

                    ax.legend()

                    fig.savefig(os.path.join(output_dir, f"{model_name}_{well}_{param_label}_kde.png"))

                    ax.cla()


if __name__ == '__main__':
    main()
