
import multiprocessing
import regex as re
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from MarkovModels import common
from MarkovModels.BeattieModel import BeattieModel
from MarkovModels.ClosedOpenModel import ClosedOpenModel
from MarkovModels.KempModel import KempModel
import argparse
import regex as re
from glob import glob

import matplotlib
matplotlib.use('agg')

import os
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('--output', '-o')

    args = parser.parse_args()

    glob_string = args.data_dir + r'/*/*fitted_params*csv'
    print(glob_string)
    res = glob(glob_string)
    print(res)

    rgex = re.compile(r'([A-Z|0-9][0-9][0-9])_([a-z|A-Z|0-9]*)_fitted_params.csv$')

    dfs = []
    for fname in res:
        well = re.search(rgex, fname).groups()[0]
        protocol = re.search(rgex, fname).groups()[1]

        this_df = pd.read_csv(fname)
        this_df['protocol'] = protocol
        this_df['well'] = well
        dfs.append(this_df)

    df = pd.concat(dfs, ignore_index=True)
    print(df)

    output_dir = common.setup_output_directory(args.output, 'combine_fitting_results')

    df.to_csv(os.path.join(output_dir, 'combined_fitting_results.csv'))


if __name__ == '__main__':
    main()
