import argparse
from glob import glob

import os
import pandas as pd
import numpy as np
import itertools
import regex as re
import markovmodels
from markovmodels.model_generation import make_model_of_class

# from fit_all_wells_and_protocols import compute_predictions_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directories', nargs='+')
    parser.add_argument('--filename', default='fitting.csv')
    parser.add_argument('--output', '-o')
    parser.add_argument('--figsize', '-f', nargs=2, type=int)
    parser.add_argument('--experiment_name', default='newtonrun4', type=str)
    parser.add_argument('--model_class')

    global args 
    args = parser.parse_args()

    glob_string = f"*/{args.filename}"

    res = [glob(os.path.join(directory, glob_string), recursive=True) for directory\
           in args.data_directories]
    print(res)

    # Flatten res
    res = list(itertools.chain(itertools.chain(res)))
    print(res)

    dfs = []
    for entry in res:
        if len(entry) == 0:
            continue
        fname = entry[0]
        if not isinstance(fname, str):
            continue
        this_df = pd.read_csv(fname)
        dfs.append(this_df)

    df = pd.concat(dfs, ignore_index=True)
    print(df)

    output_dir = markovmodels.utilities.setup_output_directory(args.output, 'combine_fitting_results')

    df.to_csv(os.path.join(output_dir, 'combined_fitting_results.csv'))

    if args.model_class:
        model = markovmodels.make_model_of_class(args.model_class)
        param_labels = model.get_parameter_labels()
        # filter out parameters that dont lie in parameter set
        remove = []
        for i, row in df.iterrows():
            params = row[param_labels]
            remove.append(not check(model, parameters=params))

        df = df[~np.array(remove)]


def check(mm, parameters):
    parameters = parameters.copy()

    # rates function
    rates_func = mm.get_rates_func(njitted=False)

    Vs = [-120, 60]
    rates_1 = rates_func(parameters, Vs[0])
    rates_2 = rates_func(parameters, Vs[1])

    if max(rates_1.max(), rates_2.max()) > 1e4:
        return False

    if min(rates_1.min(), rates_2.min()) < 1e-8:
        return False

    if max([p for i, p in enumerate(parameters) if i != mm.GKr_index]) > 1e5:
        return False

    if min([p for i, p in enumerate(parameters) if i != mm.GKr_index]) < 1e-7:
        return False

    # Ensure that all parameters > 0
    return np.all(parameters > 0)


if __name__ == '__main__':
    main()
