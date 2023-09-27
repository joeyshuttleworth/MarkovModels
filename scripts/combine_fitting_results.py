import argparse
from glob import glob

import os
import pandas as pd
import numpy as np
import itertools
import regex as re
import markovmodels

# from fit_all_wells_and_protocols import compute_predictions_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('combine_dir')
    parser.add_argument('data_directories')
    parser.add_argument('--filename', default='fitting.csv')
    parser.add_argument('--output', '-o')
    parser.add_argument('--figsize', '-f', nargs=2, type=int)
    parser.add_argument('--experiment_name', default='newtonrun4', type=str)

    args = parser.parse_args()
    model_class = markovmodels.get_model_class(args.model_class)

    glob_string = args.combine_dir + f"/*/{args.filename}"

    res = [glob(os.path.join(directory, glob_string)) for directory\
           in args.data_directories]

    # Flatten res
    res = itertools.chain(res)
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

    output_dir = markovmodels.setup_output_directory(args.output, 'combine_fitting_results')

    df.to_csv(os.path.join(output_dir, 'combined_fitting_results.csv'))

    param_labels = model_class().get_parameter_labels()

    # filter out parameters that dont lie in parameter set
    remove = []
    for i, row in df.iterrows():
        params = row[param_labels]
        remove.append(not check(model_class, parameters=params))

    df = df[~np.array(remove)]

    # predictions_df = compute_predictions_df(df, output_dir, args=args,
    #                                         model_class=model_class)
    # predictions_df.to_csv(os.path.join(output_dir, 'combined_predictions_df.csv'))


def check(model_class, parameters):
    parameters = parameters.copy()

    mm = model_class()

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
