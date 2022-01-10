#!/usr/bin/env python3
import multiprocessing
import pandas as pd
import regex as re
import matplotlib.pyplot as plt
import logging
from MarkovModels import common
from MarkovModels.BeattieModel import BeattieModel

import argparse
import os
import numpy as np
import pandas as pd

def fit_func(protocol, well):
    default_parameters = None
    this_output_dir = os.path.join(output_dir, f"fitting_{args.removal_duration}ms_removed", f"{protocol}_{well}")

    common.fit_well_to_data(BeattieModel, well, protocol, args.data_directory, args.max_iterations, this_output_dir, T=298, K_in=5, K_out=120, default_parameters=default_parameters, removal_duration=args.removal_duration, repeats=args.repeats)

def main():
    parser = common.get_parser(
        data_reqd=True, description="Fit a given well to the data from each of the protocols. Output the resulting parameters to a file for later use")
    parser.add_argument('--max_iterations', '-i', type=int, default="100000")
    parser.add_argument('--repeats', type=int, default=8)
    parser.add_argument('--wells', '-w', type=str, default=[], nargs='+')
    parser.add_argument('--protocols', type=str, default=[], nargs='+')
    parser.add_argument('--removal_duration', '-r', default=5, type=int)

    global args
    args = parser.parse_args()

    global output_dir
    output_dir = args.output

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    regex = re.compile("^newtonrun4-([a-z|A-Z|0-9]*)-([A-Z][0-9][0-9]).csv$")

    if len(args.wells) == 0:
        args.wells = common.get_all_wells_in_directory(args.data_directory, regex=regex, group=1)

    if len(args.protocols) == 0:
        protocols = common.get_protocol_list()
    else:
        protocols = args.protocols

    pool = multiprocessing.Pool(processes=os.cpu_count())

    print(args.wells, protocols)

    tasks = []
    for f in filter(regex.match, os.listdir(args.data_directory)):
        protocol, well = re.search(regex, f).groups()
        if protocol not in protocols or well not in args.wells:
            continue
        else:
            tasks.append((protocol, well))

    pool.starmap(fit_func, tasks)
    print("=============\nfinished\n=============")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
