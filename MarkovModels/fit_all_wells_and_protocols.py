#!/usr/bin/env python3

from BeattieModel import BeattieModel
import argparse
import os
import common
import numpy as np
import logging
import matplotlib.pyplot as plt
import regex as re
import pandas as pd
import common
import multiprocessing
from BeattieModel import BeattieModel

def fit_func(protocol, well):

    default_parameters = np.array([0.00023215680795174809,0.07422110165735675,2.477501557744992e-05,0.04414799725791213,0.11023652619943154,0.015996823969951217,0.015877336172564104,0.027816696279347616,49.70368237942998])
    common.fit_well_to_data(BeattieModel, well, protocol, args.data_file_path, args.max_iterations, os.path.join(output_dir, "fitting", f"{protocol}_{well}"), T=298, K_in=5, K_out=130, default_parameters=default_parameters, removal_duration=args.removal_duration)

def main():
    parser = common.get_parser(data_reqd=True, description="Fit a given well to the data from each of the protocols. Output the resulting parameters to a file for later use")
    parser.add_argument('--max_iterations', '-i', type=int, default="100000")
    parser.add_argument('--wells', '-w', type=str, action='append', default=[], nargs='+')
    parser.add_argument('--protocols', type=str, default=[], nargs='+')
    parser.add_argument('--removal_duration', '-r', default=5)
    global args
    args = parser.parse_args()

    global output_dir
    output_dir = args.output

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    regex = re.compile("^newtonrun4-([a-z|A-Z|0-9]*)-([A-Z][0-9][0-9]).csv$")

    if len(args.wells) == 0:
        args.wells = common.get_all_wells_in_directory(args.data_file_path, regex=regex, group=1)

    if len(args.protocols)==0:
        protocols = common.get_protocol_list()
    else:
        protocols  = args.protocols

    pool = multiprocessing.Pool(processes=os.cpu_count())

    tasks = []
    for f in filter(regex.match, os.listdir(args.data_file_path)):
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
