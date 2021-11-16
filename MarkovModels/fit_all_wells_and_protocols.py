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

args = None

def fit_func(protocol, well):
    common.fit_well_to_data(BeattieModel, well, protocol, args.data_file_path, args.max_iterations, output_dir)

def main():
    parser = common.get_parser(data_reqd=True, description="Fit a given well to the data from each of the protocols. Output the resulting parameters to a file for later use")
    parser.add_argument('--max_iterations', '-i', type=int, default="100000")

    global args
    args = parser.parse_args()

    global output_dir
    output_dir = os.path.join('output', args.output)

    protocols = common.get_protocol_list()

    regex = re.compile("^newtonrun4-([a-z|A-Z|0-9]*)-([A-Z][0-9][0-9]).csv$")

    pool = multiprocessing.Pool(processes=os.cpu_count())

    tasks = []
    for f in filter(regex.match, os.listdir(args.data_file_path)):
        protocol, well = re.search(regex, f).groups()

        if protocol not in protocols:
            logging.error(f"Protocol, {protocol}, is not one of {protocols}")
            continue
        else:
            tasks.append((protocol, well))


    pool.starmap(fit_func, tasks)
    print("=============\nfinished\n==========")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
