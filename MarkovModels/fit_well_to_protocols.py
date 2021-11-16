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

def main():
    parser = common.get_parser(data_reqd=True, description="Fit a given well to the data from each of the protocols. Output the resulting parameters to a file for later use")
    parser.add_argument('protocol', type=str)
    parser.add_argument('--well', "-w", type=str, default="C04")
    parser.add_argument('--max_iterations', '-i', type=int, default="100000")
    args = parser.parse_args()
    output_dir = os.path.join('output', args.output)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not re.compile("^[A-Z][0-9][0-9]$").match(args.well):
        raise ValueError(f"Well {well} not valid.")

    common.fit_well_to_data(args.well, args.protocol, args.data_file_path, args.max_iterations, output_dir)

if __name__ == "__main__":
    main()
