#!/usr/bin/env python3

import multiprocessing
import regex as re
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from MarkovModels import common
from MarkovModels.BeattieModel import BeattieModel
from MarkovModels.ClosedOpenModel import ClosedOpenModel
from MarkovModels.KempModel import KempModel
import argparse
import seaborn as sns
import os
import string
import re
from matplotlib.gridspec import GridSpec

from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

protocol_chrono_order = ['staircaseramp1',
                         'sis',
                         'rtovmaxdiff',
                         'rvotmaxdiff',
                         'spacefill10',
                         'spacefill19',
                         'spacefill26',
                         'longap',
                         'hhbrute3gstep',
                         'hhsobol3step',
                         'wangbrute3gstep',
                         'wangsobol3step',
                         'staircaseramp2']

relabel_dict = dict(zip(protocol_chrono_order,
                        string.ascii_uppercase[:len(protocol_chrono_order)]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datafiles', nargs='+')
    parser.add_argument('--repeats', type=int, default=16)
    parser.add_argument('--wells', '-w', type=str, default=[], nargs='+')
    parser.add_argument('--removal_duration', '-r', default=5, type=float)
    parser.add_argument('--experiment_name', default='newtonrun4', type=str)
    parser.add_argument('--no_chains', '-N', default=0, help='mcmc chains to run', type=int)
    parser.add_argument('--chain_length', '-l', default=500, help='mcmc chains to run', type=int)
    parser.add_argument('--figsize', '-f', type=int, nargs=2)
    parser.add_argument('--use_parameter_file')
    parser.add_argument('-i', '--ignore_protocols', nargs='+', default=[])
    parser.add_argument('-o', '--output_dir')
    parser.add_argument("-A", "--alphabet_labels", action='store_true')
    parser.add_argument("-F", "--file_format", default='pdf')
    parser.add_argument("-m", "--model", default='Beattie')
    parser.add_argument('--true_param_file')

    global args
    args = parser.parse_args()

    fig = plt.figure(figsize=args.figsize)
    axes = create_axes(fig)


def create_axes(fig):
    gs = GridSpec(3, 6, height_ratios=[0.2, 1, 1, 1, 1, 1])
    axes = [fig.add_subplot(cell) for cell in gs]
    return axes


if __name__ == "__main__":
    main()
