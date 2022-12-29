from MarkovModels import common
from MarkovModels.BeattieModel import BeattieModel

import matplotlib.pyplot as plt
import argparse

import numpy as np
import pandas as pd
import os
from matplotlib.gridspec import GridSpec

from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 8})
rc('text', usetex=True)
rc('figure', dpi=300)

protocols_list = sorted(['staircaseramp1', 'sis', 'spacefill19', 'hhbrute3gstep', 'wangbrute3gstep'])
protocols_list = ['longap'] + protocols_list

relabel_dict = {p: f"$d_{i+1}$" for i, p in enumerate([p for p in protocols_list if p != 'longap'])}

relabel_dict['longap'] = '$d^*$'


def setup_axes(fig):
    gs = GridSpec(6, 3, width_ratios=[.25, 1, 1])

    gs.update(top=.975, bottom=0.075, wspace=0.5, hspace=0.3, right=.85)
    axes = [fig.add_subplot(cell) for cell in gs]

    return axes


def main():

    parser = argparse.ArgumentParser('--figsize')
    global args
    args = parser.parse_args()

    for model_name in ['Beattie', 'Wang']:
        model_class = common.get_model_class(model_name)
        model = model_class()

        print(model.mc.as_latex())


if __name__ == '__main__':
    main()
