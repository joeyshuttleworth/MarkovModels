import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import statsmodels as sm
import statsmodels.formula.api as smf

import markovmodels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('parameter_estimates')
    parser.add_argument('--ignore_protocols', nargs='+', default=['longap'])

    global args
    args = parser.parse_args()

    estimates_df = pd.read_csv(args.parameter_estimates)
    estimates_df = markovmodels.fitting.get_best_params(estimates_df)

    estimates_df = estimates_df[~estimates_df.protocol.isin(args.ignore_protocols)]

