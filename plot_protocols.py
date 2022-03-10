#!/usr/bin/env

import matplotlib.pyplot as plt
from MarkovModels import common
import argparse
import pandas as pd
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', type=str, default='output')
    args = parser.parse_args()
    prot_dir = common.get_protocol_directory()
    plot_dir = os.path.join(args.output, 'protocols')

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    for protocol in common.get_protocol_list():
        df = pd.read_csv(os.path.join(prot_dir, protocol + '.csv'))
        df.set_index('time', inplace=True)
        fig = plt.figure(figsize=(22, 20))
        ax = fig.subplots()
        df.plot(title=protocol, ax=ax)
        fig.savefig(f"{os.path.join(plot_dir, protocol)}.pdf")
        fig.clf()


if __name__ == "__main__":
    main()
