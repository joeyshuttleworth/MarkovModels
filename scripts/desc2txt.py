import markovmodels.voltage_protocols
import os
import numpy as np
from argparse import ArgumentParser
from ast import literal_eval as createTuple


def main():
    parser = ArgumentParser()
    parser.add_argument('fname')
    global args
    args = parser.parse_args()

    desc = np.loadtxt(args.fname, delimiter=',')
    output_dir = os.path.dirname(args.fname)

    with open(os.path.join(output_dir, 'found_design.txt'), 'w') as fout:
        for line in markovmodels.voltage_protocols.desc_to_table(desc)[:-1]:
            fout.write(line)
            print(line)
            fout.write('\n')

if __name__ == '__main__':
    main()
