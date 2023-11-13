import markovmodels.voltage_protocols
import os
import numpy as np
from argparse import ArgumentParser
from ast import literal_eval as createTuple


def main():
    parser = ArgumentParser()
    parser.add_argument('fnames', nargs='+')
    global args
    args = parser.parse_args()

    for fname in args.fnames:
        desc = np.loadtxt(fname, delimiter=',')
        output_dir = os.path.dirname(fname)
        out_fname = f"{os.path.splitext(fname)[0]}_txt.txt"
        with open(os.path.join(output_dir, out_fname), 'w') as fout:
            for line in markovmodels.voltage_protocols.desc_to_table(desc)[:-1]:
                fout.write(line)
                print(line)
                fout.write('\n')

if __name__ == '__main__':
    main()
