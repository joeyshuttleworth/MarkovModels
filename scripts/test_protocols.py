from MarkovModels import common
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os


def main():

    directory = common.get_protocol_directory()

    output_dir = common.setup_output_directory(None, 'test_protocols')

    for protocol in common.get_protocol_list():
        voltage_func, times, protocol_desc = common.get_ramp_protocol_from_csv(protocol)

        protocol_df = pd.read_csv(os.path.join(directory, protocol + ".csv"))
        real_times = protocol_df["time"].values.flatten().astype(np.float64)
        real_vs = protocol_df["voltage"].values.flatten().astype(np.float64)

        interpolated_voltages = np.array([voltage_func(t) for t in real_times])

        print(protocol)
        print(np.max(real_vs - interpolated_voltages))
        print(real_times[np.argmax(real_vs - interpolated_voltages)])

        print(protocol_desc)

        plt.plot(real_times, interpolated_voltages - real_vs)
        plt.savefig(os.path.join(output_dir, protocol))
        plt.clf()


if __name__ == '__main__':
    main()
