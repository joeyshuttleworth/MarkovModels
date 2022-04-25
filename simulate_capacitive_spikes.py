#!/usr/bin/env python3

from MarkovModels.BeattieModel import BeattieModel
import argparse
import os
from MarkovModels import common
from MarkovModels.ArtefactModel import ArtefactModel
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Plot output from different protocols")
    parser.add_argument("--parameter_file", type=str, default=None)
    parser.add_argument("--noise", "-s", type=float, default=0.01)
    parser.add_argument("--protocols", "-p", nargs='+', default=['staircase'])
    parser.add_argument("--output", "-o")

    params = np.array([2.07E-3, 7.17E-2, 3.44E-5, 6.18E-2, 4.18E-1, 2.58E-2,
                       4.75E-2, 2.51E-2, 3.33E-2])

    args = parser.parse_args()

    output_dir = common.setup_output_directory(args.output, 'simulate_capacitive_spikes')

    for protocol in args.protocols:
        protocol_func, t_start, t_end, t_step, protocol_description = common.get_ramp_protocol_from_csv(protocol)

        times = np.linspace(t_start, t_end, int((t_end - t_start) / t_step))
        channel_model = BeattieModel(protocol_func, times, tolerances=(1e-9, 1e-9),
                                     parameters=params)

        channel_model.protocol_description = protocol_description

        combined_model = ArtefactModel(channel_model, C_m=80)

        data = combined_model.simulate_model(return_current=True) + np.random.normal(0, args.noise, len(times))

        fig, ax = plt.subplots(2)
        ax[0].plot(times, data, label='combined model current')
        ax[0].plot(times, channel_model.SimulateForwardModel(), label='idealised model current')
        ax[0].legend()

        voltages = [protocol_func(t) for t in times]

        ax[1].plot(times, voltages, label='V_in')
        ax[1].plot(times, combined_model.simulate_model()[:, -1], label='V_m')
        ax[1].legend()

        fig.savefig(os.path.join(output_dir, f"simulate_capacitive_spikes_{protocol}.png"))



if __name__ == "__main__":
    main()
