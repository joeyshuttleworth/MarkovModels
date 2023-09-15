#!/usr/bin/env python3

import logging
import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import pandas as pd
import seaborn as sns
import myokit
from numba import njit

from markovmodels import common
from markovmodels.ArtefactModel import ArtefactModel


class TestModelGeneration(unittest.TestCase):

    def setUp(self):
        test_output_dir = common.setup_output_directory('test_output', 'test_artefact_model')
        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)
        self.output_dir = test_output_dir

        self.model_names = [f"model{i}" for i in range(15)]
        logging.info("outputting to " + test_output_dir)

    def test_artefact_model(self):
        protocol = 'staircaseramp1'
        voltage_func, times, desc = common.get_ramp_protocol_from_csv(protocol)

        channel_model = common.make_model_of_class('model9', times,
                                                   voltage_func,
                                                   protocol_description=desc)

        artefact_model = ArtefactModel(channel_model)

        current = artefact_model.SimulateForwardModel()

        plt.plot(times, current, label='artefact model')
        plt.plot(times, channel_model.SimulateForwardModel(), label='no_artefacts')
        plt.ylabel(r'$I_{post}$ (nA)')
        plt.xlabel(r'$t$ (ms)')

        plt.legend()

        plt.savefig(os.path.join(self.output_dir, 'artefact_comparison'))
        plt.clf()

        print('states are')
        print(artefact_model.get_state_labels())
        plt.plot(times, artefact_model.make_hybrid_solver_states(hybrid=False, njitted=False)(),
                 label=artefact_model.get_state_labels())

        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'artefact_states'))
        plt.clf()


if __name__ == "__main__":
    # logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()
