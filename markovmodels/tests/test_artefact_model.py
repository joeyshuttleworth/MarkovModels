#!/usr/bin/env python3

import logging
import os
import unittest

import matplotlib.pyplot as plt
import numpy as np

import markovmodels
from markovmodels.ArtefactModel import ArtefactModel
from markovmodels.utilities import setup_output_directory
from markovmodels.model_generation import make_model_of_class
from markovmodels.voltage_protocols import get_ramp_protocol_from_csv


class TestModelGeneration(unittest.TestCase):

    def setUp(self):
        test_output_dir = setup_output_directory('test_output', 'test_artefact_model')
        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)
        self.output_dir = test_output_dir

        self.model_names = [f"model{i}" for i in range(15)]
        logging.info("outputting to " + test_output_dir)

    def test_artefact_model(self):
        protocol = 'staircaseramp1'
        voltage_func, times, desc = get_ramp_protocol_from_csv(protocol)

        channel_model = make_model_of_class('model3', times,
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

        plt.plot(times, artefact_model.make_hybrid_solver_states(hybrid=False,
                                                                 njitted=False)(),
                 label=artefact_model.get_state_labels())

        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'artefact_states'))
        plt.clf()

    def test_generated_model_output(self):
        protocol = 'staircaseramp'

        for original_model, generated_model in [['Beattie', 'model3']]:
            voltage_func, times, desc = get_ramp_protocol_from_csv(protocol)
            tolerances = 1e-10, 1e-10

            c_model1 = markovmodels.make_model_of_class(original_model, times,
                                                        voltage=voltage_func,
                                                        protocol_description=desc,
                                                        tolerances=tolerances)
            c_model2 = markovmodels.make_model_of_class(generated_model,
                                                        voltage=voltage_func,
                                                        times=times,
                                                        protocol_description=desc,
                                                        tolerances=tolerances)

            model1 = ArtefactModel(c_model1)
            model2 = ArtefactModel(c_model2)

            default_parameters = model1.get_default_parameters()

            h_solver1 = model1.make_forward_solver_current(njitted=False)
            h_solver2 = model2.make_forward_solver_current(njitted=False)

            output1 = h_solver1(default_parameters)
            output2 = h_solver2(default_parameters)

            rmse_error = np.sqrt(((output1 - output2)**2).mean())

            logging.debug('rmse error is: ' + f"{rmse_error}")

            plt.plot(times, output1, label=original_model)
            plt.plot(times, output2, label='MarkovBuilder model')

            plt.legend()
            plt.savefig(os.path.join(self.output_dir, f"{original_model}_{generated_model}_comparison_lsoda.pdf"))
            plt.clf()
            self.assertLess(rmse_error, 1e-2)

    def test_steady_state_unique(self):
        pass


if __name__ == "__main__":
    # logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()
