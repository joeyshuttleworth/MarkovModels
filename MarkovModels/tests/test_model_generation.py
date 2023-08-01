#!/usr/bin/env python3

import itertools
import logging
import os
import unittest

import matplotlib.pyplot as plt
import myokit
import myokit as mk
import networkx as nx
import numpy as np
import sympy as sp

from MarkovModels import common
from MarkovModels.BeattieModel import BeattieModel
from MarkovModels.ClosedOpenModel import ClosedOpenModel
from MarkovModels.KempModel import KempModel


class TestModelGeneration(unittest.TestCase):

    def setUp(self):
        test_output_dir = common.setup_output_directory('test_output', 'test_model_generation')
        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)
        self.output_dir = test_output_dir
        logging.info("outputting to " + test_output_dir)

    def test_Beattie_model_output(self):
        protocol = 'staircaseramp'

        voltage_func, times, desc = common.get_ramp_protocol_from_csv(protocol)

        print(voltage_func(0))

        model1 = common.make_model_of_class('Beattie', times, voltage_func,
                                            protocol_description=desc)
        model2 = common.make_model_of_class('model3', voltage=voltage_func,
                                            times=times, protocol_description=desc)

        default_parameters = model1.get_default_parameters()
        h_solver1 = model1.make_hybrid_solver_current()
        h_solver2 = model2.make_hybrid_solver_current()

        output1 = h_solver1(default_parameters)
        output2 = h_solver2(default_parameters)

        rmse_error = np.sqrt(((output1 - output2)**2).mean())

        logging.debug('rmse error is: ' + f"{rmse_error}")

        plt.plot(times, output1, label='Beattie model')
        plt.plot(times, output2, label='MarkovBuilder model')

        plt.savefig(os.path.join(self.output_dir, "Beattie_model3_comparison.pdf"))

        self.assertLess(rmse_error, 1e-10)

