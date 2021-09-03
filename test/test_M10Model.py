#!/usr/bin/env python3

import os
import sys
import unittest
import pints
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.integrate as integrate
import symengine as se
import argparse

import logging

import MarkovModels

def simple_step_protocol(t : float):
    if t >= 1000 and t < 5000:
        V = 20
    else:
        V = -80
    return V

class TestM10Model(unittest.TestCase):
    protocol_names = ["staircase", "sine-wave"]

    # TODO put this somewhere else
    def setUp(self):
        test_output_dir = os.environ.get('MARKOVMODELS_TEST_OUTPUT', os.path.join(os.path.dirname(os.path.realpath(__file__)), self.__class__.__name__))
        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)
        self.output_dir = test_output_dir
        logging.info("outputting to " + test_output_dir)

    def test_sensitivities(self):
        """test_sensitivities

        Check that the initial conditions for the sensitivity ODEs correspond
        to the limit when t -> \infty. Perform this test for all protocols.

        """
        for protocol_name in self.protocol_names:
            protocol = MarkovModels.get_protocol(protocol_name)
            model=MarkovModels.M10Model(protocol)

            times = np.linspace(0, 20000, 10000)
            full_solution = model.SimulateForwardModelSensitivities(times=times)
            solution_without_sensitivities = model.SimulateForwardModel(times=times)

            # difference = full_solution[0] - solution_without_sensitivities

            #TODO Assert that solutions are similar

            #TODO Assert that initial and final sensitivities are similar (should be
            # true by definition)

            occupancies = model.solve_rhs(model.get_default_parameters(), times=times)
            occupancies = np.concatenate((occupancies, 1-occupancies.sum(axis=1)[:,None]), axis=1)
            print(occupancies)

            plt.plot(times, occupancies, label=model.state_labels)
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, "{}_occupancies.pdf".format(protocol_name)))
            plt.clf()
            plt.plot(times, solution_without_sensitivities)
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, "{}_solution.pdf".format(protocol_name)))
            plt.clf()
            plt.plot(times, full_solution[1])
            plt.savefig(os.path.join(self.output_dir, "{}_sensitivities.pdf".format(protocol_name)))
            plt.clf()

            initial_sens = full_solution[1][0,:]
            final_sens   = full_solution[1][-1,:]
            mrms = np.sqrt(((initial_sens - final_sens)**2/(1+np.abs(initial_sens))**2).sum()/initial_sens.shape[0])
            logging.info("mrms error is {}".format(mrms))
            self.assertLess(mrms, 1e-5)

    def test_analytic_solution(self):
        return
        """Compute an analytic solution and check that it matches the numerical solution
        """
        voltages = [-70, -40, 0, 20, 40, 180]

        for v in voltages:
            times = np.linspace(0, 100, 10)
            voltage = lambda t : 20

            model=MarkovModels.M10Model(voltage)
            analytic_solution=model.get_analytic_solution(times=times, voltage=voltage(times[-1]))
            numerical_solution=model.SimulateForwardModel(times=times)
            errors = analytic_solution - numerical_solution

            plt.plot(times, analytic_solution)
            plt.plot(times, numerical_solution)
            plt.show()

            self.assertLess(max(errors), 1e-5)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    unittest.main()
