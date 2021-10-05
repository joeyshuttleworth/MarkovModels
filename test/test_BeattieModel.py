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

class TestBeattieModel(unittest.TestCase):
    protocol_names = ["staircase", "Beattie-AP", "sine-wave"]
    def test_sensitivities(self):
        """test_sensitivities

        Check that the initial conditions for the sensitivity ODEs correspond
        to the limit when t -> \infty. Perform this test for all protocols.

        """
        for protocol_name in self.protocol_names:
            protocol = MarkovModels.get_protocol(protocol_name)
            model=MarkovModels.BeattieModel(protocol)

            times = np.linspace(-0.1, 30000, 1000)
            full_solution = model.SimulateForwardModelSensitivities(times=times)
            solution_without_sensitivities = model.SimulateForwardModel(times=times)

            difference = full_solution[0] - solution_without_sensitivities

            # Assert that solutions are similar

            # Assert that initial and final sensitivities are similar (should be
            # true by definition)

            sens_difference = full_solution[1][0,:] - full_solution[1][-1,:]
            sse=sum(sens_difference**2)
            # logging.info("errors are {}".format(sens_difference))
            self.assertLess(max(sens_difference), 1e-5)

    def test_analytic_solution(self):
        """Compute an analytic solution and check that it matches the numerical solution
        """
        voltages = [-70, -40, 0, 20, 40, 180]

        for v in voltages:
            times = np.linspace(0, 100, 250)
            voltage = lambda t : 20

            model=MarkovModels.BeattieModel(voltage)
            analytic_solution=model.get_analytic_solution(times=times, voltage=voltage(times[-1]))
            numerical_solution=model.SimulateForwardModel(times=times)
            errors = analytic_solution - numerical_solution

            # logging.info("analytic_solution is {}".format(analytic_solution))
            # logging.info("numerical_solution is {}".format(numerical_solution))
            # logging.info("errors are {}".format(errors))


            self.assertLess(max(errors), 1e-5)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    unittest.main()
