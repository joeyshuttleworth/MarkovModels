#!/usr/bin/env python3

import os
import unittest
import pints
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.integrate as integrate
import symengine as se
import argparse

import argparse
import matplotlib.pyplot as plt

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
            model.SimulateForwardModel()
            model.SimulateForwardModelSensitivities()

            times = np.linspace(-0.1, 20000, 200)
            full_solution = model.SimulateForwardModelSensitivities(times=times)
            solution_without_sensitivities = model.SimulateForwardModel(times=times)

            difference = full_solution[0] - solution_without_sensitivities

            # Assert that solutions are similar
            assert(sum(difference**2) < 1e-5)

            # Assert that initial and final sensitivities are similar (should be
            # true by definition)

            sens_difference = full_solution[1][0,:] - full_solution[1][-1,:]
            sse=sum(sens_difference**2)
            self.assertLess(sse, 1e-2)

    def test_analytic_solution(self):
        """Compute an analytic solution and check that it matches the numerical solution
        """
        times = np.linspace(0, 8000,1000)
        voltage = lambda t : 40

        model=MarkovModels.BeattieModel(voltage, times=times)
        analytic_solution=model.get_analytic_solution(times=times, voltage=voltage(0))
        numerical_solution=model.SimulateForwardModel(times=times)
        errors = analytic_solution - numerical_solution
        sse = max(errors)

        logging.info("analytic_solution is {}".format(analytic_solution))
        logging.info("numerical_solution is {}".format(numerical_solution))
        logging.info("errors are {}".format(errors))


        self.assertLess(sse, 1e-2)
