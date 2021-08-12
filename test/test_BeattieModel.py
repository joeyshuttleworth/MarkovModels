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

import MarkovModels

def simple_step_protocol(t : float):
    if t >= 1000 and t < 5000:
        V = 20
    else:
        V = -80
    return V

class TestBeattieModel(unittest.TestCase):
    protocol_names = ["staircase", "Beattie-AP", "sine-wave"]
    def test_initialise_and_run(self):
        for protocol_name in self.protocol_names:
            protocol = MarkovModels.get_protocol(protocol_name)
            model=MarkovModels.BeattieModel(protocol)
            model.SimulateForwardModel()
            model.SimulateForwardModelSensitivities()
        model = MarkovModels.BeattieModel(simple_step_protocol)

        times = np.linspace(-0.1, 20000, 200)
        full_solution = model.SimulateForwardModelSensitivities(times=times)
        solution_without_sensitivities = model.SimulateForwardModel(times=times)

        difference = full_solution[0] - solution_without_sensitivities

        # Assert that solutions are similar
        assert(sum(difference**2) < 1e-5)

        # Assert that initial and final sensitivities are similar (should be
        # true by definition)

        sens_difference = full_solution[1][0,:] - full_solution[1][-1,:]
        assert(sum(sens_difference**2) < 1e-5)



