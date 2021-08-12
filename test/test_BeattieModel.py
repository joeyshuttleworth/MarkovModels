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

class TestBeattieModel(unittest.TestCase):
    protocols = ["staircase", "Beattie-AP", "sine-wave"]
    def test_initialise_and_run(self):
        for protocol in self.protocols:
            model=MarkovModels.BeattieModel(protocol)
            model.SimulateForwardModel()
            model.SimulateForwardModelSensitivities()
