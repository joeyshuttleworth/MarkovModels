#!/usr/bin/env python3

import numpy as np
import pandas as pd
import scipy.interpolate
import symengine as se
import sympy
import matplotlib.pyplot as plt
import argparse
import os

from settings import Params
from sensitivity_equations import GetSensitivityEquations, CreateSymbols
from common import calculate_resting_potential

class PintsWrapper(pints.ForwardModelS1):

    def __init__(self, settings, args, times_to_use):
        par = Params()
        self.times_to_use = times_to_use
        self.starting_parameters = [2.26E-04, 0.0699, 3.45E-05, 0.05462, 0.0873, 8.92E-03, 5.150E-3, 0.03158, 0.1524]
        # Create symbols for symbolic functions
        p, y, v = CreateSymbols(settings)

        # Choose starting parameters (from J Physiol paper)
        para = [2.26E-04, 0.0699, 3.45E-05, 0.05462, 0.0873, 8.92E-03, 5.150E-3, 0.03158, 0.1524]
        # Choose parameters (make sure conductance is the last parameter)
        para = np.array([2.07, 7.17E1, 3.44E-2, 6.18E1, 4.18E2, 2.58E1, 4.75E1, 2.51E1, 3.33E1])
        para = para*1E-3
    # Compute resting potential for 37 degrees C
    resting_potential = calculate_resting_potential(temp=37)
    par.Erev = resting_potential



        # Create symbols for symbolic functions
        p, y, v = CreateSymbols(par)

        # Define system equations and initial conditions
        k1 = p[0] * se.exp(p[1] * v)
        k2 = p[2] * se.exp(-p[3] * v)
        k3 = p[4] * se.exp(p[5] * v)
        k4 = p[6] * se.exp(-p[7] * v)

        # Write in matrix form taking y = ([C], [O], [I])^T

        A = se.Matrix([[-k1 - k3 - k4, k2 -  k4, -k4], [k1, -k2 - k3, k4], [-k1, k3 - k1, -k2 - k4 - k1]])
        B = se.Matrix([k4, 0, k1])

        rhs = np.array(A * y + B)

        self.funcs = GetSensitivityEquations(par, p, y, v, A, B, para, times_to_use, sine_wave=args.sine_wave)

    def n_parameters(self):
        return len(self.starting_parameters)

    def simulate(self, parameters, times):
        ret = self.funcs.SimulateForwardModel(parameters)
        # print(ret.shape)
        return ret

    def simulateS1(self, parameters, times):
        return self.funcs.SimulateForwardModelSensitivites(parameters, data), self.times_to_use, 1



class Boundaries(pints.Boundaries):
    def check(self, parameters):
        '''Check that each rate constant lies in the range 1.67E-5 < A*exp(B*V) < 1E3
        '''

        for i in range(0, 4):
            alpha = parameters[2*i]
            beta  = parameters[2*i + 1]

            vals = [0,0]
            vals[0] = alpha * np.exp(beta * -90 * 1E-3)
            vals[1] = alpha * np.exp(beta *  50 * 1E-3)

            for val in vals:
                if val < 1E-7 or val > 1E3:
                    return False
        # Check maximal conductance
        if parameters[8] > 0 and parameters[8] < 2:
            return True
        else:
            return False

    def n_parameters(self):
        return 9

def extract_times(lst, time_ranges, step):


