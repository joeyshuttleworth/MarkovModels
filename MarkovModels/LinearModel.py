import numpy as np
import sympy as sp
from scipy.integrate import odeint

from . MarkovModel import MarkovModel
from . import common

from numba import njit


class LinearModel():
    """

    A simple linear model which mocks some of the functions in MarkovModel.
    Used for testing and debugging.

    """

    n_params = 9
    n_states = 4
    n_state_vars = n_states - 1
    GKr_index = 8
    open_state_index = 1
    holding_potential = -80

    def get_default_parameters(self):
        return self.default_parameters

    def __init__(self, voltage=None, times=None, Erev: float = None,
                 parameters=None, *args, **kwargs):
        # Create symbols for symbolic functions

        if parameters is None:
            self.default_parameters = np.array((2.26E-4, 6.99E-2, 3.44E-5, 5.460E-2, 0.0873,
                                                8.91E-3, 5.15E-3, 0.003158, 0.1524))
        else:
            self.default_parameters = parameters
        self.times = times

    def get_design_matrix(self):
        # Ignore njit keyword
        t_indices = np.array(range(len(self.times)))

        # Construct the columns of the design matrix (arbitrary choice)
        vecs = [(t_indices % ((i+1)**4) != 0).astype(np.float64) for i in range(self.n_params)]

        # Return design matrix
        return np.column_stack(vecs) + np.eye(len(self.times), self.n_params)

    def get_no_parameters(self):
        return self.n_params

    def GetStateVariables(self, p=None, times=None):
        if p is None:
            p = self.get_default_parameters()
        if times is None:
            times = self.times

        p = p.flatten()

        return p[None, :] * self.get_design_matrix()

    def make_hybrid_solver_current(self, njitted=True):
        # Design matrix
        X = self.get_design_matrix()

        def solver(p=self.default_parameters, times=self.times):
            return X @ p
        return njit(solver) if njitted else solver

    def SimulateForwardModelSensitivities(self, p=None, times=None):
        X = self.get_design_matrix()
        print(X)
        return self.make_hybrid_solver_current()(p, times), X

