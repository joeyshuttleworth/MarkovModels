import numpy as np
import sympy as sp
import pints
from scipy.integrate import odeint

from markov_builder.example_models import construct_four_state_chain
from . MarkovModel import MarkovModel
from . import common

class BeattieModel(MarkovModel):
    """
    A four state Hodgkin-Huxley model with k=e^{av+b} parameters
    """

    n_params = 9
    n_states = 4
    n_state_vars = n_states - 1
    GKr_index = -1
    open_state_index = 1
    holding_potential = -80

    def __init__(self, times=None, voltage=None,
                 parameters=None, *args, **kwargs):
        # Create symbols for symbolic functions
        symbols = self.CreateSymbols()

        self.mc = construct_four_state_chain()

        if parameters is None:
            self.default_parameters = np.array((2.26E-4, 6.99E-2, 3.448E-5,
                                                5.460E-2, 0.0873, 8.91E-3,
                                                5.15E-3, 0.03158, 0.1524))
        else:
            self.default_parameters = parameters

        if times is None:
            times = np.linspace(0, 15000, 1000)

        self.state_labels = ['C', 'O', 'I', 'IC']
        self.parameter_labels = [f"p{i+1}" for i in range(len(self.default_parameters) - 1)] + ['Gkr']

        p = symbols['p']
        v = symbols['v']

        rates = {"k%i" % (i + 1):
                 p[2 * i] * sp.exp((-1)**i * p[2 * i + 1] * v)
                 for i in range(int(self.n_params / 2))}

        # Notation is consistent between the two papers
        A = sp.Matrix([['-k1 - k3 - k4', 'k2 - k4', '-k4'],
                       ['k1', '-k2 - k3', 'k4'],
                       ['-k1', 'k3 - k1', '-k2 - k4 - k1']])
        B = sp.Matrix([['k4', 0, 'k1']]).T

        Q = sp.Matrix([['-k3 - k1', 'k2', '0', 'k4'],
                       ['k1', '-k2 - k3', 'k4', '0'],
                       ['0', 'k3', '-k2 - k4', 'k1'],
                       ['k3', '0', 'k2', '-k4 - k1']]).T

        self.transformations = [
            pints.LogTransformation(1),
            pints.IdentityTransformation(1),

            pints.LogTransformation(1),
            pints.IdentityTransformation(1),

            pints.LogTransformation(1),
            pints.IdentityTransformation(1),

            pints.LogTransformation(1),
            pints.IdentityTransformation(1),

            pints.LogTransformation(1)
        ]

        super().__init__(symbols, A, B, rates, times=times, voltage=voltage,
                         Q=Q, name='BeattieModel', *args, **kwargs)

    def CreateSymbols(self):
        """
        Create SymEngine symbols to contain the parameters, state variables and the voltage.
        These are used to generate functions for the right hand side and Jacobian
        """
        # Create parameter symbols
        p = sp.Matrix([sp.symbols('p%d' % j) for j in range(self.n_params)])
        # Create state variable symbols
        y = sp.Matrix([sp.symbols('y%d' % i) for i in range(self.n_state_vars)])
        # Create voltage symbol
        v = sp.symbols('v')
        return {'p': p, 'y': y, 'v': v}
