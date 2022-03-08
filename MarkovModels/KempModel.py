import numpy as np
import sympy as sp
from markov_builder.example_models import construct_kemp_model
from scipy.integrate import odeint

from . MarkovModel import MarkovModel
from . import common

class KempModel(MarkovModel):
    """The model described in https://doi.org/10.1085/jgp.202112923
    constructed using Markov_builder
    """

    def get_default_parameters(self):
        return self.default_parameters

    def __init__(self, voltage=None, times=None, Erev: float = None,
                 parameters=None, *args, **kwargs):
        # Create symbols for symbolic functions

        mc = construct_kemp_model()

        self.default_parameters = np.fromiter(mc.default_values.values(),
                                              dtype='float')

        if parameters is not None:
            self.default_parameters = parameters

        if Erev is None:
            self.Erev = common.calculate_reversal_potential()
        else:
            self.Erev = Erev

        if times is None:
            times = np.linspace(0, 15000, 1000)

        self.state_labels = list(mc.graph)
        self.parameter_labels = list(mc.default_values)

        A, B = mc.eliminate_state_from_transition_matrix()

        Q = mc.get_transition_matrix()

        symbols = {}
        symbols['v'] = sp.sympify('V')
        symbols['p'] = [sp.sympify(p) for p in self.parameter_labels]
        symbols['y'] = sp.Matrix([mc.get_state_symbol(s)
                                  for s in self.state_labels[:-1]])

        self.n_params = len(self.parameter_labels)
        self.n_states = len(symbols['y']) - 1
        self.n_state_vars = self.n_states - 1
        self.GKr_index = -1
        self.open_state_index = 0
        # self.holding_potential = -80

        super().__init__(symbols, A, B, mc.rate_expressions, times, voltage=voltage,
                         Q=Q, *args, **kwargs)

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
