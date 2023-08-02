import numpy as np
import sympy as sp
from markov_builder.example_models import construct_kemp_model
from scipy.integrate import odeint

from . import common
from . MarkovModel import MarkovModel


class KempModel(MarkovModel):
    """The model described in https://doi.org/10.1085/jgp.202112923
    constructed using Markov_builder
    """

    def __init__(self, times=None, voltage=None,
                 parameters=None, *args, **kwargs):
        # Create symbols for symbolic functions

        mc = construct_kemp_model()

        self.default_parameters = np.array([val
                                            for key, val in mc.default_values.items()
                                            if str(key) not in ['V', 'E_Kr']])
        self.parameter_labels = [key
                                 for key in mc.default_values
                                 if str(key) not in ['E_Kr', 'V']]

        if parameters is not None:
            self.default_parameters = parameters

        if times is None:
            times = np.linspace(0, 15000, 1000)

        self.state_labels = list(mc.graph)

        A, B = mc.eliminate_state_from_transition_matrix()

        _, Q = mc.get_transition_matrix()

        symbols = {}
        symbols['v'] = sp.sympify('V')
        symbols['p'] = sp.Matrix([sp.sympify(p) for p in self.parameter_labels if p != 'E_Kr'])
        symbols['y'] = sp.Matrix([mc.get_state_symbol(s)
                                  for s in self.state_labels[:-1]])

        self.n_params = len(self.parameter_labels)
        self.n_states = len(symbols['y']) + 1
        self.n_state_vars = self.n_states - 1
        self.GKr_index = self.parameter_labels.index('g_Kr')
        self.open_state_index = 0

        if 'tolerances' not in kwargs:
            # use fine tolerances
            kwargs['tolerances'] = (1e-8, 1e-8)

        super().__init__(symbols, A, B, mc.rate_expressions, times,
                         voltage=voltage, Q=Q, *args, **kwargs,
                         name='KempModel')

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
