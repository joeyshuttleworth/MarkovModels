import numpy as np
import sympy as sp
import markov_builder
from scipy.integrate import odeint

from . MarkovModel import MarkovModel

class ClosedOpenModel(MarkovModel):

    def get_default_parameters(self):
        return self.default_parameters

    def __init__(self, voltage=None, times=None, Erev: float = None,
                 parameters=None, *args, **kwargs):
        # Create symbols for symbolic functions

        name = 'ClosedOpenModel'
        mc = markov_builder.MarkovChain(name=name)
        mc.add_state('O', open_state=True)
        mc.add_state('C')

        mc.add_both_transitions('O', 'C', 'b', 'a')

        rates_dict = {
            'b': markov_builder.rate_expressions.negative_rate_expr + ((2.26E-4, 6.99E-2),),
            'a': markov_builder.rate_expressions.positive_rate_expr + ((3.44E-5, 5.46E-2),)
        }

        mc.parameterise_rates(rates_dict, shared_variables=('V',))

        auxiliary_expression = sp.sympify("g_Kr * s_O * (V + E_Kr)")

        mc.define_auxiliary_expression(auxiliary_expression, 'I_kr',
                                       {
                                           'g_Kr': 7.05e-02,
                                           'E_Kr': -88})

        self.default_parameters = [val
                                   for key, val in mc.default_values.items()
                                   if str(key) != 'E_Kr']

        if parameters is not None:
            self.default_parameters = parameters

        if Erev is None:
            self.Erev = -80
        else:
            self.Erev = Erev

        if times is None:
            times = np.linspace(0, 15000, 1000)

        self.state_labels = list(mc.graph)
        self.parameter_labels = [key for key in mc.default_values if str(key) != 'E_Kr']

        A, B = mc.eliminate_state_from_transition_matrix()

        Q = mc.get_transition_matrix()

        symbols = {}
        symbols['v'] = sp.sympify('V')
        symbols['p'] = [sp.sympify(p) for p in self.parameter_labels]
        symbols['y'] = sp.Matrix([mc.get_state_symbol(s)
                                  for s in self.state_labels[:-1]])

        self.n_params = len(self.parameter_labels)
        self.n_states = len(symbols['y']) + 1
        self.n_state_vars = self.n_states - 1
        self.GKr_index = self.parameter_labels.index('g_Kr')
        self.open_state_index = 0

        super().__init__(symbols, A, B, mc.rate_expressions, times, voltage=voltage,
                         Q=Q, *args, **kwargs, name=name)

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
