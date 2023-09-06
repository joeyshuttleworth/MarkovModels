import numpy as np
import sympy as sp
from scipy.integrate import odeint

from . MarkovModel import MarkovModel
from . MarkovChain import MarkovChain
from . common import *
from . settings import settings


class M10Model(MarkovModel):
    """
    A four state Hodgkin-Huxley model with k=e^{av+b} parameters
    """

    n_params = 13
    n_states = 6
    n_state_vars = n_states - 1
    GKr_index = 12
    open_state_index = 0
    Erev = -88
    holding_potential = -80

    state_labels = ('O', 'IC1', 'IC2', 'IO', 'C2', 'C1')

    def get_default_parameters(self):
        return np.array((8.53183002138620944e-03, 8.31760044455376601e-02, 1.26287052202195688e-02, -1.03628499834739776e-07, 2.70276339808042609e-01, 1.58000446046794897e-02, 7.66699486356391818e-02, -2.24575000694940963e-02, 1.49033896782688496e-01, 2.43156986537036227e-02, 5.58072076984100361e-04, -4.06619125485430874e-02, 0.1524))

    def __init__(self, protocol=None, times=None):
        # Create symbols for symbolic functions
        symbols = self.CreateSymbols()

        if times is None:
            times = np.linspace(0, 15000, 1000)

        mc = MarkovChain()

        states = ('O', 'IC1', 'IC2', 'IO', 'C1', 'C2')
        mc.add_states(states)
        rates = (('IC2', 'IC1', 'a1', 'b1'), ('IC1', 'IO', 'a2', 'b2'), ('IO', 'O', 'ah', 'bh'), ('O', 'C1',
                 'b2', 'a2'), ('C1', 'C2', 'b1', 'a1'), ('C2', 'IC2', 'bh', 'ah'), ('C1', 'IC1', 'bh', 'ah'))

        for r in rates:
            mc.add_both_transitions(*r)

        A, B = mc.eliminate_state_from_transition_matrix(('O', 'IC1', 'IC2', 'IO', 'C2'))

        p = symbols['p']
        rate_vals = [p[2 * i] * sp.exp(p[2 * i + 1] * symbols['v']) for i in range(int((len(p)) / 2))]
        rate_labels = ('a1', 'b1', 'bh', 'ah', 'a2', 'b2')

        rates_dict = dict(zip(rate_labels, rate_vals))

        A = A.subs(rates_dict)
        B = B.subs(rates_dict)

        self.mc = mc
        # Call the constructor of the parent class, MarkovModel
        super().__init__(symbols, A, B, times, rate_labels, voltage=protocol)

    def CreateSymbols(self):
        """
        Create SymEngine symbols to contain the parameters, state variables and the voltage.
        These are used to generate functions for the right hand side and Jacobian
        """
        # Create parameter symbols
        p = [sp.symbols('p%d' % j) for j in range(self.n_params)]
        # Create state variable symbols
        y = sp.Matrix([sp.symbols('y%d' % i) for i in range(self.n_state_vars)])
        # Create voltage symbol
        v = sp.symbols('v')
        return {'p': p, 'y': y, 'v': v}
