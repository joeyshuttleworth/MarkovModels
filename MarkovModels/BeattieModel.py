import numpy as np
import sympy as sp
from scipy.integrate import odeint

from MarkovModel import MarkovModel
from MarkovChain import MarkovChain
from settings import settings
import common

class BeattieModel(MarkovModel):
    """
    A four state Hodgkin-Huxley model with k=e^{av+b} parameters
    """

    n_params = 9
    n_states = 4
    n_state_vars = n_states-1
    GKr_index = 8
    open_state_index = 1
    Erev = common.calculate_reversal_potential()
    holding_potential = -80

    @classmethod
    def get_default_parameters(self):
        return np.array([2.07E-3, 7.17E-2, 3.44E-5, 6.18E-2, 20, 2.58E-2, 2,
                         2.51E-2, 3.33E-2])

    def __init__(self, protocol=None, times=None, Erev: float = None):
        # Create symbols for symbolic functions
        symbols = self.CreateSymbols()

        if Erev is None:
            self.Erev = common.calculate_reversal_potential()
        else:
            self.Erev = Erev

        if times is None:
            times = np.linspace(0, 15000, 1000)

        mc = MarkovChain()
        rates = ['k{}'.format(i) for i in [1,2,3,4]]
        mc.add_rates(rates)
        states = [('O', True) , ('C', False), ('I', False), ('IC', False)]
        mc.add_states(states)

        rates = [('O', 'C', 'k2', 'k1'), ('I', 'IC', 'k1', 'k2'), ('IC', 'I', 'k1', 'k2'), ('O', 'I', 'k3', 'k4'), ('C', 'IC', 'k3', 'k4')]

        for r in rates:
            mc.add_both_transitions(*r)

        self.state_labels = ['C', 'O', 'I']
        A, B = mc.eliminate_state_from_transition_matrix(self.state_labels)

        rates = dict([("k{}".format(i+1), symbols['p'][2*i]*sp.exp(symbols['p'][2*i+1]*symbols['v'])) for i in range(int(self.n_params/2))])

        A = A.subs(rates)
        B = B.subs(rates)

        p = symbols['p']
        y = symbols['y']
        v = symbols['v']

        # Define system equations and initial conditions
        k1 = p[0] * sp.exp(p[1] * v)
        k2 = p[2] * sp.exp(-p[3] * v)
        k3 = p[4] * sp.exp(p[5] * v)
        k4 = p[6] * sp.exp(-p[7] * v)


        # Notation is consistent between the two papers
        A = sp.Matrix([[-k1 - k3 - k4, k2 -  k4, -k4], [k1, -k2 - k3, k4], [-k1, k3 - k1, -k2 - k4 - k1]])
        B = sp.Matrix([k4, 0, k1])

        # Call the constructor of the parent class, MarkovModel
        super().__init__(symbols, A, B, times, rates, voltage=protocol)

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
        return {'p' : p, 'y' : y, 'v' : v}
