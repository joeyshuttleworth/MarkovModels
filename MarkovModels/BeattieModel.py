import numpy as np
import symengine as se
from scipy.integrate import odeint

from MarkovModel import MarkovModel
from common import *
from settings import settings


class BeattieModel(MarkovModel):
    """
    A four state Hodgkin-Huxley model with k=e^{av+b} parameters
    """

    n_params = 9
    n_states = 4
    n_state_vars = n_states-1
    GKr_index = 8
    open_state_index = 1
    Erev = calculate_reversal_potential()
    holding_potential = -80

    def get_default_parameters(self):
        return np.array([2.07E-3, 7.17E-2, 3.44E-5, 6.18E-2,
                         20, 2.58E-2, 2, 2.51E-2, 3.33E-2])

    def __init__(self, protocol, times=None):
        # Create symbols for symbolic functions
        symbols = self.CreateSymbols()

        # Two params for each rate constant, one for the maximal conductance
        k = se.symbols('k1, k2, k3, k4')

        # Define system equations and initial conditions
        k1 = symbols['p'][0] * se.exp(symbols['p'][1] * symbols['v'])
        k2 = symbols['p'][2] * se.exp(-symbols['p'][3] * symbols['v'])
        k3 = symbols['p'][4] * se.exp(symbols['p'][5] * symbols['v'])
        k4 = symbols['p'][6] * se.exp(-symbols['p'][7] * symbols['v'])

        # Notation is consistent between the two papers
        A = se.Matrix([[-k1 - k3 - k4, k2 - k4, -k4],
                    [k1, -k2 - k3, k4], [-k1, k3 - k1, -k2 - k4 - k1]])
        B = se.Matrix([k4, 0, k1])

        if times is None:
            times = np.linspace(0, 15000, 1000)

        # Call the constructor of the parent class, MarkovModel
        super().__init__(symbols, A, B, times,
                            voltage=protocol)

    def CreateSymbols(self):
        """
        Create SymEngine symbols to contain the parameters, state variables and the voltage.
        These are used to generate functions for the right hand side and Jacobian
        """
        # Create parameter symbols
        p = se.Matrix([se.symbols('p%d' % j) for j in range(self.n_params)])
        # Create state variable symbols
        y = se.Matrix([se.symbols('y%d' % i) for i in range(self.n_state_vars)])
        # Create voltage symbol
        v = se.symbols('v')
        return {'p' : p, 'y' : y, 'v' : v}
