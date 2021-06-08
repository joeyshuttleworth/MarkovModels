
class Params(object):
    def __init__(self):

        '''Fixed parameter settings used for simulations'''
        # Define number of parameters and variables
        self.n_state_vars = 3
        self.n_params = 9
        self.open_state = 2

        self.tmax = 8000
        self.Erev = -88
        self.GKr = 0.1524

        # Relative and absolute tolerances to solve the system with, [rtol, atol]
        self.solver_tolerances = [1e-6, 1e-6]
