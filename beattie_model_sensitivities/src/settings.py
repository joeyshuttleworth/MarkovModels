class Params(object):
    def __init__(self):

        '''Fixed parameter settings used for simulations'''
        # Define number of parameters and variables
        self.n_state_vars = 3
        self.n_params = 9
        self.open_state = 1

        self.tmax = 8000
        self.timestep = 1
        self.Erev = -88
        self.GKr_index = -1

        self.conductance_index = 8

        # Relative and absolute tolerances to solve the system with, [rtol, atol]
        self.solver_tolerances = [1e-8, 1e-8]

        # The value that the membrane potential is clamped too before the
        # protocol is applied (mV)
        self.holding_potential = -80
