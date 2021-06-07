
class Params(object):
    def __init__(self):

        '''Fixed parameter settings used for simulations'''
        # Define number of parameters and variables
        self.n_state_vars = 3
        self.n_params = 9
        # The index of the open state
        self.open_state = 2

        self.tmax = 8000

        # The resting Nernst potential
        self.Erev = -88

        # The conductance of an open hERG channel
        self.GKr = 0.1524
