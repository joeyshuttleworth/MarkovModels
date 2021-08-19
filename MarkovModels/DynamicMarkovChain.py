from . MarkovChain import MarkovChain
from typing import Union, Tuple
import sympy as sp
import logging

import networkx as nx
import pandas as pd
import numpy as np
import scipy

from numpy.random import default_rng

class DynamicMarkovChain(MarkovChain):
    def __init__(self, states : Union[list, None] = None, seed : Union[int, None] = None):
        # Needs to be a MultiDiGraph to visualise properly (??)
        self.graph = nx.DiGraph()
        if states is not None:
            self.graph.add_nodes_from(states)
        self.rates = []
        self.rng = default_rng(seed)

    def get_embedded_chain(self,  rate_values : dict):
        raise NotImplementedError

    def sample_trajectories(self, no_trajectories : int, rate_functions : dict, time_range : list=[0,1], starting_distribution : Union[list, None] = None):
        no_nodes = len(self.graph.nodes)
        logging.debug("There are {} nodes".format(no_nodes))

        if starting_distribution is None:
            starting_distribution = np.around(np.array([no_trajectories]*no_nodes)/no_nodes)
            starting_distribution[0] += no_trajectories - starting_distribution.sum()

        print("starting distribution is {}".format(starting_distribution))
        distribution = starting_distribution
        _, Q = self.get_transition_matrix()
        Q_evaled = sp.lambdify(self.rates, Q)

        def eval_Q(t : float):
            params = [rate_functions[rate](t) for rate in self.rates]
            return sp.lambdify(self.rates, Q)(*params)

        t=0
        data = [(t, *distribution)]
        while True:
            waiting_times = np.zeros([Q.shape[0],1])
            for state_index, s_i in enumerate(distribution):
                if s_i == 0:
                    waiting_times[state_index] = np.inf
                    continue
                # Compute waiting time
                rand_u = self.rng.uniform()
                def lmda(t : float):
                    params = [rate_functions[rate](t) for rate in self.rates]
                    return -sp.lambdify(self.rates, Q[state_index, state_index])(*params)
                f = lambda t, x : lmda(t) * ( 1 - x)
                event = lambda t,x : rand_u - (1-x[0])**s_i
                event.terminal=True
                sol=scipy.integrate.solve_ivp(f, [0,1000], [0], events=event, atol=1e-1, rtol=1e-3)
                waiting_times[state_index]=sol.t[-1]

            if t+min(waiting_times) > time_range[1]:
                break

            new_t= t+min(waiting_times)
            if new_t == t:
                logging.warning("Underflow warning: timestep too small {}".format(min(waiting_times)))
            t = new_t

            state_to_jump = list(waiting_times).index(min(waiting_times))

            # Find what state we will jump to
            rand = self.rng.uniform()

            sum=0
            culm_row = eval_Q(new_t)[state_index,:]
            for i,val in enumerate(culm_row):
                culm_row[i]=val+sum
                sum+=val

            jump_to = next(i for i, x in enumerate(culm_row) if rand < x)

            distribution[state_to_jump] -= 1
            distribution[jump_to] += 1

            data.append((t, *distribution))

            df =  pd.DataFrame(data, columns=['time', *self.graph.nodes])
        return df



