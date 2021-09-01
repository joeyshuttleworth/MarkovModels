#!/usr/bin/env python3

import sys
import os
import logging
import unittest
import MarkovModels
import sympy as sp
import networkx as nx
from MarkovModels import MarkovChain
from MarkovModels import BeattieModel

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class TestMarkovChain(unittest.TestCase):
    def setUp(self):
        test_output_dir = os.environ.get('MARKOVMODELS_TEST_OUTPUT', os.path.join(os.path.dirname(os.path.realpath(__file__)), self.__class__.__name__))
        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)
        self.output_dir = test_output_dir
        logging.info("outputting to " + test_output_dir)

    def test_construct_chain(self):

        logging.info("Constructing four-state Beattie model")
        mc = MarkovChain()
        rates = ['k{}'.format(i) for i in [1,2,3,4]]
        mc.add_rates(rates)
        states = [('O', True) , ('C', False), ('I', False), ('IC', False)]
        mc.add_states(states)

        rates = [('O', 'C', 'k2', 'k1'), ('I', 'IC', 'k1', 'k2'), ('IC', 'I', 'k1', 'k2'), ('O', 'I', 'k3', 'k4'), ('C', 'IC', 'k3', 'k4')]

        for r in rates:
            mc.add_both_transitions(*r)

        pos=nx.spring_layout(mc.graph)

        # Output DOT file
        nx.drawing.nx_agraph.write_dot(mc.graph, "dotfile.dot")

        logging.debug(mc.graph)

        labels, Q = mc.get_transition_matrix()
        logging.debug("Q^T matrix is {}, labels are {}".format(Q.T, labels))

        system = mc.eliminate_state_from_transition_matrix(['C', 'O', 'I'])

        pen_and_paper_A = sp.Matrix([['-k1 - k3 - k4', 'k2 - k4', '-k4'],
                    ['k1', '-k2 - k3', 'k4'], ['-k1', 'k3 - k1', '-k2 - k4 - k1']])

        pen_and_paper_B = sp.Matrix(['k4', 0, 'k1'])

        self.assertEqual(pen_and_paper_A, system[0])
        self.assertEqual(pen_and_paper_B, system[1])


    def test_SimulateStepProtocol(self):
        mc = MarkovChain()
        rates = ['k{}'.format(i) for i in [1,2,3,4]]
        mc.add_rates(rates)
        states = [('O', True) , ('C', False), ('I', False), ('IC', False)]
        mc.add_states(states)

        rates = [('O', 'C', 'k2', 'k1'), ('I', 'IC', 'k1', 'k2'), ('IC', 'I', 'k1', 'k2'), ('O', 'I', 'k3', 'k4'), ('C', 'IC', 'k3', 'k4')]

        for r in rates:
            mc.add_both_transitions(*r)

        params = BeattieModel().get_default_parameters()
        def get_rates(voltage : float, parameters : list):
            # Now get the waiting times and embedded MC
            rates=[params[2*i]+np.exp(params[2*i+1]*voltage) for i in range(int((len(params)-1)/2))]

            rate_vals = {"k1" : rates[0],
                    "k2" : rates[1],
                    "k3" : rates[2],
                    "k4" : rates[3]
            }
            return rate_vals

        protocol = ((-80, 100), (20, 200))
        # protocol=((-80,1000),)

        fig = plt.figure(figsize=(8,8))
        ax1 = fig.add_subplot(211)
        no_trajectories = 1000
        dist=None
        data = [pd.DataFrame(columns=("time", *mc.graph.nodes))]
        last_time=0
        eqm_data = []
        for voltage, time_to in protocol:
            data.append(mc.sample_trajectories(no_trajectories, get_rates(voltage, params), (last_time, time_to), starting_distribution=dist))
            dist = data[-1].values[-1,1:]
            _,A = mc.eval_transition_matrix(get_rates(voltage,params))
            model=BeattieModel()
            # compute steady states
            labels, ss = mc.get_equilibrium_distribution(get_rates(voltage, params))
            ss = ss*no_trajectories
            eqm_data=eqm_data + [[last_time, *ss]] + [[time_to, *ss]]
            print("Transition rates should be {}".format(A))
            last_time = time_to

        eqm_data = pd.DataFrame(eqm_data, columns = ['time'] + [l + ' eqm distribution' for l in labels]).set_index("time")
        data=pd.concat(data).set_index("time").sort_index()

        data.plot(ax=ax1)
        eqm_data.plot(style="--", ax=ax1)

        ax2= fig.add_subplot(212)

        model = BeattieModel()

        # Need each voltage twice - at the beginning and end of each step
        voltages = [[v, v] for v,_ in protocol]
        voltages = [v for voltage in voltages for v in voltage]
        times=[0]
        for _, time_to in protocol:
            times=times+[time_to]*2
        times=times[0:-1]
        ax2.plot(times,voltages)
        plt.savefig(os.path.join(self.output_dir, "SimulateStepProtocol.pdf"))


if __name__ == "__main__":
    # logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    unittest.main()
