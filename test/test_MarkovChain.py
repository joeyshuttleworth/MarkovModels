#!/usr/bin/env python3

import sys
import logging
import unittest
import MarkovModels
import sympy as sp
import networkx as nx
from MarkovModels import MarkovChain

import matplotlib.pyplot as plt

class TestMarkovChain(unittest.TestCase):
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
        print(system)

        pen_and_paper_A = sp.Matrix([['-k1 - k3 - k4', 'k2 - k4', '-k4'],
                    ['k1', '-k2 - k3', 'k4'], ['-k1', 'k3 - k1', '-k2 - k4 - k1']])

        pen_and_paper_B = sp.Matrix(['k4', 0, 'k1'])

        self.assertEqual(pen_and_paper_A, system[0])
        self.assertEqual(pen_and_paper_B, system[1])


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    unittest.main()
