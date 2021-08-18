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
        mc = MarkovChain()
        parameters = ['k{}'.format(i) for i in [1,2,3,4]]
        mc.add_symbols(parameters)
        states = [('O', True) , ('C', False), ('I', False), ('IC', False)]
        mc.add_states(states)

        rates = [('O', 'C', 'k2', 'k1'), ('I', 'IC', 'k1', 'k2'), ('IC', 'I', 'k1', 'k2'), ('O', 'I', 'k3', 'k4'), ('C', 'IC', 'k3', 'k4')]

        for r in rates:
            mc.add_both_rates(*r)

        pos=nx.spring_layout(mc.graph)

        # nx.draw(mc.graph, pos, with_labels=True)
        # nx.draw_networkx_edge_labels(mc.graph, pos, edge_labels = nx.get_edge_attributes(mc.graph, 'rate'))
        # Output file

        nx.drawing.nx_agraph.write_dot(mc.graph, "dotfile.dot")
        print(mc.graph)
        print(mc.get_transition_matrix())

if __name__ == "__main__":
    # logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    unittest.main()
