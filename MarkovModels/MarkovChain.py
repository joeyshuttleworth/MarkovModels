from typing import Union
import sympy as sp
import logging

import networkx as nx
import numpy as np

class MarkovChain():
    def __init__(self, states : Union[list, None] = None):
        # Needs to be a MultiDiGraph to visualise properly (??)
        self.graph = nx.DiGraph()
        if states is not None:
            self.graph.add_nodes_from(states)
        self.rates = []

    def add_state(self, label : Union[str, None], open : bool = False):
        if label is None:
            # TODO Generate new label name
            label='undefined'
        self.graph.add_node(label, open=open)

    def add_states(self, states : list):
        for state in states:
            self.add_state(*state)

    def add_rate(self, rate : str):
        if rate in self.rates:
            # TODO
            raise Exception()
        else:
            self.rates = self.rates + [rate]

    def add_rates(self, rates : list):
        for rate in rates:
            self.add_rate(rate)

    def add_both_transitions(self, frm: str, to : str, fwd_rate : Union[str, sp.Expr, None], bwd_rate : Union[str, sp.Expr, None]):
        self.add_transition(frm, to, fwd_rate)
        self.add_transition(to, frm, bwd_rate)

    def add_transition(self, from_node : str, to_node : str, transition_rate : Union[str, sp.Expr, None]):
        # TODO: Nice exceptions

        if from_node not in self.graph.nodes or to_node not in self.graph.nodes:
            raise Exception("A node wasn't present in the graph ({} or {})".format(from_node, to_nodes))

        if not isinstance(transition_rate, sp.Expr):
            transition_rate = sp.sympify(transition_rate)

        # First check that all of the symbols in sp.expr are defined (if it exists)
        print(transition_rate)
        if transition_rate is not None:
            for expr in transition_rate.free_symbols:
                if str(expr) not in self.rates:
                    raise Exception("symbol {} is not in the symbols list".format(expr))

        self.graph.add_edge(from_node, to_node, rate=transition_rate)

    def get_transition_matrix(self):
        """
        Returns a pair : labels, and the transition matrix
        """
        edges = self.graph.edges

        matrix=[]
        for current_state in self.graph.nodes:
            row=[]
            # Get edges incident to the state
            for incident_state in self.graph.nodes:
                if current_state == incident_state:
                    row.append(0)
                else:
                    edge = self.graph.get_edge_data(current_state, incident_state)
                    if edge is not None:
                        row.append(edge["rate"])
                    else:
                        row.append(0)
            matrix.append(row)

        matrix = sp.Matrix(matrix)
        # Compute diagonals
        n = matrix.shape[0]
        for i in range(n):
            matrix[i, i] = -sum(matrix[i,:])

        return self.graph.nodes, matrix

    def eliminate_state_from_transition_matrix(self, labels : Union[list, None] = None):
        """eliminate_state_from_transition_matrix

        Because the state occupancy probabilities must add up to zero, the
        transition matrix is always singular. We can use this fact to remove
        one state variable from the system of equations. The labels parameter
        allows you to choose which variable is eliminated and also the ordering
        of the states.

        params:

        labels: A list of labels. This must be one less than the number of
        states in the model. The order of this list determines the ordering of
        the state variable in the outputted dynamical system.

        returns:

        Returns a pair of symbolic matrices, A & B, defining a system of ODEs of the format dX/dt = AX + B.
        """
        _, matrix = self.get_transition_matrix()
        matrix=matrix.T
        shape = sp.shape(matrix)
        assert(shape[0] == shape[1])

        if labels is None:
            labels = self.graph.nodes[:-1]

        # List describing the mapping from self.graph.nodes to labels.
        # permutation[i] = j corresponds to a mapping which takes
        # graph.nodes[i] to graph.nodes[j]. Map the row to be eliminated to the
        # end.
        permutation = [labels.index(n) if n in labels else shape[0]-1 for n in self.graph.nodes]

        matrix = matrix[permutation,permutation]

        M = sp.eye(shape[0])
        replacement_row = np.array([-1 for i in range(shape[0])])[None,:]

        print(M, replacement_row)
        M[-1,:] = replacement_row

        matrix = M @ matrix
        print(matrix)

        # Construct vector
        vec = sp.Matrix([0 for i in range(shape[0])])
        for j, el in enumerate(matrix[:,-1]):
            if el != 0:
                vec[j,0] = el
                for i in range(shape[0]):
                    matrix[j, i] -= el


        return matrix[0:-1,0:-1], vec[0:-1,:]
