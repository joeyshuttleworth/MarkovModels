from typing import Union
import networkx as nx
import sympy as sp
import logging
from scipy.integrate import odeint


class MarkovChain():
    def __init__(self, states : Union[list, None] = None):
        # Needs to be a MultiDiGraph to visualise properly (??)
        self.graph = nx.DiGraph()
        if states is not None:
            self.graph.add_nodes_from(states)
        self.symbols = []

    def add_state(self, label : Union[str, None], open : bool = False):
        if label is None:
            # TODO Generate new label name
            label='undefined'
        self.graph.add_node(label, open=open)

    def add_states(self, states : list):
        for state in states:
            self.add_state(*state)

    def add_symbol(self, symbol : str):
        if symbol in self.symbols:
            # TODO
            raise Exception()
        else:
            self.symbols = self.symbols + [symbol]

    def add_symbols(self, symbols : list):
        for symbol in symbols:
            self.add_symbol(symbol)

    def add_both_rates(self, frm: str, to : str, fwd_rate : Union[str, sp.Expr, None], bwd_rate : Union[str, sp.Expr, None]):
        self.add_rate(frm, to, fwd_rate)
        self.add_rate(to, frm, bwd_rate)

    def add_rate(self, from_node : str, to_node : str, transition_rate : Union[str, sp.Expr, None]):
        # self.graph is of type nx.DiGraph not nx.multigraph so nx won't
        # let us add two edges between the same node

        # TODO: Nice exceptions

        if from_node not in self.graph.nodes or to_node not in self.graph.nodes:
            raise Exception("A node wasn't present in the graph ({} or {})".format(from_node, to_nodes))

        if not isinstance(transition_rate, sp.Expr):
            transition_rate = sp.sympify(transition_rate)

        # First check that all of the symbols in sp.expr are defined (if it exists)
        print(transition_rate)
        if transition_rate is not None:
            for expr in transition_rate.free_symbols:
                if str(expr) not in self.symbols:
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
        return sp.Matrix(matrix)
