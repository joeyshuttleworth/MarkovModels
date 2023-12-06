import networkx
import numpy as np
import pints
import regex as re
import sympy as sp

from markov_builder import MarkovChain
from markovmodels.DisconnectedMarkovModel import DisconnectedMarkovModel
from markovmodels.MarkovModel import MarkovModel


def make_myokit_model(model_name: str):

    from markov_builder.models.thirty_models import (
        model_00,
        model_01,
        model_02,
        model_03,
        model_04,
        model_05,
        model_06,
        model_07,
        model_08,
        model_09,
        model_10,
        model_11,
        model_12,
        model_13,
        model_14,
        model_30,
    )

    thirty_models_regex = re.compile(r'^model([0-9]*)$')

    thirty_models = [
        model_00, model_01, model_02, model_03, model_04,
        model_05, model_06, model_07, model_08, model_09, model_10,
        model_11, model_12, model_13, model_14, model_30
    ]
    if thirty_models_regex.match(model_name):
        model_no = int(thirty_models_regex.search(model_name).group(1))
        mk_model = thirty_models[model_no]().generate_myokit_model()

    return mk_model


def make_model_of_class(name: str, times=None, voltage=None, *args, **kwargs):
    from markov_builder.models.thirty_models import (
        model_00,
        model_01,
        model_02,
        model_03,
        model_04,
        model_05,
        model_06,
        model_07,
        model_08,
        model_09,
        model_10,
        model_11,
        model_12,
        model_13,
        model_14,
        model_30,
    )

    thirty_models = [
        model_00, model_01, model_02, model_03, model_04,
        model_05, model_06, model_07, model_08, model_09, model_10,
        model_11, model_12, model_13, model_14, model_30
    ]

    thirty_models_regex = re.compile(r'^model([0-9]+)$')

    if name == 'Beattie' or name == 'BeattieModel':
        from markovmodels.BeattieModel import BeattieModel
        model = BeattieModel(times, voltage, *args, **kwargs)
    elif name == 'Kemp' or name == 'KempModel':
        from markovmodels.KempModel import KempModel
        model = KempModel(times, voltage, *args, **kwargs)
    elif name == 'CO' or name == 'ClosedOpenModel':
        from markovmodels.ClosedOpenModel import ClosedOpenModel
        model = ClosedOpenModel(times, voltage, *args, **kwargs)
    elif name == 'Wang' or name == 'WangModel':
        from markovmodels.WangModel import WangModel
        model = WangModel(times, voltage, *args, **kwargs)
    elif thirty_models_regex.match(name):
        model_no = int(thirty_models_regex.search(name).group(1))
        model = generate_markov_model_from_graph(thirty_models[model_no](), times, voltage,
                                                 *args, **kwargs)
    else:
        assert False, f"no model with name {name}"
    return model


def generate_markov_model_from_graph(mc: MarkovChain, times, voltage,
                                     log_transform=True, *args, **kwargs):

    if 'default_parameters' not in kwargs:
        kwargs['default_parameters'] = np.array([val
                                                 for key, val in mc.default_values.items()
                                                 if (str(key) not in ['E_Kr', 'E_rev', 'V'] and val is not None)])

    parameter_labels = [key
                        for key, val in mc.default_values.items()
                        if str(key) not in ['E_Kr', 'E_rev', 'V'] and val is not None]

    if len(parameter_labels) != len(kwargs['default_parameters']):
        raise Exception("Parameter labels length is not equal to parameter vector length"
                        f"({len(parameter_labels)} vs {len(kwargs['default_parameters'])})")

    state_labels = list(mc.graph)

    symbols = {}
    symbols['v'] = sp.sympify('V')
    symbols['p'] = sp.Matrix([sp.sympify(p) for p in parameter_labels])

    GKr_index = len(parameter_labels) - 1

    if log_transform:
        transformations = [[pints.LogTransformation(1),
                            pints.IdentityTransformation(1)] \
                           for i, rate in enumerate(mc.rate_expressions)\
                           if 2*i < len(parameter_labels) - 1]
        # flatten
        transformations = [t for pair in transformations for t in pair]
        # append identity for conductance parameters
        transformations += [pints.IdentityTransformation(1)]

    if mc.is_connected():
        # Graph is connected, generate MarkovModel
        open_state_index = state_labels.index('O')
        label_order = (['O'] + [s for s in state_labels if s != 'O'])
        remaining_states = label_order[:-1]
        A, B = mc.eliminate_state_from_transition_matrix(remaining_states)
        state_labels, Q = mc.get_transition_matrix(label_order=label_order)
        symbols['y'] = sp.Matrix([mc.get_state_symbol(s)
                                  for s in remaining_states])

        return MarkovModel(symbols, A, B, mc.rate_expressions, times=times,
                           voltage=voltage, Q=Q, *args, **kwargs, name=mc.name,
                           parameter_labels=parameter_labels,
                           GKr_index=GKr_index,
                           open_state_index=open_state_index,
                           state_labels=state_labels,
                           transformations=transformations)
    else:
        # Graph is disconnected: generated DisconnectedMarkovModel
        comps = [list(comp) for comp in networkx.connected_components(mc.graph.to_undirected())]

        auxiliary_states = [str(s) for s in mc.auxiliary_expression.free_symbols\
                            if len(str(s)) > 6]
        auxiliary_states = [s[6:] for s in auxiliary_states if s[:6] == 'state_']

        labels, Q = mc.get_transition_matrix()

        Qs = []
        As = []
        Bs = []
        ys = []

        for comp in comps:
            state_indices = [labels.index(state) for state in comp]

            sub_Q = Q[state_indices, state_indices]
            Qs.append(sub_Q)

            subset_labels = [labels[i] for i in state_indices]

            eliminated_state = [state for state in comp if state not in auxiliary_states][0]

            # Move eliminated state to the last state
            labels_without_state = [lab for lab in subset_labels if lab != eliminated_state]
            permutation = [list(comp).index(n) for n in \
                           labels_without_state + [eliminated_state]]

            y = [mc.get_state_symbol(lab) for lab in labels_without_state]

            sub_Q = sub_Q[permutation, permutation]
            matrix = sub_Q.T
            shape = matrix.shape

            M = sp.eye(shape[0])
            replacement_row = np.full(shape[0], -1)
            M[-1, :] = replacement_row[None, :]

            A_matrix = matrix @ M
            B_vec = matrix @ sp.Matrix([[0] * (shape[0] - 1) + [1]]).T

            A_matrix = A_matrix[0:-1, 0:-1]
            B_vec = B_vec[0:-1, :]

            As.append(A_matrix)
            Bs.append(B_vec)
            ys.append(y)

        # state_symbols = [mc.get_state_symbol(lab) for lab in labels]
        remaining_states = [lab for lab in state_labels if lab != eliminated_state]

        A = sp.Matrix.zeros(sum([len(y) for y in ys]))[:, :]
        c = 0
        for sub_A in As:
            A[c:c + sub_A.shape[0], c:c + sub_A.shape[1]] = sub_A[:, :]
            c += sub_A.shape[0]

        B = sp.Matrix.vstack(*Bs)

        symbols['y'] = sp.Matrix([var for y in ys for var in y])
        return DisconnectedMarkovModel(symbols, A, B, Qs, As, Bs, ys, comps,
                                       mc.rate_expressions, times,
                                       parameter_labels,
                                       mc.auxiliary_expression, *args,
                                       transformations=transformations,
                                       name=mc.name, GKr_index=len(parameter_labels) - 1,
                                       voltage=voltage, **kwargs,
                                       state_labels=state_labels)
