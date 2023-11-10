import numpy as np
import pandas as pd
import markovmodels.voltage_protocols


def D_opt_utility(desc, params, s_model, hybrid=False, crhs=None, indices=None):
    """ Evaluate the D-optimality of design, d for a certain parameter vector"""
    s_model.protocol_description = desc
    s_model.voltage = markovmodels.voltage_protocols.make_voltage_function_from_description(desc)
    # output = model.make_hybrid_solver_current(njitted=False, hybrid=hybrid)

    sens = s_model.make_hybrid_solver_states(njitted=False, hybrid=False, crhs=crhs)()
    voltages = np.array([s_model.voltage(t) for t in s_model.times])
    I_Kr_sens = s_model.auxiliary_function(sens.T, params, voltages)[:, 0, :].T

    if indices is not None:
        I_Kr_sens = I_Kr_sens[indices]

    return np.log(np.linalg.det(I_Kr_sens.T @ I_Kr_sens))
