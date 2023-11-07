import logging
import os
import unittest

import matplotlib.pyplot as plt
import myokit
import numpy as np
import pandas as pd
import seaborn as sns
import sympy as sp
from numba import njit
import markov_builder

import markovmodels
from markovmodels.utilities import setup_output_directory, calculate_reversal_potential
from markovmodels.voltage_protocols import get_ramp_protocol_from_csv
from markovmodels.model_generation import make_model_of_class, make_myokit_model
from markovmodels.SensitivitiesMarkovModel import SensitivitiesMarkovModel


class TestSensitivities(unittest.TestCase):

    def setUp(self):
        test_output_dir = setup_output_directory('test_output', 'test_sensitivities')
        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)
        self.output_dir = test_output_dir

        self.model_names = [f"model{i}" for i in range(15)]
        logging.info("outputting to " + test_output_dir)

    def test_compare_with_myokit(self):
        model_name = 'model3'
        protocol = 'staircaseramp1'
        voltage_func, times, desc = get_ramp_protocol_from_csv(protocol)
        tolerances = 1e-10, 1e-10
        channel_model = make_model_of_class(model_name, times, voltage_func,
                                            protocol_description=desc,
                                            tolerances=tolerances)

        sensitivities_model = SensitivitiesMarkovModel(channel_model)

        voltages = np.array([voltage_func(t) for t in times])
        p = sensitivities_model.get_default_parameters()
        res = sensitivities_model.make_hybrid_solver_states(hybrid=False, njitted=False)()
        I_out_sens = sensitivities_model.auxiliary_function(res.T, p, voltages)[:, 0, :].T

        plt.plot(times, I_out_sens, label=channel_model.get_parameter_labels())
        plt.savefig(os.path.join(self.output_dir, 'model3_sensitivities'))

        mk_model = markov_builder.models.thirty_models.model_03().generate_myokit_model()
        voltages = np.array([voltage_func(t) for t in times])
        mk_protocol = myokit.TimeSeriesProtocol(times, voltages)

        mk_param_labels = ['markov_chain.' + lab for lab in channel_model.get_parameter_labels()]

        mk_simulation = myokit.Simulation(mk_model, mk_protocol,
                                          sensitivities=(['markov_chain.state_C'],
                                                         mk_param_labels))
        mk_simulation.set_tolerance(1e-12, 1e-12)
        mk_simulation.set_max_step_size(dtmax=100)
        mk_simulation.pre(1e3)

        d, e = mk_simulation.run(mk_protocol.times()[-1]+1, log_times=times)

        print(e)

        sens = np.array(e)

        S = sens.reshape(sens.shape[0], -1, order='C')
        print(S)

        d.save_csv(os.path.join(self.output_dir, 'model_3_staircase_sensitivities_log.csv'))

    def test_solver_sensitivities(self):
        protocol = 'staircaseramp1'
        voltage_func, times, desc = get_ramp_protocol_from_csv(protocol)

        channel_model = make_model_of_class('model3', times,
                                            voltage_func,
                                            protocol_description=desc)

        sensitivities_model = SensitivitiesMarkovModel(channel_model)

        res = sensitivities_model.make_hybrid_solver_states(hybrid=False, njitted=False)()

        voltages = np.array([voltage_func(t) for t in times])
        p = sensitivities_model.get_default_parameters()
        I_out_sens = sensitivities_model.auxiliary_function(res.T, p, voltages)[:, 0, :].T

        plt.plot(times, I_out_sens, label=channel_model.get_parameter_labels())
        plt.savefig(os.path.join(self.output_dir, 'model3_sensitivities'))


