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

        tolerances = (1e-8, 1e-8)
        channel_model = make_model_of_class(model_name, times, voltage_func,
                                            protocol_description=desc,
                                            tolerances=tolerances)

        fig = plt.figure()
        ax = fig.subplots()

        sensitivities_model = SensitivitiesMarkovModel(channel_model)

        voltages = np.array([voltage_func(t) for t in times])
        p = sensitivities_model.get_default_parameters()
        res = sensitivities_model.make_hybrid_solver_states(hybrid=False, njitted=False)()

        ax.plot(times, res[:, :2])
        fig.savefig(os.path.join(self.output_dir, 'sensitivity_state_output'))

        Erev = markovmodels.utilities.calculate_reversal_potential()

        ax.cla()
        ax.plot(times, res[:, 0] * res[:, 1] * (voltages - Erev))
        fig.savefig(os.path.join(self.output_dir, 'sensitivity_current_output'))
        ax.cla()

        I_Kr_sens = sensitivities_model.auxiliary_function(res.T, p, voltages)[:, 0, :].T

        mk_model = markov_builder.models.thirty_models.model_03().generate_myokit_model()
        voltages = np.array([voltage_func(t) for t in times])
        mk_protocol = myokit.TimeSeriesProtocol(times, voltages)

        mk_param_labels = ['markov_chain.' + lab for lab in channel_model.get_parameter_labels()]

        mk_simulation = myokit.Simulation(mk_model, mk_protocol,
                                          sensitivities=(['markov_chain.state_C'],
                                                         mk_param_labels))
        mk_simulation.set_max_step_size(dtmax=100)
        mk_simulation.set_tolerance(*tolerances)
        mk_simulation.pre(1e5)

        d, e = mk_simulation.run(mk_protocol.times()[-1]+1, log_times=times)
        sens = np.array(e)
        mk_S = sens.reshape(sens.shape[0], -1, order='C')
        error = I_Kr_sens - mk_S

        # Plot bot sensitivities
        ax.plot(times, I_Kr_sens[:, 0])
        ax.plot(times, mk_S[:, 0])

        fig.savefig(os.path.join(self.output_dir, 'first_sensitivitity_compare'))
        ax.cla()

        plot_labels = [f"dI/d{lab}" for lab in channel_model.get_parameter_labels()]
        fig.clf()
        axs = fig.subplots(2)

        mk_IKr = d['markov_chain.I_Kr']
        axs[0].plot(times, mk_IKr, label='I_Kr')
        axs[0].legend()
        axs[1].plot(times, mk_S, label=plot_labels)
        plt.close(fig)
        ax.legend()
        fig.savefig(os.path.join(self.output_dir, 'model3_sensitivities_mk'))

        self.assertLess(np.sum(error**2), 1e-1)

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


