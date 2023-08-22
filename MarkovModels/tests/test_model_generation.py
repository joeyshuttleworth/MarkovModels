#!/usr/bin/env python3

import logging
import os
import unittest

import matplotlib.pyplot as plt
import numpy as np

from MarkovModels import common


class TestModelGeneration(unittest.TestCase):

    def setUp(self):
        test_output_dir = common.setup_output_directory('test_output', 'test_model_generation')
        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)
        self.output_dir = test_output_dir
        logging.info("outputting to " + test_output_dir)

    def test_model16_solver_error(self):

        protocol = 'staircaseramp1'

        voltage_func, times, desc = common.get_ramp_protocol_from_csv(protocol)
        model = common.make_model_of_class('model14', times, voltage_func, protocol_description=desc)

        full_parameters = model.get_default_parameters()

        boundaries = common.fitting_boundaries(full_parameters, model)

        n_samples = 50
        sampled_parameter_sets = boundaries.sample(n=5000)

        hsolver = model.make_hybrid_solver_current(p_cond_threshold=np.inf)
        solver = model.make_forward_solver_current()

        atol = 1e-9
        rtol = 1e-9

        max_error = 0
        for i, p in enumerate(sampled_parameter_sets):
            hres = hsolver(p, atol=atol, rtol=rtol)
            res = solver(p, atol=atol, rtol=rtol)

            error = np.max((hres - res)**2)
            if error > max_error:
                max_error = error
                max_error_index = i

        # Plot max error
        plt.plot(times,
                 hsolver(sampled_parameter_sets[max_error_index], atol=atol, rtol=rtol) - \
                 solver(sampled_parameter_sets[max_error_index], atol=atol, rtol=rtol),
                 label='hybrid')

        plt.savefig(os.path.join(self.output_dir, "model_16_max_solver_error"))
        plt.clf()

        plt.plot(times, solver(sampled_parameter_sets[max_error_index], atol=atol, rtol=rtol),
                 'forward solver')
        plt.plot(times, hsolver(sampled_parameter_sets[max_error_index], atol=atol, rtol=rtol),
                 'hybrid solver')

        plt.savefig(os.path.join(self.output_dir, "model_16_solver_comparison"))
        plt.clf()

    # def test_hybrid_solver(self):

    #     model_names = [f"model{i}" for i in range(15)]
    #     protocol = 'staircaseramp'
    #     voltage_func, times, desc = common.get_ramp_protocol_from_csv(protocol)

    #     models = [common.make_model_of_class(m, times, voltage_func, protocol_description=desc)\
    #               for m in model_names]

    #     for name, model in zip(model_names, models):
    #         output1 = model.make_forward_solver_current(njitted=False)()
    #         output2 = model.make_hybrid_solver_current(njitted=False)()

    #         plt.plot(times, output1)
    #         plt.plot(times, output2)
    #         plt.savefig(os.path.join(self.output_dir, f"{name} solver comparison"))
    #         plt.clf()

    #         self.assertLess(np.sqrt(np.mean((output1 - output2)**2)), 1e-1)

    # def test_generated_model_output(self):
    #     protocol = 'staircaseramp'

    #     for original_model, generated_model in [['Beattie', 'model3']]:
    #         voltage_func, times, desc = common.get_ramp_protocol_from_csv(protocol)

    #         model1 = common.make_model_of_class(original_model, times, voltage_func,
    #                                             protocol_description=desc)
    #         model2 = common.make_model_of_class(generated_model, voltage=voltage_func,
    #                                             times=times, protocol_description=desc)

    #         default_parameters = model1.get_default_parameters()

    #         h_solver1 = model1.make_forward_solver_current()
    #         h_solver2 = model2.make_forward_solver_current()

    #         output1 = h_solver1(default_parameters)
    #         output2 = h_solver2(default_parameters)

    #         rmse_error = np.sqrt(((output1 - output2)**2).mean())

    #         logging.debug('rmse error is: ' + f"{rmse_error}")

    #         plt.plot(times, output1, label=original_model)
    #         plt.plot(times, output2, label='MarkovBuilder model')
    #         plt.savefig(os.path.join(self.output_dir, f"{original_model}_{generated_model}_comparison_lsoda.pdf"))
    #         plt.clf()

    #         h_solver1 = model1.make_hybrid_solver_current()
    #         h_solver2 = model2.make_hybrid_solver_current()

    #         output1 = h_solver1(default_parameters)
    #         output2 = h_solver2(default_parameters)

    #         rmse_error = np.sqrt(((output1 - output2)**2).mean())

    #         logging.debug('rmse error is: ' + f"{rmse_error}")

    #         plt.plot(times, output1, label=original_model)
    #         plt.plot(times, output2, label='MarkovBuilder model')

    #         plt.savefig(os.path.join(self.output_dir, f"{original_model}_{generated_model}_comparison_hybrid.pdf"))
    #         plt.clf()

    #         self.assertLess(rmse_error, 1e-5)

