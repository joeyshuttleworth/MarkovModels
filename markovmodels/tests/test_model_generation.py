#!/usr/bin/env python3

import logging
import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import pandas as pd
import seaborn as sns

import myokit

from numba import njit

from markovmodels import common


class TestModelGeneration(unittest.TestCase):

    def setUp(self):
        test_output_dir = common.setup_output_directory('test_output', 'test_model_generation')
        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)
        self.output_dir = test_output_dir

        self.model_names = [f"model{i}" for i in range(15)]
        logging.info("outputting to " + test_output_dir)

    def test_model16_solver_error(self):

        protocol = 'staircaseramp1'
        voltage_func, times, desc = common.get_ramp_protocol_from_csv(protocol)
        voltages = [vstart for vstart, vend, _, _ in desc if vstart != vend]

        tolerances = 1e-10, 1e-10

        model = common.make_model_of_class('model14', times, voltage_func, protocol_description=desc,
                                           tolerances=tolerances)

        full_parameters = model.get_default_parameters()

        boundaries = common.fitting_boundaries(full_parameters, model)

        n_samples = 1000
        sampled_parameter_sets = boundaries.sample(n=n_samples)

        hsolver = model.make_hybrid_solver_current(cond_threshold=np.inf, njitted=False)
        solver = model.make_forward_solver_current(njitted=False)

        rates_func = model.get_rates_func(njitted=True)
        A_func = njit(sp.lambdify((model.rates_dict.keys(),), model.A), fastmath=True)

        def P_cond_func(p, v):
            rates = rates_func(p, v).flatten()
            A = A_func(rates)
            if not np.all(np.isfinite(A)):
                return 0

            _, P = np.linalg.eig(A)
            try:
                return np.linalg.norm(P, 2) * np.linalg.norm(np.linalg.inv(P), 2)
            except np.linalg.LinAlgError:
                return 0

        def A_cond_func(p, v):
            rates = rates_func(p, v).flatten()
            A = A_func(rates)
            if not np.all(np.isfinite(A)):
                return 0
            try:
                return np.linalg.norm(A, 2) * np.linalg.norm(np.linalg.inv(A), 2)

            except np.linalg.LinAlgError:
                return 0

        rows = []
        for i, p in enumerate(sampled_parameter_sets):
            hres = hsolver(p)
            res = solver(p)

            P_conds = [P_cond_func(p, v) for v in voltages]
            A_conds = [A_cond_func(p, v) for v in voltages]
            error = np.max((hres - res)**2)

            rows.append([max(A_conds), max(P_conds), error])

        results_df = pd.DataFrame(rows, columns=['max_A_cond', 'max_P_cond', 'max_error'])
        results_df.to_csv(os.path.join(self.output_dir, 'cond_P_error_table'))

        max_A_cond_i = np.argmax(results_df['max_A_cond'].values)
        max_P_cond_i = np.argmax(results_df['max_A_cond'].values)
        # Plot max error
        plt.plot(times,
                 np.abs(hsolver(sampled_parameter_sets[max_A_cond_i]) -\
                        solver(sampled_parameter_sets[max_A_cond_i])),
                 label='hybrid')
        plt.savefig(os.path.join(self.output_dir, "model_14_max_solver_error"))
        plt.clf()

        max_A_cond = results_df.loc[max_A_cond_i]['max_A_cond']
        max_A_error = results_df.loc[max_A_cond_i]['max_error']
        plt.title(f"max_A_cond {max_A_cond}, max_error={max_A_error}")
        plt.yscale('log')
        plt.plot(times, solver(sampled_parameter_sets[max_A_cond_i]),
                 label='forward solver')
        plt.plot(times, hsolver(sampled_parameter_sets[max_A_cond_i]),
                 label='hybrid solver')
        plt.savefig(os.path.join(self.output_dir, "model_14_solver_comparison_max_A_cond"))
        plt.clf()

        max_P_cond = results_df.loc[max_P_cond_i]['max_P_cond']
        max_P_error = results_df.loc[max_P_cond_i]['max_error']
        plt.title(f"max_P_cond {max_P_cond}, max_error={max_P_error}")
        plt.plot(times, solver(sampled_parameter_sets[max_P_cond_i]),
                 label='forward solver')
        plt.plot(times, hsolver(sampled_parameter_sets[max_P_cond_i]),
                 label='hybrid solver')
        plt.savefig(os.path.join(self.output_dir, "model_14_solver_comparison_max_P_cond"))
        plt.clf()

        max_max_error_i = results_df.max_error.idxmax()
        plt.title(f"max_max_error {results_df.max_error.max()}")
        plt.plot(times, solver(sampled_parameter_sets[max_max_error_i]),
                 label='forward solver')
        plt.plot(times, hsolver(sampled_parameter_sets[max_max_error_i]),
                 label='hybrid solver')
        plt.savefig(os.path.join(self.output_dir, "model_14_solver_comparison_max_max_error"))
        plt.clf()

        splot = sns.scatterplot(data=results_df, x='max_A_cond', y='max_error')
        splot.set(xscale="log", yscale="log")
        fig = splot.get_figure()
        fig.savefig(os.path.join(self.output_dir, "model_14_comparison_summary_A_cond"))
        plt.close(fig)

        splot = sns.scatterplot(data=results_df, x='max_P_cond', y='max_error')
        splot.set(xscale="log", yscale="log")
        fig = splot.get_figure()
        fig.savefig(os.path.join(self.output_dir, "model_14_comparison_summary_P_cond"))
        plt.close(fig)

    def test_hybrid_solver(self):

        protocol = 'staircaseramp'
        voltage_func, times, desc = common.get_ramp_protocol_from_csv(protocol)

        models = [common.make_model_of_class(m, times, voltage_func,
                                             protocol_description=desc) \
                  for m in self.model_names]

        for name, model in zip(self.model_names, models):
            output1 = model.make_forward_solver_current(njitted=False,
                                                        atol=1e-8, rtol=1e-8)()

            output2 = model.make_hybrid_solver_current(njitted=False,
                                                       atol=1e-8, rtol=1e-8)()

            plt.plot(times, output1)
            plt.plot(times, output2)
            plt.savefig(os.path.join(self.output_dir, f"{name}_solver_comparison"))
            plt.clf()

            plt.plot(times, np.abs(output1 - output2))
            plt.yscale('log')
            plt.savefig(os.path.join(self.output_dir, f"{name}_solver_difference_log"))
            plt.clf()

            rmse_error = np.sqrt(np.mean((output1 - output2)**2))

            self.assertLess(rmse_error, 1e-1)

    def test_lsoda_solution(self):
        protocol = 'staircaseramp'
        voltage_func, times, desc = common.get_ramp_protocol_from_csv(protocol)

        tolerances = 1e-9, 1e-9

        voltages = np.array([voltage_func(t) for t in times])
        mk_protocol = myokit.TimeSeriesProtocol(times, voltages)

        this_test_output_dir = os.path.join(self.output_dir, "test_lsoda_ode_compare")
        if not os.path.exists(this_test_output_dir):
            os.makedirs(this_test_output_dir)

        for model in self.model_names:
            mk_model = common.make_myokit_model(model)
            mk_model.set_value('markov_chain.E_Kr', common.calculate_reversal_potential())

            mm_model = common.make_model_of_class(model, times,
                                                  voltage_func,
                                                  protocol_description=desc,
                                                  tolerances=tolerances)

            mk_simulation = myokit.Simulation(mk_model, mk_protocol)
            mk_simulation.set_tolerance(1e-12, 1e-12)
            mk_simulation.set_max_step_size(dtmax=100)
            mk_simulation.pre(1e6)

            d = mk_simulation.run(mk_protocol.times()[-1]+1, log_times=times)
            mk_output = np.array(d['markov_chain.I_Kr'], dtype=np.float64)
            mm_solver = mm_model.make_hybrid_solver_current(njitted=False)
            mm_output = mm_solver(hybrid=False)
            mm_output_hybrid = mm_solver(hybrid=True)

            error = np.sqrt(np.mean((mk_output - mm_output)**2))

            plt.plot(times, mm_output, label='lsoda output')
            plt.plot(times, mm_output_hybrid, label='hybrid/lsoda output')
            plt.plot(times, mk_output, label='Myokit/CVODE output')
            plt.legend()
            plt.savefig(os.path.join(this_test_output_dir,
                                     f"{model}"))
            plt.clf()

            plt.plot(times, np.abs(mm_output - mk_output), label='lsoda error')
            plt.plot(times, np.abs(mm_output_hybrid - mk_output), label='lsoda/hybrid error')
            plt.legend()
            plt.yscale('log')
            plt.savefig(os.path.join(this_test_output_dir,
                                     f"{model}_error"))

            plt.clf()

            for state in mk_model.states():
                plt.plot(d['engine.time'], d[state])
            plt.savefig(os.path.join(this_test_output_dir, f"mk_state_variables_{model}"))
            plt.clf()

            # self.assertLess(error, 1e-1)

    def test_generated_model_output(self):
        protocol = 'staircaseramp'

        for original_model, generated_model in [['Beattie', 'model3']]:
            voltage_func, times, desc = common.get_ramp_protocol_from_csv(protocol)

            tolerances = 1e-10, 1e-10

            model1 = common.make_model_of_class(original_model, times, voltage=voltage_func,
                                                protocol_description=desc, tolerances=tolerances)
            model2 = common.make_model_of_class(generated_model, voltage=voltage_func,
                                                times=times, protocol_description=desc,
                                                tolerances=tolerances)

            default_parameters = model1.get_default_parameters()

            h_solver1 = model1.make_hybrid_solver_current(njitted=False)
            h_solver2 = model2.make_hybrid_solver_current(njitted=False)

            output1 = h_solver1(default_parameters, hybrid=False)
            output2 = h_solver2(default_parameters, hybrid=False)

            rmse_error = np.sqrt(((output1 - output2)**2).mean())

            logging.debug('rmse error is: ' + f"{rmse_error}")

            plt.plot(times, output1, label=original_model)
            plt.plot(times, output2, label='MarkovBuilder model')

            plt.legend()
            plt.savefig(os.path.join(self.output_dir, f"{original_model}_{generated_model}_comparison_lsoda.pdf"))
            plt.clf()

            output1 = h_solver1(default_parameters)
            output2 = h_solver2(default_parameters)

            rmse_error = np.sqrt(((output1 - output2)**2).mean())

            logging.debug('rmse error is: ' + f"{rmse_error}")

            plt.plot(times, output1, label=original_model)
            plt.plot(times, output2, label='MarkovBuilder model')
            plt.legend()

            plt.savefig(os.path.join(self.output_dir, f"{original_model}_{generated_model}_comparison_hybrid.pdf"))
            plt.clf()

            self.assertLess(rmse_error, 1e-2)

    def test_tolerances(self):
        protocol = 'staircaseramp1'
        voltage_func, times, desc = common.get_ramp_protocol_from_csv(protocol)

        model_name = 'model11'
        model = common.make_model_of_class(model_name, times, voltage_func, protocol_description=desc)
        tol_range = 10**np.linspace(-3, -8, 6)
        solver = model.make_hybrid_solver_current(njitted=False, hybrid=False)
        reference_sol = solver(atol=1e-8, rtol=1e-8, hybrid=False)

        rmses = []
        for tol in tol_range:
            sol = solver(atol=tol, rtol=tol, hybrid=False)
            rmse = np.sqrt(np.mean((sol - reference_sol)**2))

            if rmse > 0:
                plt.plot(times, np.abs(sol - reference_sol))

            plt.yscale('log')

            plt.savefig(os.path.join(self.output_dir, f"error_tol={tol}.pdf"))
            plt.clf()

            plt.plot(times, sol)
            plt.plot(times, reference_sol, label='reference')

            plt.savefig(os.path.join(self.output_dir, f"comparison_tol={tol}.pdf"))
            plt.clf()

            rmses.append(rmse)

        plt.plot(tol_range, rmses)
        plt.yscale('log')
        plt.xscale('log')

        plt.savefig(os.path.join(self.output_dir, f"{protocol}_{model_name}_tolerance_vs_error"))
        plt.clf()


