import numba as nb
import numpy as np
from numba import cfunc, njit
from numbalsoda import lsoda_sig
import scipy
import numba
import sympy as sp

no_artefact_parameters = 7

from markovmodels.MarkovModel import MarkovModel
import markovmodels


# TODO Description
class ArtefactModel(MarkovModel):
    def __init__(self, channel_model, E_leak=0, g_leak=0, C_m=5e-3, R_series=5e-3,
                 g_leak_leftover=0, E_leak_leftover=0, ignore_states=[]):

        # Membrane capacitance (nF)
        self.C_m = C_m

        # Series resistance (GOhm)
        self.R_s = R_series

        self.E_rev = channel_model.E_rev
        if self.E_rev is None:
            self.E_rev = markovmodels.utilties.calculate_reversal_potential()

        self.no_artefact_parameters = no_artefact_parameters

        self.g_leak_leftover = g_leak_leftover
        self.E_leak_leftover = E_leak_leftover

        self.E_leak = E_leak
        self.g_leak = g_leak

        self.channel_model = channel_model

        self.p = sp.Matrix.vstack(channel_model.p, sp.Matrix([('p_a_%i' % i) for i in range(7)]))
        self.y = sp.Matrix.vstack(channel_model.y, sp.Matrix([sp.sympify('Vm')]))
        self.v = channel_model.v
        self.n_state_vars = channel_model.n_state_vars + 1

        self.voltage = channel_model.voltage
        self.rates_dict = channel_model.rates_dict

        self.times = channel_model.times
        self.protocol_description = channel_model.protocol_description

        self.solver_tolerances = channel_model.solver_tolerances
        self.E_rev = channel_model.E_rev
        self.channel_model.compute_steady_state_expressions()
        self.auxiliary_function = self.define_auxiliary_function()

        self.rhs_inf = self.define_steady_state_function()
        p = self.p
        y = self.y

        I_out_expr = channel_model.auxiliary_expression +\
            sp.sympify('g_leak * (V_m - E_leak) + g_leak_leftover * (V_m - E_leak_leftover)')

        I_out_expr = I_out_expr.subs({'V': 'Vm'})

        artefact_rhs_expr = sp.sympify('(V_off - V_m)/(C_m * R_s) * 1e-3 - I_out * 1e-3 / C_m')
        subs_dict = {'V_m': y[-1],
                     'R_s': p[-1],
                     'C_m': p[-2],
                     'V_off': p[-3],
                     'E_leak_leftover': p[-4],
                     'g_leak_leftover': p[-5],
                     'E_leak': p[-6],
                     'g_leak': p[-7],
                     'I_out': I_out_expr,
                     'E_Kr': self.E_rev}

        artefact_rhs_expr = artefact_rhs_expr.subs(subs_dict).subs(subs_dict)

        self.auxiliary_expression = I_out_expr.subs(subs_dict).subs(subs_dict)

        self.rhs_expr = sp.Matrix.vstack(channel_model.rhs_expr, sp.Matrix([artefact_rhs_expr]))

        self.initial_condition = np.append(
            self.channel_model.rhs_inf(self.channel_model.get_default_parameters(),
                                       self.voltage(0)),
            self.voltage(0))


    def define_steady_state_function(self):
        def rhs_inf(p, voltage):
            # Find a steady state of the system actual steady state

            channel_params = p[:-no_artefact_parameters]

            g_leak, E_leak, g_leak_leftover, E_leak_leftover, V_off, C_m, R_s = p[-no_artefact_parameters:]
            channel_rhs_inf = self.channel_model.rhs_inf
            channel_model_auxiliary_function = njit(self.channel_model.define_auxiliary_function())

            def I_inf(V_m):
                g_leak, E_leak, g_leak_leftover, E_leak_leftover, V_off, C_m, R_s = p[-no_artefact_parameters:]
                x_inf = channel_rhs_inf(channel_params, V_m).flatten()
                I_inf = channel_model_auxiliary_function(x_inf, channel_params, V_m) \
                    + g_leak_leftover * (V_m - E_leak_leftover) \
                    + g_leak*(V_m - E_leak)
                return I_inf

            # Function to find root of
            def f_func(V_m):
                return voltage + V_off - V_m - I_inf(V_m) * R_s

            # V_m = scipy.optimize.root_scalar(f_func, x0=-80, method='secant').root
            V_m = -80
            rhs_inf = np.append(channel_rhs_inf(channel_params, V_m).flatten(), V_m)
            return np.expand_dims(rhs_inf, -1)
        return rhs_inf

    def get_default_parameters(self):
        channel_parameters = self.channel_model.get_default_parameters()
        # g_leak_leftover, E_leak_leftover, V_off, C_m, R_s = p[-no_artefact_parameters:]
        default_artefact_parameters = np.array([self.g_leak, self.E_leak, .0,
                                                .0, .0, self.C_m, self.R_s]).astype(np.float64)

        return np.concatenate((channel_parameters,
                               default_artefact_parameters)).astype(np.float64).flatten()

    def get_parameter_labels(self):
        return self.channel_model.get_parameter_labels() + ['g_leak_leftover', 'E_leak_leftover',
                                                            'V_off', 'C_m', 'R_s']

    def define_auxiliary_function(self):
        channel_auxiliary_function = njit(self.channel_model.define_auxiliary_function())

        def auxiliary_func(x, p, _):
            g_leak, E_leak, g_leak_leftover, E_leak_leftover, V_off, C_m, R_s = p[-no_artefact_parameters:]
            V_m = x[-1, :]
            I_Kr = channel_auxiliary_function(x[:-1], p[:-no_artefact_parameters], V_m)

            I_leak = g_leak * (V_m - E_leak)
            I_leak_leftover = g_leak_leftover * (V_m - E_leak_leftover)
            I_post = I_Kr + I_leak + I_leak_leftover

            return I_post

        return auxiliary_func

    def get_no_state_vars(self):
        ny = self.channel_model.get_no_state_vars() + 1
        return ny

    def make_hybrid_solver_states(self, protocol_description=None,
                                  njitted=False, analytic_solver=None,
                                  strict=True, cond_threshold=None, atol=None,
                                  rtol=None, hybrid=True):
        if hybrid:
            raise NotImplementedError()
        else:
            solver = super().make_hybrid_solver_states(
                protocol_description,
                njitted=False,
                analytic_solver=analytic_solver,
                strict=strict,
                cond_threshold=cond_threshold,
                atol=atol, rtol=rtol,
                hybrid=False
            )

        if njitted:
            solver = numba.jit(solver, nopython=False)
        return solver

    def make_hybrid_solver_current(self, protocol_description=None,
                                   njitted=False, analytic_solver=None,
                                   strict=True, cond_threshold=None, atol=None,
                                   rtol=None, hybrid=True):
        if hybrid:
            raise NotImplementedError()
        else:
            solver = super().make_hybrid_solver_current(
                protocol_description,
                njitted=False,
                analytic_solver=analytic_solver,
                strict=strict,
                cond_threshold=cond_threshold,
                atol=atol, rtol=rtol,
                hybrid=False
            )

        if njitted:
            solver = numba.jit(solver, nopython=False)
        return solver

    def get_cfunc_rhs(self):
        channel_rhs = self.channel_model.get_rhs_func(njitted=True)

        # States are:
        # [channel model], V_m
        #

        # Eliminate one state and add equation for V_m
        ny = self.channel_model.get_no_state_vars() + 1

        # Add artefact parameters and t_offset
        n_p = len(self.get_default_parameters())

        prot_func = self.channel_model.voltage

        channel_auxiliary_function = njit(self.channel_model.define_auxiliary_function())

        @cfunc(lsoda_sig)
        def crhs_func(t, y, dy, data):
            y = nb.carray(y, ny)
            dy = nb.carray(dy, ny)
            data = nb.carray(data, n_p + 1)

            t_offset = data[-1]
            p = data[:-1]

            #  get artefact parameters
            g_leak, E_leak, g_leak_leftover, E_leak_leftover, V_off, C_m, R_s = p[-no_artefact_parameters:]

            V_m = y[-1]
            V_cmd = prot_func(t, offset=t_offset)

            I_leak = (V_m - E_leak) * g_leak
            I_leak_leftover = (V_m - E_leak_leftover) * g_leak_leftover

            I_Kr = channel_auxiliary_function(y[:-1], p[:-no_artefact_parameters], V_m)

            I_out = I_Kr + I_leak + I_leak_leftover

            # No series resistance compensation
            V_p = V_cmd

            # V_m derivative
            dy[-1] = (V_p + V_off - V_m)/(C_m * R_s) * 1e-3 - I_out * 1e-3 / C_m

            # compute derivative for channel model
            dy[0:-1] = channel_rhs(y[0:-1], p[:-no_artefact_parameters], V_m).flatten()
            return

        return crhs_func

    def get_analytic_solution_func(self, njitted=None, cond_threshold=None):
        raise NotImplementedError()

    def get_state_labels(self):
        return np.append(self.channel_model.get_state_labels(),
                         'V_m')
