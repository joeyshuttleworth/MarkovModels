import numba as nb
import numpy as np
from numba import cfunc, njit
from numbalsoda import lsoda_sig, lsoda
import pints
import numba
import sympy as sp

from markovmodels.MarkovModel import MarkovModel
import markovmodels

no_artefact_parameters = 7


# TODO Description
class ArtefactModel(MarkovModel):
    def __init__(self, channel_model, E_leak=0, g_leak=0, C_m=5e-3, R_series=5e-3,
                 g_leak_leftover=0, E_leak_leftover=0, V_off=0, ignore_states=[]):

        # Membrane capacitance (nF)
        self.C_m = C_m

        # Series resistance (GOhm)
        self.R_s = R_series

        self.V_off = V_off

        self.E_rev = channel_model.E_rev
        if self.E_rev is None:
            self.E_rev = markovmodels.utilties.calculate_reversal_potential()

        self.no_artefact_parameters = no_artefact_parameters

        self.g_leak_leftover = g_leak_leftover
        self.E_leak_leftover = E_leak_leftover

        self.E_leak = E_leak
        self.g_leak = g_leak

        self.channel_model = channel_model

        self.p = sp.Matrix.vstack(channel_model.p,
                                  sp.Matrix([[sp.sympify(label) for label in self.get_parameter_labels()[:-no_artefact_parameters]]]).T)
        self.y = sp.Matrix.vstack(channel_model.y, sp.Matrix([sp.sympify('Vm')]))
        self.v = channel_model.v
        self.n_state_vars = channel_model.n_state_vars + 1

        self.voltage = channel_model.voltage
        self.rates_dict = channel_model.rates_dict

        self.times = channel_model.times

        self.protocol_description = channel_model.protocol_description

        if self.protocol_description is None:
            raise ValueError()

        self.solver_tolerances = channel_model.solver_tolerances
        self.E_rev = channel_model.E_rev
        self.channel_model.compute_steady_state_expressions()
        self.auxiliary_function = njit(self.define_auxiliary_function())

        self.transformations = self.channel_model.transformations + \
            [pints.IdentityTransformation(1)] * no_artefact_parameters

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

    def define_steady_state_function(self, tend=5000):
        # Assume a holding potential of -80mV and simulate forwards for 5 seconds
        # start from -80mV steady state of the Markov model and Vm=-80mV 
        atol, rtol = self.solver_tolerances
        p = self.get_default_parameters()

        crhs = self.get_cfunc_rhs()
        crhs_ptr = crhs.address

        y0 = np.full(self.channel_model.get_no_state_vars(), .0)
        y0[0] = 1.0
        y0 = np.append(y0, -80.0)

        n_max_steps = 64

        @njit
        def rhs_inf(p=p, v=-80):
            data = np.append(p, 0.0)
            data = np.concatenate((data, np.full(n_max_steps*2, np.inf)))

            res, _ = lsoda(crhs_ptr, y0,
                           np.array((-tend, .0)),
                           data=data,
                           rtol=rtol,
                           atol=atol,
                           exit_on_warning=True)

            return res[-1, :].flatten()
        return rhs_inf

    def get_default_parameters(self):
        channel_parameters = self.channel_model.get_default_parameters()
        # g_leak_leftover, E_leak_leftover, V_off, C_m, R_s = p[-no_artefact_parameters:]
        default_artefact_parameters = np.array([self.g_leak, self.E_leak,
                                                self.g_leak_leftover,
                                                self.E_leak_leftover,
                                                self.V_off, self.C_m,
                                                self.R_s]).astype(np.float64)

        return np.concatenate((channel_parameters,
                               default_artefact_parameters)).astype(np.float64).flatten()

    def get_parameter_labels(self):
        return self.channel_model.get_parameter_labels() + ['g_leak', 'E_leak',
                                                            'g_leak_leftover',
                                                            'E_leak_leftover',
                                                            'V_off', 'C_m',
                                                            'R_s']

    def define_auxiliary_function(self, return_var='I_Kr', **kwargs):
        channel_auxiliary_function = njit(self.channel_model.define_auxiliary_function())

        def auxiliary_func(x, p, _):
            g_leak, E_leak, g_leak_leftover, E_leak_leftover, V_off, C_m, R_s = p[-no_artefact_parameters:]
            V_m = x[-1, :]
            I_Kr = channel_auxiliary_function(x[:-1], p[:-no_artefact_parameters], V_m)

            I_leak = g_leak * (V_m - V_off - E_leak)
            I_leak_leftover = g_leak_leftover * (V_m - E_leak_leftover)
            I_post = I_Kr + I_leak + I_leak_leftover

            if return_var == 'I_Kr':
                return I_Kr
            else:
                return I_post

        return auxiliary_func

    def get_no_state_vars(self):
        ny = self.channel_model.get_no_state_vars() + 1
        return ny

    def make_hybrid_solver_states(self, protocol_description=None,
                                  njitted=False, analytic_solver=None,
                                  strict=True, cond_threshold=None, atol=None,
                                  rtol=None, hybrid=True, crhs=None):
        if hybrid:
            raise NotImplementedError()
        else:
            solver = super().make_hybrid_solver_states(
                protocol_description=protocol_description,
                njitted=False,
                analytic_solver=analytic_solver,
                strict=strict,
                cond_threshold=cond_threshold,
                atol=atol, rtol=rtol,
                hybrid=False,
                crhs=crhs
            )

        if njitted:
            solver = numba.njit(solver)
        return solver

    def make_hybrid_solver_current(self, protocol_description=None,
                                   njitted=False, analytic_solver=None,
                                   strict=True, cond_threshold=None, atol=None,
                                   rtol=None, hybrid=True, return_var='I_Kr',
                                   cfunc=None):
        if hybrid:
            raise NotImplementedError()
        else:
            solver = super().make_hybrid_solver_current(
                protocol_description=protocol_description,
                njitted=False,
                analytic_solver=analytic_solver,
                strict=strict,
                cond_threshold=cond_threshold,
                atol=atol, rtol=rtol, return_var=return_var,
                hybrid=False, cfunc=cfunc
            )

        if njitted:
            solver = numba.njit(solver)
        return solver

    def get_cfunc_rhs(self):
        channel_rhs = self.channel_model.get_rhs_func(njitted=True)

        # States are:
        # [channel model], V_m

        # Eliminate one state and add equation for V_m
        ny = self.channel_model.get_no_state_vars() + 1

        # Add artefact parameters and t_offset
        n_p = len(self.get_default_parameters())

        prot_func = self.channel_model.voltage

        # Maximum steps in protocol
        n_max_steps = 64

        channel_auxiliary_function = njit(self.channel_model.define_auxiliary_function())

        @cfunc(lsoda_sig)
        def crhs_func(t, y, dy, data):
            y = nb.carray(y, ny)
            dy = nb.carray(dy, ny)
            data = nb.carray(data, n_p + 1 + n_max_steps * 4)

            p = data[:n_p]
            t_offset = data[n_p]
            desc = data[n_p + 1:].reshape((-1, 4))

            #  get artefact parameters
            g_leak, E_leak, g_leak_leftover, E_leak_leftover, V_off, C_m, R_s = p[-no_artefact_parameters:]

            V_m = y[-1]
            V_cmd = prot_func(t, offset=t_offset,
                              protocol_description=desc)

            I_leak = (V_m - V_off - E_leak) * g_leak
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
