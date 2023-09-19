from .MarkovModel import MarkovModel
import numba as nb
from numba import njit, cfunc
import numpy as np
from numbalsoda import lsoda_sig


# TODO Description
class ArtefactModel(MarkovModel):
    def __init__(self, channel_model, E_leak=0, g_leak=0, C_m=5e-3, R_s=5,
                 g_leak_leftover=0, E_leak_leftover=0):

        # Membrane capacitance (nF)
        self.C_m = C_m

        # Series resistance (MOhm)
        self.R_s = R_s

        self.g_leak_leftover = g_leak_leftover
        self.E_leak_leftover = E_leak_leftover

        self.E_leak = E_leak
        self.g_leak = g_leak

        self.channel_model = channel_model

        self.p = channel_model.p
        self.y = channel_model.y
        self.v = channel_model.v

        self.voltage = channel_model.voltage

        self.rates_dict = channel_model.rates_dict

        self.times = channel_model.times
        self.protocol_description = channel_model.protocol_description

        self.solver_tolerances = channel_model.solver_tolerances

        self.E_rev = channel_model.E_rev

        self.channel_model.compute_steady_state_expressions()

        self.auxiliary_function = self.define_auxiliary_function()

        channel_rhs_inf = self.channel_model.rhs_inf

        @njit
        def rhs_inf(p, voltage):
            # TODO find actual steady state
            channel_inf = channel_rhs_inf(p[:-3], voltage).flatten()
            rhs_inf = np.expand_dims(np.append(channel_inf, -80), -1)
            return rhs_inf

        self.rhs_inf = rhs_inf

    def get_default_parameters(self):
        channel_parameters = self.channel_model.get_default_parameters()
        default_artefact_parameters = np.array([0.0]*3)

        return np.concatenate((channel_parameters,
                               default_artefact_parameters))

    def get_parameter_labels(self):
        return self.channel_model.get_parameter_labels() + ['g_leak_leftover', 'E_leak_leftover',
                                                            'V_off']

    def define_auxiliary_function(self):
        channel_auxiliary_function = njit(self.channel_model.define_auxiliary_function())

        @njit
        def auxiliary_func(x, p, _):
            g_leak_leftover, E_leak_leftover, V_off = p[-3:]
            V_m = x[-1, :]
            I_Kr = channel_auxiliary_function(x[:-1], p[:-3], V_m)

            I_post = I_Kr - g_leak_leftover * (V_m - E_leak_leftover)

            return I_post

        return auxiliary_func

    def get_no_states(self):
        ny = self.channel_model.get_no_states() + 1
        return ny

    def make_hybrid_solver_states(self, protocol_description=None,
                                  njitted=False, analytic_solver=None,
                                  strict=True, cond_threshold=None, atol=None,
                                  rtol=None, hybrid=True):
        if hybrid:
            raise NotImplementedError()
        else:
            return super().make_hybrid_solver_states(
                protocol_description,
                njitted,
                analytic_solver,
                strict,
                cond_threshold,
                atol, rtol,
                hybrid=False
            )

    def get_cfunc_rhs(self):
        channel_rhs = self.channel_model.get_rhs_func(njitted=True)

        # States are:
        # [channel model], V_m
        #

        # Eliminate one state and add equation for V_m
        ny = self.channel_model.get_no_states() + 1
        print(f"ny is {ny}")

        # Add artefact parameters and t_offset
        n_p = len(self.get_default_parameters())

        prot_func = self.channel_model.voltage

        R_s = self.R_s
        C_m = self.C_m

        E_leak = self.E_leak
        g_leak = self.g_leak

        channel_auxiliary_function = njit(self.channel_model.define_auxiliary_function())

        @cfunc(lsoda_sig)
        def crhs_func(t, y, dy, data):
            y = nb.carray(y, ny)
            dy = nb.carray(dy, ny)
            data = nb.carray(data, n_p + 1)

            t_offset = data[-1]
            p = data[:-1]

            #  get artefact parameters
            g_leak_leftover, E_leak_leftover, V_off = p[-3:]

            V_m = y[-1]
            V_cmd = prot_func(t, offset=t_offset)

            I_leak = (V_cmd - E_leak) * g_leak
            # I_Kr = y[open_index] * (V_m - E_Kr) * p[gkr_index]

            I_Kr = channel_auxiliary_function(y[:-1], p[:-3], V_m)

            I_out = I_Kr + I_leak

            # No series resistance compensation
            V_p = V_cmd

            # V_m derivative
            dy[-1] = (V_p + V_off - V_m)/(C_m * R_s) - I_out / C_m

            # compute derivative for channel model
            dy[0:-1] = channel_rhs(y[0:-1], p[:-3], V_m).flatten()
            return

        return crhs_func

    def get_analytic_solution_func(self, njitted=None, cond_threshold=None):
        raise NotImplementedError()

    def get_state_labels(self):
        return np.append(self.channel_model.get_state_labels(),
                         'V_m')
