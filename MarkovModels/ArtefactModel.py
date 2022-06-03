from .MarkovModel import MarkovModel
import sympy as sp
import numba as nb
from numba import njit, cfunc
import numpy as np
import numbalsoda
from numbalsoda import lsoda, lsoda_sig


# Simplest Artefact model containing cell capacitance
class ArtefactModel():
    def __init__(self, channel_model, C_m=20e-3, R_s=30):

        # Membrane capacitance (nF)
        self.C_m = C_m

        # Series resistance (MOhm)
        self.R_s = R_s

        self.channel_model = channel_model

    def get_rhs_cfunc(self):
        channel_rhs = self.channel_model.get_rhs_func(njitted=True)

        # States are:
        # [channel mode], V_m
        #

        # Eliminate one state and add equation for V_m
        ny = len(self.channel_model.state_labels) - 1 + 1
        np = self.channel_model.get_no_parameters()

        open_index = self.channel_model.open_state_index

        E_Kr = self.channel_model.Erev

        prot_func = self.channel_model.voltage

        R_s = self.R_s
        C_m = self.C_m

        gkr_index = self.channel_model.GKr_index

        @cfunc(lsoda_sig)
        def crhs_func(t, y, dy, p):
            y = nb.carray(y, ny)
            dy = nb.carray(dy, ny)
            p = nb.carray(p, np)
            V_m = y[-1]

            I_Kr = y[open_index] * (V_m - E_Kr) * p[gkr_index]

            # d_Vm = dy[-1]
            V_in = prot_func(t)

            # V_m derivative
            dy[-1] = (V_in - V_m)/(C_m * R_s)

            # compute derivative for channel model
            dy[0:-1] = channel_rhs(y[0:-1], p, V_m).flatten()
            return

        return crhs_func

    def make_solver(self, p=None, times=None, atol=None, rtol=None, njitted=True, return_current=True):

        if atol is None:
            atol = self.channel_model.solver_tolerances[0]
        if rtol is None:
            rtol = self.channel_model.solver_tolerances[1]

        crhs = self.get_rhs_cfunc()
        crhs_ptr = crhs.address

        rhs_inf = self.channel_model.rhs_inf
        voltage = self.channel_model.voltage

        # Number of state variables
        no_states = self.channel_model.get_no_states() - 1 + 1

        times = self.channel_model.times

        protocol_description = self.channel_model.protocol_description
        default_params = self.channel_model.get_default_parameters()

        open_state_index = self.channel_model.open_state_index
        conductance_index = self.channel_model.GKr_index
        reversal_potential = self.channel_model.Erev

        R_s = self.R_s
        C_m = self.C_m

        def forward_solver(p=default_params, times=times, atol=atol, rtol=rtol):
            rhs0 = np.array(list(rhs_inf(p, voltage(0)).flatten()) + [voltage(0)])
            solution = np.empty((len(times), no_states))
            for tstart, tend, vstart, vend in protocol_description:
                istart = np.argmax(times >= tstart)
                iend = np.argmax(times >= tend)
                if iend == 0:
                    step_times = times[istart:]
                    iend = len(times)
                else:
                    iend += 1
                    step_times = times[istart:iend + 1]

                step_sol, _ = lsoda(crhs_ptr, rhs0, step_times, data=p,
                                    rtol=rtol, atol=atol)
                if iend == len(times):
                    solution[istart:, ] = step_sol[:, ]
                    break

                else:
                    rhs0 = step_sol[-1, :]
                    solution[istart:iend, ] = step_sol[:-1, ]
            if return_current:
                V_m = solution[:, -1]
                I_Kr = solution[:, open_state_index] * (V_m - reversal_potential) * p[-1]
                voltages = np.empty(shape=times.shape)
                for i in range(voltages.shape[0]):
                    voltages[i] = voltage(times[i])
                I_m = (voltages - solution[:, -1])/(R_s)

                I_in = I_m + I_Kr
                return I_in
            else:
                return solution

        return njit(forward_solver) if njitted else forward_solver

    def SimulateForwardModel(self, p=None, times=None, atol=None, rtol=None, return_current=True):
        channel_model = self.channel_model
        if p is None:
            p = self.channel_model.get_default_parameters()
        if times is None:
            times = channel_model.times
        if atol is None:
            atol = channel_model.solver_tolerances[0]
        if rtol is None:
            rtol = channel_model.solver_tolerances[1]
        sol = self.make_solver(njitted=False, return_current=return_current)(p, times, atol, rtol)

        return sol
