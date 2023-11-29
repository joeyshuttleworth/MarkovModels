from markovmodels.ODEModel import ODEModel
from markovmodels.MarkovModel import MarkovModel

import sympy as sp
import numpy as np
from numba import njit, cfunc
from numbalsoda import lsoda, lsoda_sig
import numba as nb


class SensitivitiesMarkovModel(ODEModel):

    def __init__(self, markov_model, parameters_to_use=None,
                 solver_tolerances=None):
        self.markov_model = markov_model

        if not parameters_to_use:
            self.parameters_to_use = self.markov_model.get_parameter_labels()
        else:
            self.parameters_to_use = parameters_to_use

        self.rates_dict = markov_model.rates_dict
        self.p = sp.sympify(self.parameters_to_use)
        self.v = markov_model.v
        self.E_rev = self.markov_model.E_rev

        if not solver_tolerances:
            self.solver_tolerances = self.markov_model.solver_tolerances
        else:
            self.solver_tolerances = solver_tolerances

        self.default_parameters = markov_model.get_default_parameters()

        self.n_params = len(self.parameters_to_use)

        self.times = markov_model.times.copy()
        self.voltage = markov_model.voltage

        self.protocol_description = markov_model.protocol_description

        self.setup_sensitivities()
        self.compute_steady_state_expressions()

        self.parameter_labels = markov_model.get_parameter_labels()

    def get_default_parameters(self):
        return self.default_parameters.copy()

    def get_no_state_vars(self):
        # Include the additional Vm state (membrane voltage)
        return self.markov_model.get_no_state_vars() * (1 + self.n_params)

    def compute_steady_state_expressions(self, tend=5000):
        p = self.get_default_parameters()
        atol, rtol = self.solver_tolerances

        y0 = np.full(self.get_no_state_vars(), .0)
        y0[:self.markov_model.get_no_state_vars()] = self.markov_model.rhs_inf(p, self.voltage(0)).flatten()

        crhs = self.get_cfunc_rhs()
        crhs_ptr = crhs.address

        @njit
        def rhs_inf(p=p, v=-80):
            data = np.append(p, 0)
            res, _ = lsoda(crhs_ptr, y0,
                           np.array((-tend, .0)),
                           data=data,
                           rtol=rtol,
                           atol=atol,
                           exit_on_warning=True)
            return res[-1, :].flatten()

        self.rhs_inf = rhs_inf

        return rhs_inf

    def get_analytic_solver_func():
        raise NotImplementedError()

    def get_cfunc_rhs(self):
        rhs = nb.njit(self.func_S1)
        voltage = self.voltage
        n_p = len(self.get_default_parameters())
        ny = self.get_no_state_vars()
        n_max_steps = 64

        @cfunc(lsoda_sig)
        def crhs(t, y, dy, data):
            y = nb.carray(y, ny)
            dy = nb.carray(dy, ny)
            data = nb.carray(data, n_p + 1 + n_max_steps * 4)

            p = data[:n_p]
            t_offset = data[n_p]
            desc = data[n_p + 1:]

            v = voltage(t, offset=t_offset,
                        protocol_description=desc)

            res = rhs(y, p, v).flatten()
            dy[:] = res.flatten()

        return crhs

    def setup_sensitivities(self):
        inputs = (list(self.markov_model.y), self.p, self.v)
        n_state_vars = self.get_no_state_vars()
        # Create symbols for 1st order sensitivities
        dydp = [[sp.symbols(f"dy{i}_dp{j}") for j in range(self.n_params)]
                for i in range(len(self.markov_model.y))]

        y = self.markov_model.y

        # Append 1st order sensitivities to inputs
        for i in range(len(dydp)):
            for j in range(self.n_params):
                inputs[0].append(dydp[i][j])

        self.y = inputs[0]

        S = sp.Matrix([[dydp[i][j] for j in range(self.n_params)]
                       for i in range(len(self.markov_model.y))])

        # Create 1st order sensitivities function
        J = self.markov_model.rhs_expr.jacobian(self.markov_model.y)

        F = sp.Matrix([[sp.diff(f, p) for f in self.markov_model.rhs_expr] for p in self.p]).T
        dS = J @ S + F

        # The sensitivity of the auxiliary function wrt the state variables
        self.dIdo = {y: sp.diff(self.markov_model.auxiliary_expression, y) for y in y}

        self.auxiliary_expression = sp.Matrix([sum([self.dIdo[y] * S[i, j] for
                                                    i, y in enumerate(y)]) for j, p
                                               in enumerate(self.p)])

        self.auxiliary_expression += sp.Matrix([sp.diff(self.markov_model.auxiliary_expression, p)
                                                for p in self.p])

        self.auxiliary_expression = self.auxiliary_expression.subs({'E_Kr': self.E_rev})

        self.auxiliary_function = sp.lambdify(inputs, self.auxiliary_expression, cse=True)

        # Define number of 1st order sensitivities
        self.n_state_var_sensitivities = self.n_params * n_state_vars

        # Concatenate RHS and 1st order sensitivities
        fS1 = sp.flatten(dS)
        fS1 = sp.Matrix(np.concatenate((sp.flatten(self.markov_model.rhs_expr), fS1)))
        self.fS1 = fS1
        self.func_S1 = sp.lambdify(inputs, fS1)

        # Create Jacobian of the 1st order sensitivities function
        Ss = sp.Matrix(np.concatenate((sp.flatten(y), sp.flatten(S))))
        jS1 = fS1.jacobian(Ss)
        self.jfunc_S1 = sp.lambdify(inputs, jS1)

    def define_auxiliary_function(self):
        return self.auxiliary_function
