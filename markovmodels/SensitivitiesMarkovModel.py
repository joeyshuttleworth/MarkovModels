from markovmodels.ODEModel import ODEModel
from markovmodels.MarkovModel import MarkovModel

import sympy as sp
import numpy as np
from numba import njit, cfunc
from numbalsoda import lsoda, lsoda_sig
import numba as nb


class SensitivitiesMarkovModel(ODEModel):

    def __init__(self, markov_model):
        self.markov_model = markov_model

        self.rates_dict = markov_model.rates_dict
        self.p = markov_model.p
        self.v = markov_model.v
        self.E_rev = self.markov_model.E_rev
        self.solver_tolerances = self.markov_model.solver_tolerances

        self.default_parameters = markov_model.get_default_parameters()

        self.n_params = len(markov_model.get_default_parameters())

        self.times = markov_model.times.copy()
        self.voltage = markov_model.voltage

        self.protocol_description = markov_model.protocol_description

        self.setup_sensitivities()
        self.compute_steady_state_expressions()

    def get_default_parameters(self):
        return self.default_parameters.copy()

    def get_no_state_vars(self):
        # Include the additional Vm state (membrane voltage)
        return self.markov_model.get_no_state_vars() * (1 + len(self.get_default_parameters()))

    def compute_steady_state_expressions(self, tend=1):
        p = self.get_default_parameters()
        atol, rtol = self.solver_tolerances

        y0 = np.full(self.get_no_state_vars(), .0)
        y0[0] = 1.0

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
        sp.pprint(self.fS1)

        y0 = np.full(self.get_no_state_vars(), 0.0)
        y0[:self.markov_model.get_no_state_vars()] = self.markov_model.initial_condition
        print(rhs(y0, self.get_default_parameters(), -80.0))

        voltage = self.markov_model.voltage

        n_p = len(self.get_default_parameters())
        ny = self.get_no_state_vars()

        assert(ny == len(y0))

        @cfunc(lsoda_sig)
        def crhs(t, y, dy, data):
            y = nb.carray(y, ny)
            dy = nb.carray(dy, ny)
            data = nb.carray(data, n_p + 1)

            # print(y, dy)

            p = data[:-1]
            t_offset = data[-1]

            res = rhs(y,
                      p,
                      voltage(t, offset=t_offset)).flatten()

            dy = res.flatten()

        return crhs

    def setup_sensitivities(self):
        inputs = (list(self.markov_model.y), self.p, self.v)
        n_state_vars = self.get_no_state_vars()
        # Create symbols for 1st order sensitivities
        dydp = [
            [
                sp.symbols(
                    'dy%d' %
                    i + 'dp%d' %
                    j) for j in range(
                    self.n_params)] for i in range(
                        len(self.markov_model.y))]

        y = self.markov_model.y

        # Append 1st order sensitivities to inputs
        for i in range(self.n_params):
            for j in range(len(dydp)):
                inputs[0].append(dydp[j][i])

        # Initialise 1st order sensitivities
        ny = self.markov_model.get_no_state_vars()
        dS = sp.zeros(ny, self.n_params)
        S = sp.Matrix([[dydp[i][j] for j in range(self.n_params)]
                       for i in range(len(self.markov_model.y))])

        # The sensitivity of each rate to the parameters
        # rate_sensitivities = {r: sp.diff(expr, self.p) for r, expr in self.rates_dict.items()}

        # Create 1st order sensitivities function
        for i in range(ny):
            for j in range(self.n_params):
                # sens_to_rates = {rate: sp.diff(self.rhs_expr[i], rate) for rate in self.rates_dict}
                sens_to_param = sp.diff(self.markov_model.rhs_expr[i], self.p[j])
                dS[i, j] = sens_to_param

                for k in range(ny):
                    state_contribution = sp.diff(self.markov_model.rhs_expr[i], y[k]) * dydp[k][j]
                    dS[i, j] += state_contribution

        print(dS)

        # The sensitivity of the auxiliary function wrt the state variables
        self.dIdo = {y: sp.diff(self.markov_model.auxiliary_expression, y) for y in y}

        self.auxiliary_expression = sp.Matrix([sum([self.dIdo[y] * dS[i, j] +
                                                    sp.diff(self.markov_model.auxiliary_expression, p)
                                                    for i, y in enumerate(y)])
                                               for j, p in enumerate(self.p)]).subs({'E_Kr': self.E_rev})

        self.auxiliary_function = sp.lambdify(inputs, self.auxiliary_expression)

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
