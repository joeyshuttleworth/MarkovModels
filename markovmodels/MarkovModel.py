import numpy as np
import sympy as sp
import logging
import time
from scipy.integrate import solve_ivp
import time
from numbalsoda import lsoda_sig, lsoda, dop853
from numba import njit, cfunc, literal_unroll
import numba as nb

from .common import calculate_reversal_potential


class MarkovModel:
    """
    A class describing a Markov Model of an ion channel

    """

    def get_default_parameters(self):
        return self.default_parameters.copy()

    def get_model_name(self):
        return self.model_name

    def get_parameter_labels(self):
        return self.parameter_labels.copy()

    def __init__(self, symbols, A, B, rates_dict, times=None, voltage=None,
                 tolerances=(1e-8, 1e-8), Q=None, protocol_description=None,
                 name=None, E_rev=None, default_parameters=None,
                 parameter_labels=None, GKr_index: int = None,
                 open_state_index: int = None, transformations=None,
                 state_labels: str = None):

        self.name = name

        if state_labels:
            self.state_labels = state_labels

        if GKr_index:
            self.GKr_index = GKr_index

        if open_state_index is not None:
            self.open_state_index = open_state_index

        self.n_states = A.shape[0] + 1

        if default_parameters is not None:
            self.default_parameters = default_parameters

        if parameter_labels:
            self.parameter_labels = parameter_labels

        if E_rev is None:
            self.E_rev = calculate_reversal_potential()
        else:
            self.E_rev = E_rev

        self.Q = sp.sympify(Q)

        self.transformations = transformations

        if voltage is not None:
            self.holding_potential = voltage(0)

        self.model_name = name

        self.protocol_description = protocol_description

        self.window_locs = None

        self.y = symbols['y']
        self.p = symbols['p']
        self.v = symbols['v']

        self.rates_dict = rates_dict

        # (atol, rtol)
        self.solver_tolerances = tuple(tolerances)

        self.symbols = symbols
        # The timesteps we want to output at

        self.times = times
        self.A = A
        self.B = B

        self.rhs_expr = (A @ sp.Matrix(self.y[:A.shape[0], :]) + B).subs(rates_dict)

        if voltage is not None:
            self.voltage = voltage

        # Create RHS function
        frhs = np.array([e for e in self.rhs_expr])
        self.frhs = frhs

        inputs = list(self.y) + list(self.p) + [self.v]

        self.func_rhs = sp.lambdify((self.y, self.p, self.v), self.rhs_expr, cse=True)

        # Create Jacobian of the RHS function
        jrhs = sp.Matrix(self.rhs_expr).jacobian(self.y)
        self.jfunc_rhs = sp.lambdify(inputs, jrhs)
        # Set the initial conditions of the model and the initial sensitivities
        # by finding the steady state of the model

        self.compute_steady_state_expressions()

        self.auxiliary_function = self.define_auxiliary_function()

    def define_auxiliary_function(self):
        return sp.lambdify((self.y, self.p, self.v), self.auxiliary_expression)

    def compute_steady_state_expressions(self):

        self.rhs_inf_expr_rates = -self.A.LUsolve(self.B)
        self.rhs_inf_expr = self.rhs_inf_expr_rates.subs(self.rates_dict)
        self.rhs_inf = nb.njit(sp.lambdify((self.p, self.v), self.rhs_inf_expr,
                                           modules='numpy', cse=True))
        self.auxiliary_expression = self.p[self.GKr_index] * \
            self.y[self.open_state_index] * (self.v - self.E_rev)

        self.current_inf_expr = self.auxiliary_expression.subs(self.y, self.rhs_inf)
        self.current_inf = lambda p: np.array(self.current_inf_expr.subs(
            dict(zip(self.p, p))).evalf()).astype(np.float64)

    def setup_sensitivities(self):
        rate_expressions = [self.rates_dict[r] for r in self.rates_dict]
        inputs = list(self.y) + list(self.p) + [self.v]
        # Create symbols for 1st order sensitivities
        dydp = [
            [
                sp.symbols(
                    'dy%d' %
                    i + 'dp%d' %
                    j) for j in range(
                    self.n_params)] for i in range(
                        self.no_states)]

        # Append 1st order sensitivities to inputs
        for i in range(self.n_params):
            for j in range(self.n_state_vars):
                inputs.append(dydp[j][i])

        rate_sens = sp.Matrix([[sp.diff(r, p) for p in self.p] for r in rate_expressions])
        drhs_rate = sp.Matrix([[sp.diff(r, p) for p in self.p] for r in self.rhs_expr])

        # Initialise 1st order sensitivities
        dS = [[0 for j in range(self.n_params)]
              for i in range(self.n_state_vars)]
        S = [[dydp[i][j] for j in range(self.n_params)]
             for i in range(self.n_state_vars)]

        # Create 1st order sensitivities function
        fS1, Ss = [], []
        for i in range(self.n_state_vars):
            for j in range(self.n_params):
                dS[i][j] = sp.diff(self.rhs_expr[i], self.p[j])
                for l in range(self.n_state_vars):
                    dS[i][j] = dS[i][j] + sp.diff(self.rhs_expr[i], self.y[l]) * S[l][j]

        # Flatten 1st order sensitivities for function
        [[fS1.append(dS[i][j]) for i in range(self.n_state_vars)]
         for j in range(self.n_params)]
        [[Ss.append(S[i][j]) for i in range(self.n_state_vars)]
         for j in range(self.n_params)]

        # dI/do
        self.dIdo = sp.diff(self.auxiliary_expression, self.y[self.open_state_index])

        self.func_S1 = sp.lambdify(inputs, fS1)
        # Define number of 1st order sensitivities
        self.n_state_var_sensitivities = self.n_params * self.n_state_vars

        # Concatenate RHS and 1st order sensitivities
        Ss = np.concatenate((list(self.y), Ss))
        fS1 = sp.Matrix(np.concatenate((self.frhs, fS1)))

        self.func_S1 = sp.lambdify(inputs, fS1)

        # Create Jacobian of the 1st order sensitivities function
        jS1 = fS1.jacobian(Ss)
        self.jfunc_S1 = sp.lambdify(inputs, jS1)

        def get_ic_sensitivty(state_no, param_no):
            sm = 0
            for i, r in enumerate(self.rates_dict):
                sm += sp.diff(self.rhs_inf_expr[state_no].subs(
                    self.v, self.voltage(0)), r) * rate_sens[i, param_no]
            return sm

        # Find sensitivity steady states at holding potential
        self.sensitivity_ics_expr = [get_ic_sensitivty(state_no, param_no)
                                     for param_no in range(len(self.p))
                                     for state_no in range(self.n_state_vars)]

        self.sensitivity_ics = sp.lambdify(self.p, self.sensitivity_ics_expr, cse=True)


    def rhs(self, t, y, p):
        """ Evaluates the RHS of the model (including sensitivities)

        """
        return self.func_rhs(y, p, self.voltage(t))

    def jrhs(self, t, y, p):
        """ Evaluates the jacobian of the RHS

            Having this function can speed up solving the system
        """
        return self.jfunc_rhs(*(*y, *p, self.voltage(t)))

    def get_linear_system(self, voltage=None, parameters=None):
        if voltage is None:
            voltage = self.voltage(0)
        if parameters is None:
            parameters = self.get_default_parameters()

        param_dict = dict(zip(self.symbols['p'], parameters))

        A_matrix = np.array(self.A.subs(self.rates_dict).subs(self.v, voltage).subs(param_dict)).astype(np.float64)
        B_vector = np.array(self.B.subs(self.rates_dict).subs(self.v, voltage).subs(param_dict)).astype(np.float64)

        return A_matrix, B_vector

    def get_steady_state(self, voltage=None, parameters=None):
        A, B = self.get_linear_system(voltage, parameters)
        steady_state = -np.linalg.solve(A, B)
        return steady_state

    def get_no_states(self):
        return self.A.shape[0] + 1

    def get_analytic_solution(self):
        """get_analytic_solution

        For any fixed voltage, we can easily compute an analytic solution for
        the system (not including sensitivities).

        TODO: Check that the matrix is well conditioned
        """

        # Solve non-homogeneous part
        X2 = -self.A.LUsolve(self.B.T)

        # Solve the homogenous part via diagonalisation
        P, D = self.A.diagonalize()

        # Consider the system dZ/dt = D Z
        # where X = CKZ, K is a diagonal matrix of constants and D is a diagonal matrix
        # with elements in the order given by linalg.eig(A) such that A = CDC^-1
        # Then Z = (e^{-D_i,i})_i and X=CKZ is the general homogenous solution to the system
        # dX/dt = AX because KdX/dt = CKdZ/dt = CKDZ = KCC^-1ACZ = KACZ = AKX

        IC = sp.Matrix(self.y)
        K = P.LUsolve(IC - X2)

        # solution = (C@K@np.exp(times*eigenvalues[:,None]) + X2[:, None]).T
        return P, K, D, X2, P.det()

    def get_analytic_solution_func(self, njitted=True, cond_threshold=None):

        rates_func = self.get_rates_func(njitted=njitted)

        times = self.times
        voltage = self.voltage(0)
        p = self.get_default_parameters()
        y0 = self.rhs_inf(p, voltage).flatten()

        if cond_threshold is None:
            cond_threshold = 1e5

        if y0.shape[0] == 1:
            A_func = sp.lambdify((self.rates_dict.keys(),), self.A[0, 0])
            B_func = sp.lambdify((self.rates_dict.keys(),), self.B[0, 0])

            if njitted:
                A_func = njit(A_func)
                B_func = njit(B_func)
            # Scalar case

            def analytic_solution_func_scalar(times=times, voltage=voltage, p=p, y0=y0):
                rates = rates_func(p, voltage).flatten()
                y0 = y0[0]
                a = A_func(rates)

                if a < 1 / cond_threshold:
                    return np.full((times.shape[0], 1), np.nan), False

                b = B_func(rates)
                sol = np.expand_dims((y0 + b/a) * np.exp(a * times) - b/a, -1)
                return sol, True

            analytic_solution_func = analytic_solution_func_scalar

        else:
            A_func = sp.lambdify((self.rates_dict.keys(),), self.A)
            B_func = sp.lambdify((self.rates_dict.keys(),), self.B)

            if njitted:
                A_func = njit(A_func)
                B_func = njit(B_func)
                # Q_func = njit(Q_func)

            def analytic_solution_func_matrix(times=times, voltage=voltage, p=p, y0=y0):
                rates = rates_func(p, voltage).flatten()
                _A = A_func(rates)
                _B = B_func(rates).flatten()

                try:
                    cond_A = np.linalg.norm(_A, 2) * np.linalg.norm(np.linalg.inv(_A), 2)
                except Exception:
                    return np.full((times.shape[0], y0.shape[0]), np.nan), False

                if cond_A > cond_threshold:
                    print("WARNING: cond_A = ", cond_A, " > ", cond_threshold)
                    print("matrix is poorly conditioned", cond_A, cond_threshold)
                    return np.full((times.shape[0], y0.shape[0]), np.nan), False

                D, P = np.linalg.eig(_A)

                # Compute condition number doi:10.1137/S00361445024180
                try:
                    cond_P = np.linalg.norm(P, 2) * np.linalg.norm(np.linalg.inv(P), 2)
                except Exception:
                    return np.full((times.shape[0], y0.shape[0]), np.nan), False

                if cond_P > cond_threshold:
                    print("WARNING: cond_P = ", cond_P, " > ", cond_threshold)
                    print("matrix is almost defective", cond_P, cond_threshold)
                    return np.full((times.shape[0], y0.shape[0]), np.nan), False

                X2 = -np.linalg.solve(_A, _B).flatten()

                K = np.diag(np.linalg.solve(P, (y0 - X2).flatten()))
                solution = (P @ K @  np.exp(np.outer(D, times))).T + X2.T

                return solution, True
            analytic_solution_func = analytic_solution_func_matrix

        return analytic_solution_func if not njitted else njit(analytic_solution_func)

    def make_Q_func(self, njitted=True):
        rates = tuple(self.rates_dict.keys())
        Q_func = sp.lambdify((rates,), self.Q, modules='numpy', cse=True)
        return njit(Q_func) if njitted else Q_func

    def make_ida_residual_func(self):
        if self.Q is None:
            Exception("Q Matrix not defined")

        Q_func = self.make_Q_func(True)
        rates_func = self.get_rates_func(True)

        voltage_func = self.voltage

        neq = self.Q.shape[0]

        n_p = len(self.get_default_parameters())

        @njit
        def f_deriv(t, u, p):
            V = voltage_func(t)
            rates = rates_func(p, V).flatten()
            Q = Q_func(rates)

            du = (Q @ u).flatten()
            return du

        # @njit
        # def f_deriv(t, u, p):
        #     V = voltage_func(t)
        #     k1, k2, k3, k4 = rates_func(p, V).flatten()

        #     O = u[1]
        #     I = u[2]
        #     IC = u[3]
        #     C = u[0]

        #     du = np.empty((4,))

        #     du[1] = -O * (k2 + k3) + k4 * I + k1 * C
        #     du[2] = -I * (k2 + k4) + k1 * IC + k3 * O
        #     du[3] = -IC * (k4 + k1) + k2 * I + k3 * C
        #     du[0] = -C * (k3 + k1) + k4 * IC + k2 * O

        #     return du

        @cfunc(ida_sig)
        def residual_func(t, u, du, res, p):
            y_vec = nb.carray(u, neq, dtype=np.float64)
            p_vec = nb.carray(p, n_p, dtype=np.float64)
            res_vec = nb.carray(res, neq, dtype=np.float64)
            dy_vec = nb.carray(du, neq, dtype=np.float64)

            # Calculate derivatives
            derivs = f_deriv(t, y_vec, p_vec)

            for i in range(neq):
                res_vec[i] = derivs[i] - dy_vec[i]

            res_vec[neq] = 1 - y_vec.sum()

            return None

        return residual_func

    def make_ida_solver_current(self, njitted=True, atol=None, rtol=None):
        state_solver = self.make_ida_solver_states(njitted=njitted, atol=atol, rtol=rtol)
        solver = self.make_solver_current(state_solver, njitted=njitted, rtol=rtol, atol=atol)
        return njit(solver) if njitted else solver

    def make_ida_jacobian_function(self):
        voltage = self.voltage

        n_p = len(self.get_default_parameters())
        neq = self.Q.shape[0]

        rates_func = self.get_rates_func(njitted=True)

        Q_func = self.make_Q_func(njitted=True)

        @cfunc(NumbaIDA.ida_jac_sig)
        def jac_func(t, cj, y, yp, JJ, p):
            jacobian = nb.carray(JJ, (neq, neq))
            p_vec = nb.carray(p, (n_p,))
            V = voltage(t)
            jacobian[:, :] = Q_func(rates_func(p_vec, V).flatten()).T - cj * np.eye(neq)

            return None

        return jac_func

    def make_ida_solver_states(self, njitted=True, atol=None, rtol=None):

        no_states = self.Q.shape[0]

        rhs_inf = self.rhs_inf

        voltage = self.voltage
        rates_func = self.get_rates_func(njitted=True)

        if atol is None:
            atol = self.solver_tolerances[0]
        if rtol is None:
            rtol = self.solver_tolerances[1]

        times = self.times
        res_func = self.make_ida_residual_func()

        func_ptr = res_func.address

        p = self.get_default_parameters()

        Q_func = self.make_Q_func(njitted=True)

        jac_func = self.make_ida_jacobian_function()

        protocol_description = self.protocol_description
        if protocol_description is None:
            Exception("No protocol defined")

        start_times = [val[0] for val in protocol_description]
        intervals = tuple(zip(start_times[:-1], start_times[1:]))

        jac_ptr = jac_func.address
        eps = np.finfo(float).eps

        def solver(p=p, times=times,
                   atol=atol, rtol=rtol):

            p = p.copy()
            y0 = np.zeros((no_states,))
            y0[:-1] = rhs_inf(p, voltage(0)).flatten()
            y0[-1] = 1 - np.sum(y0[:-1])
            solution = np.full((len(times), no_states), np.nan)
            solution[0, :] = y0

            v = voltage(0)
            rates = rates_func(p, v).flatten()

            for i, (tstart, tend) in enumerate(intervals):
                if i == len(intervals) - 1:
                    tend = times[-1] + 1

                istart = np.argmax(times > tstart)
                iend = np.argmax(times > tend)

                if iend == 0:
                    iend = len(times)

                step_times = np.full(iend-istart + 2, np.nan)
                if iend == len(times):
                    step_times[1:-1] = times[istart:]
                else:
                    step_times[1:-1] = times[istart:iend]

                step_times[0] = tstart
                step_times[-1] = tend

                step_sol = np.full((len(step_times), no_states), np.nan)

                if step_times[1] - tstart < 2 * eps * np.abs(step_times[1]):
                    start_int = 1
                    step_sol[0, :] = y0
                else:
                    start_int = 0

                if tend - step_times[-1] < 2 * eps * np.abs(tend):
                    end_int = -1
                else:
                    end_int = None

                v = voltage(tstart)
                rates = rates_func(p, v).flatten()

                Q = Q_func(rates)

                y0 = y0 / y0.sum()
                du0 = (Q.T @ y0).flatten()

                step_sol[start_int: end_int], success = ida(func_ptr, y0, du0, y0.shape[0],
                                                            step_times[start_int:end_int],
                                                            data=p,
                                                            # jac_ptr=jac_ptr,
                                                            nmaxsteps=2000)
                if not success:
                    return solution

                if end_int == -1:
                    step_sol[-1, :] = step_sol[-2, :]

                if iend == len(times):
                    solution[istart:, ] = step_sol[1:-1, ]
                    break

                else:
                    y0 = step_sol[-1, :]
                    solution[istart:iend, ] = step_sol[1:-1, ]
            return solution

        if njitted:
            solver = njit(solver)

        return solver

    def get_rates_func(self, njitted=True):
        inputs = (self.p, self.v)
        rates_expr = sp.Matrix(list(self.rates_dict.values()))

        rates_func = sp.lambdify(inputs, rates_expr)

        return njit(rates_func) if njitted else rates_func

    def make_hybrid_solver_states(self, protocol_description=None,
                                  njitted=False, analytic_solver=None,
                                  strict=True, cond_threshold=None, atol=None,
                                  rtol=None, hybrid=True):

        if protocol_description is None:
            if self.protocol_description is None:
                raise Exception("No protocol description has been provided")
            else:
                protocol_description = self.protocol_description

        crhs = self.get_cfunc_rhs()
        crhs_ptr = crhs.address

        no_states = len(self.B)

        if not analytic_solver:
            analytic_solver = self.get_analytic_solution_func(njitted=njitted,
                                                              cond_threshold=cond_threshold)

        rhs_inf = self.rhs_inf
        voltage = self.voltage

        if atol is None:
            atol = self.solver_tolerances[0]
        if rtol is None:
            rtol = self.solver_tolerances[1]

        times = self.times

        p = self.get_default_parameters()
        eps = np.finfo(float).eps

        start_times = [val[0] for val in protocol_description]
        intervals = tuple(zip(start_times[:-1], start_times[1:]))

        def hybrid_forward_solve(p=p, times=times, atol=atol, rtol=rtol,
                                 strict=strict, hybrid=hybrid):
            y0 = rhs_inf(p, voltage(.0)).flatten()

            solution = np.full((len(times), no_states), np.nan)
            solution[0, :] = y0

            for i, (tstart, tend) in enumerate(intervals):

                start_int = 0
                end_int = 0

                if i == len(intervals) - 1:
                    tend = times[-1] + 1
                istart = np.argmax(times > tstart)
                iend = np.argmax(times > tend)

                if iend == 0:
                    iend = len(times)

                vstart = protocol_description[i][2]
                vend = protocol_description[i][3]

                step_times = np.full(iend-istart + 2, np.nan)
                step_times = np.full(iend-istart + 2, np.nan)

                if iend == len(times):
                    step_times[1:-1] = times[istart:]
                else:
                    step_times[1:-1] = times[istart:iend]

                step_times[0] = tstart
                step_times[-1] = tend

                analytic_success = False
                step_sol = np.full((len(step_times), no_states), np.nan)

                if vstart == vend and hybrid:
                    step_sol[:, :], analytic_success = analytic_solver(step_times - tstart,
                                                                       vstart, p, y0)

                if not analytic_success:
                    step_times[0] = tstart
                    step_times[-1] = tend

                    if step_times[1] - tstart < 2 * eps * np.abs(step_times[1]):
                        start_int = 1
                        step_sol[0] = y0
                    else:
                        start_int = 0

                    t_offset = tstart
                    data = np.append(p, t_offset)
                    if tend - step_times[-1] < 2 * eps * np.abs(tend):
                        end_int = -1
                        step_sol[start_int: end_int], _ = lsoda(crhs_ptr, y0,
                                                                step_times[start_int:end_int] - step_times[0],
                                                                data=data, rtol=rtol,
                                                                atol=atol,
                                                                exit_on_warning=True)
                    else:
                        end_int = 0
                        step_sol[start_int:], _ = lsoda(crhs_ptr, y0,
                                                        step_times[start_int:] - step_times[0],
                                                        data=data, rtol=rtol,
                                                        atol=atol, exit_on_warning=True)

                if not np.all(np.isfinite(step_sol[start_int:end_int])):
                    return np.full(solution.shape, np.nan)

                if end_int == -1:
                    step_sol[-1, :] = step_sol[-2, :]

                if iend == len(times):
                    solution[istart:, ] = step_sol[1:-1, ]
                    break

                else:
                    y0 = step_sol[-1, :]
                    solution[istart:iend, ] = step_sol[1:-1, ]

            return solution

        return njit(hybrid_forward_solve) if njitted else hybrid_forward_solve

    def make_forward_solver_of_type(self, solver_type, **kws):
        if solver_type is None:
            solver = self.make_forward_solver_current(**kws)
        elif solver_type == 'default':
            solver = self.make_forward_solver_current(**kws)
        elif solver_type == 'hybrid':
            solver = self.make_hybrid_solver_current(**kws)
        elif solver_type == 'ida':
            solver = self.make_ida_solver_current(**kws)
        elif solver_type == 'dop853':
            solver = self.make_forward_solver_current(solver_type='dop853', **kws)
        else:
            raise Exception(f"Invalid solver type: {solver_type}")
        return solver

    def get_cfunc_rhs(self):
        rhs = nb.njit(self.func_rhs)
        voltage = self.voltage

        ny = len(self.state_labels) - 1
        np = len(self.get_default_parameters())

        @cfunc(lsoda_sig)
        def crhs(t, y, dy, data):
            y = nb.carray(y, ny)
            dy = nb.carray(dy, ny)
            data = nb.carray(data, np + 1)

            p = data[:-1]
            t_offset = data[-1]

            res = rhs(y,
                      p,
                      voltage(t, offset=t_offset)).flatten()

            dy[:] = res

        return crhs

    def make_hybrid_solver_current(self, protocol_description=None,
                                   njitted=True, strict=True,
                                   cond_threshold=None, atol=None, rtol=None,
                                   hybrid=True):
        hybrid_solver =\
            self.make_hybrid_solver_states(protocol_description=protocol_description,
                                           njitted=njitted, strict=strict,
                                           cond_threshold=cond_threshold,
                                           atol=atol, rtol=rtol, hybrid=hybrid)

        auxiliary_function = self.auxiliary_function

        if njitted:
            auxiliary_function = njit(auxiliary_function)

        times = self.times

        atol, rtol = self.solver_tolerances
        voltage_func = self.voltage

        params = self.get_default_parameters()

        def hybrid_forward_solve(p=params, times=times, atol=atol, rtol=rtol,
                                 hybrid=hybrid):
            voltages = np.empty(len(times))
            for i in range(len(times)):
                voltages[i] = voltage_func(times[i])

            states = hybrid_solver(p, times=times, hybrid=hybrid, atol=atol, rtol=rtol)

            return (auxiliary_function(states.T, p, voltages)).flatten()

        return njit(hybrid_forward_solve) if njitted else hybrid_forward_solve

    def make_forward_solver_current(self, voltages=None, njitted=True,
                                    protocol_description=None,
                                    solver_type='lsoda', atol=None, rtol=None):

        solver_states = self.make_hybrid_solver_states(njitted=njitted,
                                                       protocol_description=protocol_description,
                                                       atol=atol, rtol=rtol,
                                                       hybrid=False)

        return self.make_solver_current(solver_states, voltages=voltages,
                                        atol=atol, rtol=rtol, njitted=njitted)

    def make_solver_current(self, solver_states, voltages=None, atol=None,
                            rtol=None, njitted=False):
        if atol is None:
            atol = self.solver_tolerances[0]

        if rtol is None:
            rtol = self.solver_tolerances[1]

        if voltages is None:
            voltages = self.GetVoltage()

        times = self.times

        default_parameters = self.get_default_parameters()

        auxiliary_function = self.auxiliary_function

        def forward_solver(p=default_parameters, times=times, voltages=voltages, atol=atol, rtol=rtol):
            states = solver_states(p, times, atol, rtol)
            return (auxiliary_function(states.T, p, voltages)).flatten()

        return forward_solver

    def solve_rhs(self, p=None, times=None):
        """ Solve the RHS of the system and return the open state probability at each timestep
        """

        if times is None:
            times = self.times
        if p is None:
            p = self.get_default_parameters()
        p = np.array(p)

        y0 = self.rhs_inf(p, self.voltage(0))

        sol = solve_ivp(self.rhs,
                        (times[0], times[-1]),
                        y0.flatten(),
                        t_eval=times,
                        atol=self.solver_tolerances[0],
                        rtol=self.solver_tolerances[1],
                        jac=self.jrhs,
                        method='LSODA',
                        args=(p,))
        return sol

    def get_rhs_func(self, njitted=False):
        return self.func_rhs if not njitted else njit(self.func_rhs)

    def count_rhs_evaluations(self, p, times=None):

        if times is None:
            times = self.times

        y0 = self.rhs_inf(p, self.voltage(0)).flatten()

        evals = 0
        rhs_func = self.rhs

        class rhs_counter():
            evals = 0

            def func(self, t, y, *args):
                self.evals += 1
                return rhs_func(t, y, p)

        rhs_count = rhs_counter()
        # Chop off RHS
        sol = solve_ivp(lambda t, y, *args: rhs_count.func(t, y, *args),
                        (times[0], times[-1]),
                        y0,
                        # t_eval = times,
                        atol=self.solver_tolerances[0],
                        rtol=self.solver_tolerances[1],
                        # Dfun=self.jrhs,
                        args=(p,),
                        method='LSODA')
        return rhs_count.evals

    def drhs(self, t, y, p):
        """ Evaluate RHS analytically

        """
        return self.func_S1(*(*y[:self.n_state_vars], *p, self.voltage(t), *y[self.n_state_vars:]))

    def jdrhs(self, t, y, p):
        """  Evaluates the jacobian of the RHS (analytically)

        This allows the system to be solved faster

        """
        return self.jfunc_S1(*(*
                               y[:self.n_state_vars], *
                               p, self.voltage(t), *
                               y[self.n_state_vars:]))

    # Returns the open state 1st order sensitivities
    def solve_drhs(self, p, times=None):
        """Solve the RHS of the system and return the open state sensitivities at each
        timestep

        Returns only the sensitivities, not the state variables
        """

        if times is None:
            times = self.times

        y0 = self.rhs_inf(p, self.voltage(0))
        dy0 = self.sensitivity_ics(*p)

        ics = np.concatenate(y0, dy0, axis=None)

        # Chop off RHS
        drhs = solve_ivp(self.drhs,
                         (times[0], times[-1]),
                         ics,
                         t_eval=times,
                         atol=self.solver_tolerances[0],
                         rtol=self.solver_tolerances[1],
                         Dfun=self.jdrhs,
                         method='LSODA',
                         args=(p,
                               )).y[:, self.n_state_vars:]
        # Return only open state sensitivites
        return drhs[:, self.open_state_index::self.n_state_vars]

    def solve_drhs_full(self, p, times=None):
        """ Solve the system numerically for every time in self.times

        """
        if times is None:
            times = self.times

        y0 = np.array(self.rhs_inf(p, self.voltage(0))).astype(np.float64)
        dy0 = np.array(self.sensitivity_ics(*p)).astype(np.float64)

        step_y0 = np.concatenate((y0, dy0), axis=None).astype(np.float64)

        solution = solve_ivp(self.drhs,
                             (times[0], times[-1]),
                             step_y0,
                             t_eval=times,
                             atol=self.solver_tolerances[0],
                             rtol=self.solver_tolerances[1],
                             jac=self.jdrhs,
                             method='LSODA',
                             args=(p,))['y'].T

        assert(solution.shape[0] == len(times))
        return solution

    def voltage(self, t):
        raise NotImplementedError

    def SimulateForwardModel(self, p=None, times=None):
        if p is None:
            p = self.get_default_parameters()
        p = np.array(p)

        if times is None:
            times = self.times
        return self.make_forward_solver_current(njitted=False)(p, times)

    def GetVoltage(self, times=None):
        """
        Returns the voltage at every timepoint

        By default, there is a timestep every millisecond up to self.tmax
        """
        if times is None:
            times = self.times

        v = np.array([self.voltage(t) for t in times])
        return v

    def SimulateForwardModelSensitivities(self, p=None, times=None):
        """
        Solve the model for a given set of parameters

        Returns the state variables and current sensitivities at every timestep

        """

        self.setup_sensitivities()

        if p is None:
            p = self.get_default_parameters()

        if times is None:
            times = self.times

        solution = self.solve_drhs_full(p, times)

        # Get the open state sensitivities for each parameter
        index_sensitivities = self.n_state_vars + self.open_state_index + \
            self.n_state_vars * np.array(range(self.n_params))
        sensitivities = solution[:, index_sensitivities]
        o = solution[:, self.open_state_index]

        voltages = self.GetVoltage(times=times)

        current = p[self.GKr_index] * o * (voltages - self.E_rev)

        dIdo = (voltages - self.E_rev) * p[-1]
        values = sensitivities * dIdo[:, None]
        values[:, self.GKr_index] += o * (voltages - self.E_rev)

        return current, values

    def get_no_parameters(self):
        return len(self.get_default_parameters())

    def set_tolerances(self, abs_tol, rel_tol):
        self.solver_tolerances = (abs_tol, rel_tol)
