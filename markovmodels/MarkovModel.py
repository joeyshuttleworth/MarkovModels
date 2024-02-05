import numba as nb
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from numba import cfunc, njit
from numbalsoda import lsoda, lsoda_sig

from markovmodels.utilities import calculate_reversal_potential
from markovmodels.ODEModel import ODEModel


class MarkovModel(ODEModel):
    """
    A class describing a Markov Model of an ion channel

    """

    def get_default_parameters(self):
        return self.default_parameters.copy()

    def get_model_name(self):
        return self.model_name

    def get_parameter_labels(self):
        return self.parameter_labels.copy()

    def get_state_labels(self):
        if self.state_labels:
            return self.state_labels.copy()[:self.get_no_state_vars()]
        else:
            return ['state_%i' % i for i in range(self.get_no_state_vars())]

    def __init__(self, symbols, A, B, rates_dict, times=None, voltage=None,
                 tolerances=(1e-8, 1e-8), Q=None, protocol_description=None,
                 name=None, E_rev=None, default_parameters=None,
                 parameter_labels=None, GKr_index: int = None,
                 open_state_index: int = None, transformations=None,
                 state_labels: str = None):

        self.name = name

        if state_labels:
            self.state_labels = state_labels

        if open_state_index is not None:
            self.open_state_index = open_state_index

        self.n_states = A.shape[0] + 1
        self.n_state_vars = A.shape[0]

        if default_parameters is not None:
            self.default_parameters = default_parameters

        if GKr_index is None:
            # assume the last parameter is GKr (conductance)
            GKr_index = len(self.default_parameters - 1)
        self.GKr_index = GKr_index

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

        self.initial_condition = np.full(len(self.y), .0)

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

        self.func_rhs = sp.lambdify((self.y, self.p, self.v), self.rhs_expr, cse=True)

        # Create Jacobian of the RHS function
        jrhs = sp.Matrix(self.rhs_expr).jacobian(self.y)
        self.jfunc_rhs = sp.lambdify((self.y, self.p, self.v), jrhs)

        self.compute_steady_state_expressions()

        self.auxiliary_function = njit(self.define_auxiliary_function())

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

        return self.rhs_inf, self.rhs_inf_expr, self.current_inf, self.current_inf_expr

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
