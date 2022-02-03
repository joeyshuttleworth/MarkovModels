import numpy as np
import sympy as sp
import logging
from scipy.integrate import solve_ivp
from NumbaLSODA import lsoda_sig, lsoda
from numba import njit, cfunc, literal_unroll
import numba as nb


class MarkovModel:
    """
    A class containing describing a Markov Model of an ion channel

    Arguments are provided using the Params class

    """

    def get_default_parameters(self):
        raise NotImplementedError

    def __init__(self, symbols, A, B, rates_dict, times, voltage=None,
                 tolerances=(1e-8, 1e-10)):

        self.window_locs = None
        self.protocol_description = None

        try:
            self.y = symbols['y']
            self.p = symbols['p']
            self.v = symbols['v']
        except:
            raise Exception()

        self.rates_dict = rates_dict

        # (atol, rtol)
        self.solver_tolerances = tuple(tolerances)

        self.symbols = symbols
        # The timesteps we want to output at

        self.times = times
        self.A = A
        self.B = B

        self.rhs_expr = (A @ self.y + B).subs(rates_dict)

        if voltage is not None:
            self.voltage = voltage

        # Inputs for RHS ODEs
        inputs = list(self.y) + list(self.p) + [self.v]

        # Create RHS function
        frhs = np.array([e for e in self.rhs_expr])
        self.func_rhs = sp.lambdify(inputs, frhs)

        # Create Jacobian of the RHS function
        jrhs = sp.Matrix(self.rhs_expr).jacobian(self.y)
        self.jfunc_rhs = sp.lambdify(inputs, jrhs)

        # Create symbols for 1st order sensitivities
        dydp = [
            [
                sp.symbols(
                    'dy%d' %
                    i + 'dp%d' %
                    j) for j in range(
                    self.n_params)] for i in range(
                self.n_state_vars)]

        # Append 1st order sensitivities to inputs
        for i in range(self.n_params):
            for j in range(self.n_state_vars):
                inputs.append(dydp[j][i])

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

        self.auxillary_expression = self.p[self.GKr_index] * \
            self.y[self.open_state_index] * (self.v - self.Erev)

        # dI/do
        self.dIdo = sp.diff(self.auxillary_expression, self.y[self.open_state_index])

        self.func_S1 = sp.lambdify(inputs, fS1)
        # Define number of 1st order sensitivities
        self.n_state_var_sensitivities = self.n_params * self.n_state_vars

        # Concatenate RHS and 1st order sensitivities
        Ss = np.concatenate((list(self.y), Ss))
        fS1 = np.concatenate((frhs, fS1))

        self.func_S1 = sp.lambdify(inputs, fS1)

        # Create Jacobian of the 1st order sensitivities function
        jS1 = sp.Matrix(fS1).jacobian(Ss)
        self.jfunc_S1 = sp.lambdify(inputs, jS1)

        # Set the initial conditions of the model and the initial sensitivities
        # by finding the steady state of the model

        # TODO:
        # Check that a steady state exists for this system at this
        # voltage
        # dx/dt = Ax+B

        A_matrix, _ = self.get_linear_system()

        # eigvals = np.linalg.eig(A_matrix)[0]
        # assert(np.all(eigvals < 0))

        # Can be found analytically
        self.rhs_inf_expr = -self.A.LUsolve(self.B).subs(rates_dict)
        # self.rhs_inf = lambda p, v: np.array(self.rhs_inf_expr.subs(dict(zip(self.p, p))).subs('v', v)).astype(np.float64)
        self.rhs_inf = nb.njit(sp.lambdify((self.p, 'v'), self.rhs_inf_expr, modules='numpy'))

        self.current_inf_expr = self.auxillary_expression.subs(self.y, self.rhs_inf)
        self.current_inf = lambda p: np.array(self.current_inf_expr.subs(
            dict(zip(self.p, p))).evalf()).astype(np.float64)

        # Find sensitivity steady states at holding potential
        self.sensitivity_ics_expr = sp.Matrix([sp.diff(self.rhs_inf_expr[i].subs(
            'v', self.holding_potential), self.p[j]) for j in range(self.n_params) for i in range(self.n_state_vars)])

        self.sensitivity_ics = sp.lambdify(self.p, self.sensitivity_ics_expr)

    def rhs(self, t, y, p):
        """ Evaluates the RHS of the model (including sensitivities)

        """
        return self.func_rhs(*(*y, *p, self.voltage(t)))

    def jrhs(self, t, y, p):
        """ Evaluates the jacobian of the RHS

            Having this function can speed up solving the system
        """
        return self.jfunc_rhs(*(*y, *p, self.voltage(t)))

    def get_linear_system(self, voltage=None, parameters=None):
        if voltage is None:
            voltage = self.holding_potential
        if parameters is None:
            parameters = self.get_default_parameters()

        param_dict = dict(zip(self.symbols['p'], parameters))
        A_matrix = np.array(self.A.subs(self.rates_dict).subs(self.v, voltage).subs(param_dict)).astype(np.float64)
        B_vector = np.array(self.B.subs(self.rates_dict).subs(self.v, voltage).subs(param_dict)).astype(np.float64)

        cond = np.linalg.cond(A_matrix)
        logging.info("A matrix condition number is {} for voltage = {}".format(cond, voltage))
        if cond > 1e6:
            logging.warning("Condition number is {} for voltage = {}".format(cond, voltage))

        return A_matrix, B_vector

    def get_steady_state(self, voltage=None, parameters=None):
        A, B = self.get_linear_system(voltage, parameters)
        steady_state = -np.linalg.solve(A, B)
        return steady_state

    def get_no_states(self):
        return len(self.y)

    def get_analytic_solution(self):
        """get_analytic_solution

        For any fixed voltage, we can easily compute an analytic solution for
        the system (not including sensitivities).

        TODO: Check that the matrix is well conditioned
        """

       # Solve non-homogeneous part
        X2 = -self.A.LUsolve(self.B)

        # Solve the homogenous part via diagonalisation
        # eigenvalues, C = np.linalg.eig(self.A)

        eigen_list = list(zip(*self.A.eigenvects()))
        eigenvalues = sp.Matrix(eigen_list[0])
        eigen_vects = eigen_list[2]
        C = sp.Matrix(np.column_stack([vec[0] for vec in eigen_vects]))

        D = sp.matrices.diag(*eigenvalues)

        # Consider the system dZ/dt = D Z
        # where X = CKZ, K is a diagonal matrix of constants and D is a diagonal matrix
        # with elements in the order given by linalg.eig(A) such that A = CDC^-1
        # Then Z = (e^{-D_i,i})_i and X=CKZ is the general homogenous solution to the system
        # dX/dt = AX because dX/dt = CKdZ/dt = CKDZ = KCC^-1ACZ = KACZ = AKX

        IC = sp.Matrix(['rhs%i' % i for i in range(self.get_no_states())])
        IC_KZ = C.LUsolve(IC - X2)
        K = sp.matrices.diag(*IC_KZ)

        # solution = (C@K@np.exp(times*eigenvalues[:,None]) + X2[:, None]).T

        return C @ K, eigenvalues, X2

    def get_rates_func(self, njitted=True):

        n = len(self.rates_dict)

        inputs = (self.p, self.v)
        rates_expr = sp.Matrix(list(self.rates_dict.values()))

        rates_func = sp.lambdify(inputs, rates_expr)

        return njit(rates_func) if njitted else rates_func

    def get_analytic_solver(self, njitted=True):
        expressions = self.get_analytic_solution()

        rates_func = self.get_rates_func()

        rhs_names = ["rhs%i" % i for i in range(self.get_no_states())]

        args = (sp.Matrix(list(self.rates_dict.keys())), sp.Matrix(rhs_names))

        CK_func, eigval_func, X2_func = tuple([njit(sp.lambdify(args, expr))
                                               for expr in expressions])

        def analytic_solver(times, voltage, p, rhs0):
            rates = rates_func(p, voltage).flatten()
            CK = CK_func(rates, rhs0)
            eigvals = eigval_func(rates, rhs0)
            X2 = X2_func(rates, rhs0)

            sol = CK @ np.exp(np.outer(eigvals, times)) + X2
            return sol.T

        return njit(analytic_solver) if njitted else analytic_solver

    def make_forward_solver_states(self, atol=None, rtol=None, protocol_description=None, njitted=True):

        if atol is None:
            atol = self.solver_tolerances[0]
        if rtol is None:
            rtol = self.solver_tolerances[1]

        crhs = self.get_cfunc_rhs()

        crhs_ptr = crhs.address

        rhs_inf = self.rhs_inf

        voltage = self.voltage

        no_states = self.get_no_states()

        times = self.times

        if protocol_description is None:
            if self.protocol_description is None:
                raise Exception("No protocol description has been provided")
            else:
                protocol_description = self.protocol_description

        def forward_solver(p, times=times, atol=atol, rtol=rtol):
            rhs0 = rhs_inf(p, voltage(0)).flatten()
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
            return solution

        return njit(forward_solver) if njitted else forward_solver

    def get_cfunc_rhs(self):
        rhs = nb.njit(self.func_rhs)
        voltage = self.voltage

        @cfunc(lsoda_sig)
        def crhs(t, y, dy, p):
            res = rhs(y[0], y[1], y[2],
                      p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8],
                      voltage(t))
            for i in range(3):
                dy[i] = res[i]

        return crhs

    def make_hybrid_solver_states(self, protocol_description=None, njitted=True):

        if protocol_description is None:
            if self.protocol_description is None:
                raise Exception("No protocol description has been provided")
            else:
                protocol_description = self.protocol_description

        crhs = self.get_cfunc_rhs()
        crhs_ptr = crhs.address

        no_states = len(self.B)
        analytic_solver = self.get_analytic_solver()
        rhs_inf = self.rhs_inf
        voltage = self.voltage
        atol, rtol = self.solver_tolerances
        times = self.times

        def hybrid_forward_solve(p, times=times, atol=atol, rtol=rtol):
            rhs0 = rhs_inf(p, voltage(0)).flatten()

            solution = np.full((len(times), no_states), np.nan)
            solution[0] = rhs0

            for tstart, tend, vstart, vend in protocol_description:
                istart = np.argmax(times >= tstart)
                iend = np.argmax(times > tend)

                if iend == 0:
                    iend = len(times)

                step_times = np.full(iend-istart+2, np.nan)
                step_times[0] = tstart
                step_times[-1] = tend
                if iend == 0:
                    step_times[1:-1] = times[istart:]
                    iend = len(times)
                else:
                    step_times[1:-1] = times[istart:iend]

                # Analytic solve
                if vstart == vend:
                    step_times = step_times - step_times[0]
                    step_sol = analytic_solver(step_times, vstart, p, rhs0)

                # numerical solve
                else:
                    step_sol, _ = lsoda(crhs_ptr, rhs0, step_times, data=p,
                                        rtol=rtol, atol=atol)
                if iend == len(times):
                    solution[istart:, ] = step_sol[1:-1, ]
                    break

                else:
                    rhs0 = step_sol[-1, :]
                    solution[istart:iend, ] = step_sol[1:-1, :]
            return solution

        return njit(hybrid_forward_solve) if njitted else hybrid_forward_solve

    def make_hybrid_solver_current(self, protocol_description=None, njitted=True):
        hybrid_solver = self.make_hybrid_solver_states(protocol_description=protocol_description, njitted=njitted)
        open_index = self.open_state_index
        Erev = self.Erev
        gkr_index = self.GKr_index

        if protocol_description is None:
            if self.protocol_description is None:
                # TODO
                raise Exception()
            else:
                protocol_description = self.protocol_description

        times = self.times

        atol, rtol = self.solver_tolerances
        voltage_func = self.voltage

        def hybrid_forward_solve(p, times=times, atol=atol, rtol=rtol):
            voltages = np.empty(len(times))
            for i in range(len(times)):
                voltages[i] = voltage_func(times[i])

            states = hybrid_solver(p, times=times)
            return ((states[:, open_index] * p[gkr_index]) * (voltages - Erev)).flatten()

        return njit(hybrid_forward_solve) if njitted else hybrid_forward_solve

    def make_forward_solver_current(self, voltages=None, atol=None, rtol=None, njitted=True):
        if atol is None:
            atol = self.solver_tolerances[0]

        if rtol is None:
            rtol = self.solver_tolerances[1]

        solver_states = self.make_forward_solver_states(atol=atol, rtol=rtol, njitted=True)

        gkr_index = self.GKr_index
        open_index = self.open_state_index
        Erev = self.Erev

        if voltages is None:
            voltages = self.GetVoltage()

        times = self.times

        def forward_solver(p, times=times, voltages=voltages, atol=atol, rtol=rtol):
            states = solver_states(p, times, atol, rtol)
            return ((states[:, open_index] * p[gkr_index]) * (voltages - Erev)).flatten()

        return njit(forward_solver) if njitted else forward_solver

    def solve_rhs(self, p=None, times=None):
        """ Solve the RHS of the system and return the open state probability at each timestep
        """

        if times is None:
            times = self.times
        if p is None:
            p = self.get_default_parameters()
        p = np.array(p)

        rhs0 = self.rhs_inf(p, self.holding_potential)

        sol = solve_ivp(self.rhs,
                        (times[0], times[-1]),
                        rhs0.flatten(),
                        t_eval=times,
                        atol=self.solver_tolerances[0],
                        rtol=self.solver_tolerances[1],
                        jac=self.jrhs,
                        method='LSODA',
                        args=(p,))
        return sol

    def count_rhs_evaluations(self, p, times=None):

        if times is None:
            times = self.times

        rhs0 = self.rhs_inf(p, self.holding_potential).flatten()

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
                        rhs0,
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

        rhs0 = self.rhs_inf(p, self.holding_potential)
        drhs0 = self.sensitivity_ics(*p)

        ics = np.concatenate(rhs0, drhs0, axis=None)

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

        rhs0 = np.array(self.rhs_inf(p, self.holding_potential)).astype(np.float64)
        drhs0 = np.array(self.sensitivity_ics(*p)).astype(np.float64)

        step_rhs0 = np.concatenate((rhs0, drhs0), axis=None).astype(np.float64)

        solution = []

        # Solving for each step in the protocol is faster and more accurate
        for tstart, tend in zip(self.window_locs, self.window_locs[1:]):
            t_eval = [tstart] + [t for t in times if t >= tstart and t < tend] + [tend]
            t_eval = np.unique(t_eval)
            step_sol = solve_ivp(
                self.drhs,
                (tstart, tend),
                step_rhs0,
                t_eval=t_eval,
                atol=self.solver_tolerances[0],
                rtol=self.solver_tolerances[1],
                jac=self.jdrhs,
                method='LSODA',
                args=(p,))['y'].T

            if t_eval[0] != tstart:
                step_sol = step_sol[1:, :]

            if t_eval[-1] != tend:
                step_sol = step_sol[:-2, :]
            else:
                step_sol = step_sol[:-1, :]

            solution.append(step_sol[:-1])
            step_rhs0 = step_sol[-1, :]

        t_eval = [t for t in times if t > self.window_locs[-1]]

        if len(t_eval) > 0:
            step_sol = solve_ivp(
                self.drhs,
                (tstart, times[-1]),
                step_rhs0,
                t_eval=t_eval,
                atol=self.solver_tolerances[0],
                rtol=self.solver_tolerances[1],
                jac=self.jdrhs,
                method='LSODA',
                args=(p,))['y'].T
            solution.append(step_sol)

        else:
            solution.append(step_rhs0[None, :])

        solution = np.concatenate(solution, axis=0)

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

        print(p, self.voltage(0))
        return self.make_forward_solver_current(njitted=False)(p, times)

    def GetStateVariables(self, p=None):

        if p is None:
            p = self.get_default_parameters()

        states = self.solve_rhs(p)['y'].T
        state1 = np.array([1.0 - np.sum(row) for row in states])
        state1 = state1.reshape(len(state1), 1)
        states = np.concatenate((states, state1), axis=1)
        return states

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

        if p is None:
            p = self.get_default_parameters()

        if times is None:
            times = self.times

        solution = self.solve_drhs_full(p, times=times)

        # Get the open state sensitivities for each parameter
        index_sensitivities = self.n_state_vars + self.open_state_index + \
            self.n_state_vars * np.array(range(self.n_params))
        sensitivities = solution[:, index_sensitivities]
        o = solution[:, self.open_state_index]

        voltages = self.GetVoltage(times=times)

        current = p[self.GKr_index] * o * (voltages - self.Erev)

        dIdo = (voltages - self.Erev) * p[-1]
        values = sensitivities * dIdo[:, None]
        values[:, self.GKr_index] += o * (voltages - self.Erev)

        return current, values

    def get_no_parameters(self):
        return len(self.get_default_parameters())

    def set_tolerances(self, abs_tol, rel_tol):
        self.solver_tolerances = (abs_tol, rel_tol)
