import numpy as np
import sympy as sp
import logging
from scipy.integrate import odeint


class MarkovModel:
    """
    A class containing describing a Markov Model of an ion channel

    Arguments are provided using the Params class

    """

    solver_tolerances = (1e-5, 1e-7)

    def get_default_parameters():
        raise NotImplementedError

    def __init__(self, symbols, A, B, times, voltage=None):

        try:
            self.y = symbols['y']
            self.p = symbols['p']
            self.v = symbols['v']
        except:
            raise Exception()

        self.symbols = symbols
        # The timesteps we want to output at

        self.times = times
        self.A = A
        self.B = B

        rhs = A @ self.y + B

        if voltage is not None:
            self.voltage = voltage

        para = self.get_default_parameters()

        # Inputs for RHS ODEs
        inputs = list(self.y) + list(self.p) + [self.v]

        # Create RHS function
        frhs = [rhs[i] for i in range(self.n_state_vars)]
        self.func_rhs = sp.lambdify(inputs, frhs)

        # Create Jacobian of the RHS function
        jrhs = sp.Matrix(rhs).jacobian(sp.Matrix(self.y))
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
                dS[i][j] = sp.diff(rhs[i], self.p[j])
                for l in range(self.n_state_vars):
                    dS[i][j] = dS[i][j] + sp.diff(rhs[i], self.y[l]) * S[l][j]

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

        # Create Jacobian of the 1st order sensitivities function
        jS1 = sp.Matrix(fS1).jacobian(sp.Matrix(Ss))
        self.jfunc_S1 = sp.lambdify(inputs, jS1)

        # Set the initial conditions of the model and the initial sensitivities
        # by finding the steady state of the model

        # TODO:
        # Check that a steady state exists for this system at this
        # holding voltage
        # dx/dt = Ax+B


        A_matrix, _ = self.get_linear_system()

        eigvals = np.linalg.eig(A_matrix)[0]
        assert(np.all(eigvals < 0))

        # Can be found analytically
        self.rhs_inf_expr = -self.A.LUsolve(self.B).subs(self.v, self.holding_potential)
        self.rhs_inf = lambda p : self.rhs_inf_expr.subs(dict(zip(self.p, p))).evalf()

        param_dict = dict([('p{}'.format(i), para[i]) for i in range(self.n_params)])

        self.current_inf_expr = self.auxillary_expression.subs(self.y, self.rhs_inf)
        self.current_inf = lambda p : np.array(self.current_inf_expr.subs(dict(zip(self.p, p))).evalf()).astype(np.float64)

        # Find sensitivity steady states
        self.sensitivity_ics_expr = sp.Matrix( [ sp.diff(
            self.rhs_inf_expr[i], self.p[j]) for j in range( 0, self.n_params)
                                                        for i in range( 0, self.n_state_vars)])
        self.sensitivity_ics = lambda p : np.array(self.sensitivity_ics_expr.subs(dict(zip(self.p, p))).evalf()).astype(np.float64)

    def rhs(self, y, t, p):
        """ Evaluates the RHS of the model (including sensitivities)

        """
        return self.func_rhs(*(*y, *p, self.voltage(t)))

    def jrhs(self, y, t, p):
        """ Evaluates the jacobian of the RHS

            Having this function can speed up solving the system
        """
        return self.jfunc_rhs(*(*y, *p, self.voltage(t)))

    def get_linear_system(self, voltage=None, parameters=None):
        if voltage is None:
            voltage = self.holding_potential
        if parameters is None:
            parameters = self.get_default_parameters()
        param_dict = dict([('p{}'.format(i), parameters[i]) for i in range(self.n_params)])

        A_matrix = np.array([float(el.evalf()) for el in self.A.subs(
            self.symbols['v'], voltage).subs(param_dict)]).reshape(self.A.rows, self.A.cols)

        B_vector = np.array([float(el.evalf()) for el in self.B.subs(
            self.symbols['v'], voltage).subs(param_dict)])


        cond = np.linalg.cond(A_matrix)
        logging.info("A matrix condition number is {} for voltage = {}".format(cond, voltage))
        assert(cond < 1e6)

        return A_matrix, B_vector

    def get_steady_state(self, voltage=None, parameters=None):
        A,B = self.get_linear_system(voltage,parameters)
        steady_state = -np.linalg.solve(A,B)
        return steady_state

    def get_analytic_solution(self, voltage=None, times=None, parameters=None, rhs0 = None):
        """get_analytic_solution

        For any fixed voltage, we can easily compute an analytic solution for
        the system (not including sensitivities).

        TODO: Check that the matrix is well conditioned
        """

        if times is None:
            times = self.times
        times = np.array(times)

        if type(times) is not np.ndarray:
            raise TypeError("times is type {}".format(type(times)))

        if voltage is None:
            voltage=self.holding_potential

        if parameters is None:
            parameters = self.get_default_parameters()

        if rhs0 is None:
            rhs0 = self.rhs_inf(parameters)

        #Solve non-homogeneous part
        A_matrix, B_vector = self.get_linear_system(voltage=voltage, parameters=parameters)

        X2 = -np.linalg.solve(A_matrix, B_vector)

        # Solve the homogenous part via diagonalisation
        eigenvalues, C = np.linalg.eig(A_matrix)
        D = np.diag(eigenvalues)

        # Consider the system dZ/dt = D Z
        # where X = CKZ, K is a diagonal matrix of constants and D is a diagonal matrix
        # with elements in the order given by linalg.eig(A) such that A = CDC^-1
        # Then Z = (e^{-D_i,i})_i and X=CKZ is the general homogenous solution to the system
        # dX/dt = AX because dX/dt = CKdZ/dt = CKDZ = KCC^-1ACZ = KACZ = AKX

        IC = np.array(rhs0).astype(np.float64)
        IC_KZ = np.linalg.solve(C, IC - X2)
        K =  np.diag(IC_KZ)
        solution = (C@K@np.exp(times*eigenvalues[:,None]) + X2[:, None]).T

        # Apply auxiliary function to solution
        return solution[:, self.open_state_index]*parameters[self.GKr_index]*(voltage - self.Erev)


    # Returns the open state
    def solve_rhs(self, p, times=None):
        """ Solve the RHS of the system and return the open state probability at each timestep
        """
        rhs0 = np.array(self.rhs_inf(p)).astype(np.float64)[:,0]

        if times is None:
            times = self.times

        return odeint(
            self.rhs,
            rhs0,
            times,
            atol=self.solver_tolerances[0],
            rtol=self.solver_tolerances[1],
            Dfun=self.jrhs,
            args=(
                p,
            ))

    def drhs(self, y, t, p):
        """ Evaluate RHS analytically

        """
        outputs = self.func_rhs(
            *y[:self.n_state_vars], *p, self.voltage(t))
        outputs.extend(self.func_S1(*(*
                                     y[:self.n_state_vars], *
                                     p, self.voltage(t), *
                                     y[self.n_state_vars:])))
        return outputs

    def jdrhs(self, y, t, p):
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

        rhs0 = self.rhs_inf(p)
        drhs0 = self.sensitivity_ics(p)

        ics = np.concatenate(rhs0, drhs0, axis=None).astype(np.float64)

        # Chop off RHS
        drhs = odeint(self.drhs,
                      self.ics,
                      times,
                      atol=self.solver_tolerances[0],
                      rtol=self.solver_tolerances[1],
                      Dfun=self.jdrhs,
                      args=(p,
                            ))[:,
                               self.n_state_vars:]
        # Return only open state sensitivites
        return drhs[:, self.open_state_index::self.n_state_vars]

    def solve_drhs_full(self, p, times=None):
        """ Solve the system numerically for every time in self.times

        """
        if times is None:
            times = self.times

        rhs0 = np.array(self.rhs_inf(p)).astype(np.float64)
        drhs0 = np.array(self.sensitivity_ics(p)).astype(np.float64)

        return odeint(
            self.drhs,
            np.concatenate((rhs0, drhs0), axis=None).astype(np.float64),
            times,
            atol=self.solver_tolerances[0],
            rtol=self.solver_tolerances[1],
            Dfun=self.jdrhs,
            args=(
                p,
            ))

    def voltage(self, t):
        raise NotImplementedError
        return V

    def SimulateForwardModel(self, p=None, times=None):
        if p is None:
            p = self.get_default_parameters()
        o = self.solve_rhs(p, times)[:, self.open_state_index]
        voltages = self.GetVoltage(times=times)
        return p[8] * o * (voltages - self.Erev)

    def GetStateVariables(self, p):
        states = self.solve_rhs(p)

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

        Used by pints
        """

        if p is None:
            p = self.get_default_parameters()

        solution = self.solve_drhs_full(p, times=times)

        # Get the open state sensitivities for each parameter
        index_sensitivities = self.n_state_vars + self.open_state_index + \
            self.n_state_vars * np.array(range(self.n_params))
        sensitivities = solution[:, index_sensitivities]
        o = solution[:, self.open_state_index]

        voltages = self.GetVoltage(times=times)

        current = p[-1] * o * (voltages - self.Erev)


        dIdo = (voltages - self.Erev) * p[-1]
        values = sensitivities * dIdo[:, None]
        values[:, -1] += o * (voltages - self.Erev)

        ret_vals = (current, values)
        return ret_vals

    def GetErrorSensitivities(self, params, data):
        """Solve the model for a given set of parameters

        Returns the state variables and the sensitivities of the error measure
        with respect to each parameter at each timestep

        """
        solution = self.solve_drhs_full(p)
        index_sensitivities = self.open_state_index + \
            self.n_state_vars * np.array(range(self.n_params))

        # Get the open state sensitivities for each parameter
        sensitivites = solution[:, index_sensitivities]
        o = self.solve_rhs(p)[:, self.open_state_index]
        voltages = self.GetVoltage()
        current = params[8] * o * (voltages - self.Erev)

        # Compute the sensitivities of the error measure (chain rule)
        error_sensitivity = np.stack(
            [2 * (current - data) * current * sensitivity for sensitivity in sensitivites.T])

        return current, current_sensitivies
