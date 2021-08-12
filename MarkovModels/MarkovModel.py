import numpy as np
import symengine as se
from scipy.integrate import odeint


class MarkovModel:
    """
    A class containing describing a Markov Model of an ion channel

    Arguments are provided using the Params class

    """

    solver_tolerances = (1e-5, 1e-7)

    def get_default_parameters():
        raise NotImplementedError

    def __init__(self, p, y, v, A, B, times, voltage=None):

        # The timesteps we want to output at
        self.times = times
        self.A = A
        self.B = B
        rhs = A * y + B

        if voltage is not None:
            self.voltage = voltage

        para = self.get_default_parameters()
        self.compute_sensitivity_equations_rhs(p, y, v, rhs, para)

    def compute_sensitivity_equations_rhs(self, p, y, v, rhs, para):
        print('Creating RHS function...')

        # Inputs for RHS ODEs
        inputs = list(y) + list(p) +[v]

        # Create RHS function
        frhs = [rhs[i] for i in range(self.n_state_vars)]
        print(inputs)
        self.func_rhs = se.lambdify(inputs, frhs)

        # Create Jacobian of the RHS function
        jrhs = [se.Matrix(rhs).jacobian(se.Matrix(y))]
        self.jfunc_rhs = se.lambdify(inputs, jrhs)

        print('Creating 1st order sensitivities function...')

        # Create symbols for 1st order sensitivities
        dydp = [
            [
                se.symbols(
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
                dS[i][j] = se.diff(rhs[i], p[j])
                for l in range(self.n_state_vars):
                    dS[i][j] = dS[i][j] + se.diff(rhs[i], y[l]) * S[l][j]

        # Flatten 1st order sensitivities for function
        [[fS1.append(dS[i][j]) for i in range(self.n_state_vars)]
         for j in range(self.n_params)]
        [[Ss.append(S[i][j]) for i in range(self.n_state_vars)]
         for j in range(self.n_params)]

        self.auxillary_expression = p[self.GKr_index] * \
            y[self.open_state_index] * (v - self.Erev)

        # dI/do
        self.dIdo = se.diff(self.auxillary_expression, y[self.open_state_index])

        self.func_S1 = se.lambdify(inputs, fS1)

        # Define number of 1st order sensitivities
        self.n_state_var_sensitivities = self.n_params * self.n_state_vars

        # Concatenate RHS and 1st order sensitivities
        Ss = np.concatenate((list(y), Ss))
        fS1 = np.concatenate((frhs, fS1))

        # Create Jacobian of the 1st order sensitivities function
        jS1 = [se.Matrix(fS1).jacobian(se.Matrix(Ss))]
        self.jfunc_S1 = se.lambdify(inputs, jS1)

        print('Getting {}mV steady state initial conditions...'.format(
            self.holding_potential))
        # Set the initial conditions of the model and the initial sensitivities
        # by finding the steady state of the model

        # RHS
        # First check that a steady state exists for this system at this
        # holding voltage
        # dx/dt = Ax+B
        A_matrix = np.array([float(el.evalf()) for el in self.A.subs(
            v, self.holding_potential).subs(p, para)]).reshape(self.A.rows, self.A.cols)
        eigvals = np.linalg.eig(A_matrix)[0]
        assert(np.all(eigvals < 0))

        # Can be found analytically
        rhs_inf = (-(self.A.inv()) * self.B).subs(v,
                                                  self.holding_potential)
        rhs_inf_eval = np.array([float(row) for row in rhs_inf.subs(p, para)])

        current_inf_expr = self.auxillary_expression.subs(y, rhs_inf_eval)
        current_inf = float(current_inf_expr.subs(
            v, self.holding_potential).subs(p, para).evalf())

        # The limit of the current when voltage is held at the holding
        # potential
        print("Current limit computed as {}".format(current_inf))

        self.rhs0 = rhs_inf_eval

        # Find sensitivity steady states
        S1_inf = np.array(
            [
                float(
                    se.diff(
                        rhs_inf[i],
                        p[j]).subs(
                        p,
                        para).evalf()) for j in range(
                    0,
                    self.n_params) for i in range(
                            0,
                    self.n_state_vars)])

        self.drhs0 = np.concatenate((self.rhs0, S1_inf))
        index_sensitivities = self.n_state_vars + self.open_state_index + \
            self.n_state_vars * np.array(range(self.n_params))
        sens_inf = self.drhs0[index_sensitivities] * \
            (self.holding_potential - self.Erev) * para[-1]
        sens_inf[-1] += (self.holding_potential -
                         self.Erev) * rhs_inf_eval[self.open_state_index]
        print("sens_inf calculated as {}".format(sens_inf))

        print('Done')

    def rhs(self, y, t, p):
        """ Evaluates the RHS of the model (including sensitivities)

        """
        return self.func_rhs((*y, *p, self.voltage(t)))

    def jrhs(self, y, t, p):
        """ Evaluates the jacobian of the RHS

            Having this function can speed up solving the system
        """
        return self.jfunc_rhs((*y, *p, self.voltage(t)))

    # Returns the open state
    def solve_rhs(self, p, times=None):
        """ Solve the RHS of the system and return the open state probability at each timestep
        """

        if times is None:
            times = self.times

        return odeint(
            self.rhs,
            self.rhs0,
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
            (*y[:self.n_state_vars], *p, self.voltage(t)))
        outputs.extend(self.func_S1((*
                                     y[:self.n_state_vars], *
                                     p, self.voltage(t), *
                                     y[self.n_state_vars:])))
        return outputs

    def jdrhs(self, y, t, p):
        """  Evaluates the jacobian of the RHS (analytically)

        This allows the system to be solved faster

        """
        return self.jfunc_S1((*
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

        # Chop off RHS

        drhs = odeint(self.drhs,
                      self.drhs0,
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

        return odeint(
            self.drhs,
            self.drhs0,
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
