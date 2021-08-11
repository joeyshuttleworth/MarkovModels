import numpy as np
import symengine as se
from scipy.integrate import odeint

class MarkovModel:
    # A class to generate and solve sensitivity equations for the four state, nine parameter Hodgkin-Huxley model of the hERG channel

    # Arguments are provided using the Params class

    # The functions SimulateForwardModel and SimulateForwardModelSensitivities
    # solve the system for each time-step.

    def __init__(self, settings, p, y, v, A, B, para, times, voltage=None):
        # Settings such given in the form of a Param object
        self.par = settings
        # The timesteps we want to output at
        self.times = times
        self.A = A
        self.B = B
        rhs = A * y + B

        if voltage != None:
            self.voltage = voltage

        self.compute_sensitivity_equations_rhs(p, y, v, rhs, para)

    def compute_sensitivity_equations_rhs(self, p, y, v, rhs, para):
        print('Creating RHS function...')

        # Inputs for RHS ODEs
        inputs = [(y[i]) for i in range(self.par.n_state_vars)]
        [inputs.append(p[j]) for j in range(self.par.n_params)]
        inputs.append(v)

        # Create RHS function
        frhs = [rhs[i] for i in range(self.par.n_state_vars)]
        self.func_rhs = se.lambdify(inputs, frhs)

        # Create Jacobian of the RHS function
        jrhs = [se.Matrix(rhs).jacobian(se.Matrix(y))]
        self.jfunc_rhs = se.lambdify(inputs, jrhs)

        print('Creating 1st order sensitivities function...')

        # Create symbols for 1st order sensitivities
        dydp = [[se.symbols('dy%d' % i + 'dp%d' % j) for j in range(self.par.n_params)]
            for i in range(self.par.n_state_vars)]

        # Append 1st order sensitivities to inputs
        for i in range(self.par.n_params):
            for j in range(self.par.n_state_vars):
                inputs.append(dydp[j][i])

        # Initialise 1st order sensitivities
        dS = [[0 for j in range(self.par.n_params)] for i in range(self.par.n_state_vars)]
        S = [[dydp[i][j] for j in range(self.par.n_params)] for i in range(self.par.n_state_vars)]

        # Create 1st order sensitivities function
        fS1, Ss = [], []
        for i in range(self.par.n_state_vars):
            for j in range(self.par.n_params):
                dS[i][j] = se.diff(rhs[i], p[j])
                for l in range(self.par.n_state_vars):
                    dS[i][j] = dS[i][j] + se.diff(rhs[i], y[l]) * S[l][j]

        # Flatten 1st order sensitivities for function
        [[fS1.append(dS[i][j]) for i in range(self.par.n_state_vars)] for j in range(self.par.n_params)]
        [[Ss.append(S[i][j]) for i in range(self.par.n_state_vars)] for j in range(self.par.n_params)]

        self.auxillary_expression = p[self.par.GKr_index]*y[self.par.open_state]*(v - self.par.Erev)

        # dI/do
        self.dIdo = se.diff(self.auxillary_expression, y[self.par.open_state])

        self.func_S1 = se.lambdify(inputs, fS1)

        # Define number of 1st order sensitivities
        self.par.n_state_var_sensitivities = self.par.n_params * self.par.n_state_vars

        # Append 1st order sensitivities to initial conditions
        dydps = np.zeros((self.par.n_state_var_sensitivities))

        # Concatenate RHS and 1st order sensitivities
        Ss = np.concatenate((list(y), Ss))
        fS1 = np.concatenate((frhs, fS1))

        # Create Jacobian of the 1st order sensitivities function
        jS1 = [se.Matrix(fS1).jacobian(se.Matrix(Ss))]
        self.jfunc_S1 = se.lambdify(inputs, jS1)

        print('Getting {}mV steady state initial conditions...'.format(self.par.holding_potential))
        # Set the initial conditions of the model and the initial sensitivities
        # by finding the steady state of the model

        # RHS
        # First check that a steady state exists for this system at this holding voltage
        npmatrix = np.matrix([float(el.evalf()) for el in self.A.subs(v, self.par.holding_potential).subs(p, para)]).reshape(self.A.rows, self.A.cols)
        eigvals, eigvectors = np.linalg.eig(npmatrix)
        assert(np.all(eigvals < 0))

        # Can be found analytically
        rhs_inf = (-(self.A.inv()) * self.B).subs(v, self.par.holding_potential)
        rhs_inf_eval = np.array([float(row) for row in rhs_inf.subs(p, para)])

        current_inf_expr = self.auxillary_expression.subs(y, rhs_inf_eval)
        current_inf      = float(current_inf_expr.subs(v, self.par.holding_potential).subs(p, para).evalf())

        # The limit of the current when voltage is held at the holding potential
        print("Current limit computed as {}".format(current_inf))

        self.rhs0 = rhs_inf_eval

        # Find sensitivity steady states
        S1_inf = np.array([float(se.diff(rhs_inf[i], p[j]).subs(p, para).evalf()) for j in range(0, self.par.n_params) for i in range(0, self.par.n_state_vars)])

        self.drhs0 = np.concatenate((self.rhs0, S1_inf))
        index_sensitivities = self.par.n_state_vars +  self.par.open_state + self.par.n_state_vars*np.array(range(self.par.n_params))
        sens_inf = self.drhs0[index_sensitivities]*(self.par.holding_potential - self.par.Erev)*para[-1]
        sens_inf[-1] += (self.par.holding_potential - self.par.Erev) * rhs_inf_eval[self.par.open_state]
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
            times=self.times

        return odeint(self.rhs, self.rhs0, times, atol=self.par.solver_tolerances[0], rtol=self.par.solver_tolerances[1], Dfun=self.jrhs, \
            args=(p, ))

    def drhs(self, y, t, p):
        """ Evaluate RHS analytically

        """
        outputs = self.func_rhs((*y[:self.par.n_state_vars], *p, self.voltage(t)))
        outputs.extend(self.func_S1((*y[:self.par.n_state_vars], *p, self.voltage(t), *y[self.par.n_state_vars:])))
        return outputs

    def jdrhs(self, y, t, p):
        """  Evaluates the jacobian of the RHS (analytically)

        This allows the system to be solved faster

        """
        return self.jfunc_S1((*y[:self.par.n_state_vars], *p, self.voltage(t), *y[self.par.n_state_vars:]))

    # Returns the open state 1st order sensitivities
    def solve_drhs(self, p, times=None):
        """Solve the RHS of the system and return the open state sensitivities at each
        timestep

        Returns only the sensitivities, not the state variables
        """

        if times is None:
           times = self.times

       # Chop off RHS

        drhs = odeint(self.drhs, self.drhs0, times, atol=self.par.solver_tolerances[0], rtol=self.par.solver_tolerances[1]
        , Dfun=self.jdrhs, args=(p, ))[:, self.par.n_state_vars:]
        # Return only open state sensitivites
        return drhs[:, self.par.open_state::self.par.n_state_vars]

    def solve_drhs_full(self, p, times=None):
        """ Solve the system numerically for every time in self.times

        """
        if times is None:
            times = self.times

        return odeint(self.drhs, self.drhs0, times, atol=self.par.solver_tolerances[0], rtol=self.par.solver_tolerances[1], Dfun=self.jdrhs,
            args=(p, ))

    def voltage(self, t):
        """Returns the voltage in volts of the chosen protocol after t milliseconds has passed

        The voltage protocol used to produce the Sine Wave data in
        https://github.com/mirams/sine-wave. This code is translated from the
        function, static int f in
        https://github.com/mirams/sine-wave/blob/master/Code/MexAslanidi.c.
        """
        # Default to a simple protocol
        if t >= 1000 and t < 5000:
            V = 20
        else:
            V = -80
        # Convert to volts
        return V

    def SimulateForwardModel(self, p, times=None):
        o = self.solve_rhs(p, times)[:, self.par.open_state]
        voltages = self.GetVoltage()
        return p[8] * o * (voltages - self.par.Erev)

    def GetStateVariables(self, p):
        states = self.solve_rhs(p)

        state1 = np.array([1.0 - np.sum(row) for row in states])
        state1 = state1.reshape(len(state1), 1)
        states = np.concatenate((states, state1), axis=1)
        return states

    def GetVoltage(self):
        """
        Returns the voltage at every timepoint

        By default, there is a timestep every millisecond up to self.tmax
        """
        v = np.array([self.voltage(t) for t in self.times])
        return v

    def SimulateForwardModelSensitivities(self, p, times=None):
        """
        Solve the model for a given set of parameters

        Returns the state variables and current sensitivities at every timestep

        Used by pints
        """

        solution = self.solve_drhs_full(p, times)

        # Get the open state sensitivities for each parameter
        index_sensitivities = self.par.n_state_vars + self.par.open_state + self.par.n_state_vars*np.array(range(self.par.n_params))
        sensitivities = solution[:, index_sensitivities]
        # sensitivities = solution[:, self.par.n_state_vars:]
        o = solution[:, self.par.open_state]
        voltages = self.GetVoltage()
        current = p[-1] * o * (voltages - self.par.Erev)
        # values = np.stack(np.array([p[8] * (voltages - self.par.Erev) * sensitivity for sensitivity in sensitivities.T[:-1]]), axis=0)
        dIdo = (voltages - self.par.Erev)*p[-1]
        values = sensitivities * dIdo[:,None]
        values[:,-1] += o*(voltages - self.par.Erev)

        ret_vals = (current, values)
        return ret_vals

    def GetErrorSensitivities(self, params, data):
        """Solve the model for a given set of parameters

        Returns the state variables and the sensitivities of the error measure
        with respect to each parameter at each timestep

        """
        solution = self.solve_drhs_full(p)
        index_sensitivities = self.par.open_state + self.par.n_state_vars*np.array(range(self.par.n_params))

        # Get the open state sensitivities for each parameter
        sensitivites = solution[:, index_sensitivities]
        o = self.solve_rhs(p)[:, self.par.open_state]
        voltages = self.GetVoltage()
        current = params[8] * o * (voltages - self.par.Erev)

        # Compute the sensitivities of the error measure (chain rule)
        error_sensitivity = np.stack([2*(current - data) * current * sensitivity for sensitivity in sensitivites.T])

        return current, current_sensitivies

