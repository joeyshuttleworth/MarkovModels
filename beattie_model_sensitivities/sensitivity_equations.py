import numpy as np
import symengine as se
import sympy
from scipy.integrate import odeint

class GetSensitivityEquations(object):
    def __init__(self, settings, p, y, v, A, B, para, times, sine_wave):
        self.par = settings
        self.times = times
        self.sine_wave = sine_wave
        self.A = A
        self.B = B
        rhs = A*y + B
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
        dydp = [[se.symbols('dy%d' % i + 'dp%d' % j) for j in range(self.par.n_params)] \
            for i in range(self.par.n_state_vars)]

        # Append 1st order sensitivities to inputs
        [[inputs.append(dydp[i][j]) for i in range(self.par.n_state_vars)] for j in range(self.par.n_params)]

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

        print('Getting ' + str(self.par.holding_potential) + ' mV steady state initial conditions...')

        # Set the initial conditions of the model and the initial sensitivities
        # by finding the steady state of the model

        # RHS
        # Can be found analytically
        rhs_inf = (-(self.A.inv())*self.B).subs(v,self.voltage(0))
        self.rhs0 = [float(expr.evalf()) for expr in rhs_inf.subs(p, para)]

        # Steady state can be found analytically
        S1_inf = [float(se.diff(rhs_inf[i], p[j]).subs(p, para).evalf()) for i in range(0, self.par.n_state_vars) for j in range(0, self.par.n_params)]

        self.drhs0 = np.concatenate((self.rhs0, S1_inf))

        print('Done')


    def rhs(self, y, t, p, hold_potential=False):
        voltage = self.par.holding_potential if hold_potential else self.voltage(t)
        return self.func_rhs((*y, *p, voltage))

    def jrhs(self, y, t, p, hold_potential=False):
        voltage = self.par.holding_potential if hold_potential else self.voltage(t)
        return self.jfunc_rhs((*y, *p, voltage))

    # Returns the open state
    def solve_rhs(self, p, hold_potential=False):
        return odeint(self.rhs, self.rhs0, self.times, atol=1e-8, rtol=1e-8, Dfun=self.jrhs, args=(p, hold_potential, ))

    def drhs(self, y, t, p, hold_potential=False):
        voltage = self.par.holding_potential if hold_potential else self.voltage(t)
        outputs = self.func_rhs((*y[:self.par.n_state_vars], *p, voltage))
        outputs.extend(self.func_S1((*y[:self.par.n_state_vars], *p, voltage, *y[self.par.n_state_vars:])))
        return outputs

    def jdrhs(self, y, t, p, hold_potential=False):
        voltage = self.par.holding_potential if hold_potential else self.voltage(t)
        return self.jfunc_S1((*y[:self.par.n_state_vars], *p, voltage, *y[self.par.n_state_vars:]))

    # Returns the open state 1st order sensitivities
    def solve_drhs(self, p, hold_potential=False):
        # Chop off RHS
        drhs = odeint(self.drhs, self.drhs0, self.times, atol=1e-8, rtol=1e-8, Dfun=self.jdrhs, args=(p, hold_potential, ))[:, self.par.n_state_vars:]
        # Return only open state sensitivites
        return drhs[:, self.par.open_state::self.par.n_state_vars]

    def solve_drhs_full(self, p, hold_potential=False):
        return odeint(self.drhs, self.drhs0, self.times, atol=1e-8, rtol=1e-8, Dfun=self.jdrhs, args=(p, hold_potential, ))[:, self.par.n_state_vars:]

    def voltage(self, t):
        if self.sine_wave:
            shift = 0.0 # Kylie set to 0.1 ms for real data
            C = [54.0, 26.0, 10.0, 0.007/(2*np.pi), 0.037/(2*np.pi), 0.19/(2*np.pi)]

            if t >= 250+shift and t < 300+shift:
                return -120
            elif t >= 500+shift and t < 1500+shift:
                return 40
            elif t >= 1500+shift and t < 2000+shift:
                return -120
            elif t >= 3000+shift and t < 6500+shift:
                v = -30 + C[0] * (np.sin(2*np.pi*C[3]*(t-2500-shift))) + C[1] * \
                (np.sin(2*np.pi*C[4]*(t-2500-shift))) + C[2] * (np.sin(2*np.pi*C[5]*(t-2500-shift)))
                return(v)
            elif t >= 6500+shift and t < 7000+shift:
                return -120
            else:
                return -80
        else:
            if t >= 1000 and t < 5000:
                return 20
            else:
                return -80

    def SimulateForwardModel(self, p):
        o = self.solve_rhs(p)[:, self.par.open_state]
        return np.array([o[t] * (self.voltage(t) - self.par.Erev) for t, _ in enumerate(self.times)])

    def GetStateVariables(self, p, hold_potential=False, normalise=True):
        states = self.solve_rhs(p, hold_potential)
        if normalise:
            states = states / p[-1] # Normalise to conductance

        state1 = np.array([1.0 - np.sum(row) for row in states])
        state1 = state1.reshape(len(state1), 1)
        states = np.concatenate((state1, states), axis=1)
        return states

    def GetVoltage(self):
        return np.array([self.voltage(t) for t in self.times])

    def NormaliseSensitivities(self, S1n, params):
        # Normalise to parameter value
        for i, param in enumerate(params):
            S1n[:, i] = S1n[:, i] * param
        return S1n

    def SimulateForwardModelSensitivities(self, p):
        S1 = self.solve_drhs(p)
        return np.array([S1[t, :] * (self.voltage(t) - self.par.Erev) for t, _ in enumerate(self.times)])


def GetSymbols(par):
    # Create parameter symbols
    p = se.Matrix([se.symbols('p%d' % j) for j in range(par.n_params)])
    # Create state variable symbols
    y = se.Matrix([se.symbols('y%d' % i) for i in range(par.n_state_vars)])
    # Create voltage symbol
    v = se.symbols('v')
    return p, y, v
