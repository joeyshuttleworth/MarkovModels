import numpy as np
import symengine as se
from scipy.integrate import odeint

class GetSensitivityEquations(object):
    def __init__(self, global_stuff, p, y, v, rhs, ICs, sine_wave):
        self.par = global_stuff
        self.times = np.linspace(0, self.par.tmax, self.par.tmax + 1)
        self.compute_sensitivity_equations_rhs(p, y, v, rhs, ICs)
        self.sine_wave = sine_wave

    def compute_sensitivity_equations_rhs(self, p, y, v, rhs, ICs):
        print('Creating RHS function...')

        # Inputs for RHS ODEs
        inputs = [(y[i]) for i in range(self.par.n_state_vars)]
        [inputs.append(p[j]) for j in range(self.par.n_params)]
        inputs.append(v)

        self.rhs0 = ICs

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
        dydxs = np.zeros((self.par.n_state_var_sensitivities))
        self.drhs0 = np.concatenate((ICs, dydxs))

        # Concatenate RHS and 1st order sensitivities
        Ss = np.concatenate((y, Ss))
        fS1 = np.concatenate((frhs, fS1))

        # Create Jacobian of the 1st order sensitivities function
        jS1 = [se.Matrix(fS1).jacobian(se.Matrix(Ss))]
        self.jfunc_S1 = se.lambdify(inputs, jS1)

        print('Finished creating sensitivity functions.')

    def rhs(self, y, t, p):
        return self.func_rhs((*y, *p, self.voltage(t)))

    def jrhs(self, y, t, p):
        return self.jfunc_rhs((*y, *p, self.voltage(t)))

    # Returns the open state
    def solve_rhs(self, p):
        return odeint(self.rhs, self.rhs0, self.times, atol=1e-8, rtol=1e-8, Dfun=self.jrhs, args=(p, ))[:, self.par.open_state]

    def drhs(self, y, t, p):
        outputs = self.func_rhs((*y[:self.par.n_state_vars], *p, self.voltage(t)))
        outputs.extend(self.func_S1((*y[:self.par.n_state_vars], *p, self.voltage(t), *y[self.par.n_state_vars:])))

        return outputs

    def jdrhs(self, y, t, p):
        return self.jfunc_S1((*y[:self.par.n_state_vars], *p, self.voltage(t), *y[self.par.n_state_vars:]))

    # Returns the open state 1st order sensitivities
    def solve_drhs(self, p):
        # Chop off RHS
        drhs = odeint(self.drhs, self.drhs0, self.times, atol=1e-8, rtol=1e-8, Dfun=self.jdrhs, args=(p, ))[:, self.par.n_state_vars:]
        # Return only open state sensitivites
        return drhs[:, self.par.open_state::self.par.n_state_vars]

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
        o = self.solve_rhs(p)
        return np.array([o[t] * (self.voltage(t) - self.par.Erev) for t, _ in enumerate(self.times)])

    def GetVoltage(self):
        return [self.voltage(t) for t, _ in enumerate(self.times)]

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
    p = [se.symbols('p%d' % j) for j in range(par.n_params)]
    # Create state variable symbols
    y = [se.symbols('y%d' % i) for i in range(par.n_state_vars)]
    # Create voltage symbol
    v = se.symbols('v')

    return p, y, v

