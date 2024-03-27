import numba as nb
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from numba import cfunc, njit
from numbalsoda import lsoda, lsoda_sig

from markovmodels.utilities import calculate_reversal_potential

_lsoda_n_max_steps = 10_000


class ODEModel:
    """
    A class describing a Markov Model of an ODE model of a patch-clamp experiment

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

    def __init__(self, symbols, times=None, voltage=None,
                 tolerances=(1e-8, 1e-8), protocol_description=None,
                 name=None, E_rev=None, default_parameters=None,
                 parameter_labels=None,  transformations=None,
                 state_labels: str = None):

        self.initial_condition = None

        self.name = name

        if state_labels:
            self.state_labels = state_labels

        if default_parameters is not None:
            self.default_parameters = default_parameters

        if parameter_labels:
            self.parameter_labels = parameter_labels

        if E_rev is None:
            self.E_rev = calculate_reversal_potential()
        else:
            self.E_rev = E_rev

        self.transformations = transformations

        self.model_name = name

        self.protocol_description = protocol_description

        if self.protocol_description is None:
            protocol_description = np.array([0., np.inf, -80, -80])

        self.window_locs = None

        self.y = symbols['y']
        self.p = symbols['p']
        self.v = symbols['v']

        # (atol, rtol)
        self.solver_tolerances = tuple(tolerances)

        self.symbols = symbols
        # The timesteps we want to output at

        self.times = times

        if voltage is not None:
            self.voltage = voltage
        else:
            @njit
            def voltage(t, offset=0.0, protocol_description=None):
                return np.nan
            self.voltage = voltage

        if voltage is not None:
            self.holding_potential = voltage(0)

        inputs = [self.y, self.p, self.v]
        # Create Jacobian of the RHS function
        jrhs = sp.Matrix(self.rhs_expr).jacobian(self.y)
        self.jfunc_rhs = sp.lambdify(inputs, jrhs)

        self.func_rhs = njit(sp.lambdify(inputs, self.rhs_expr))

        # Set the initial conditions of the model and the initial sensitivities
        # by finding the steady state of the model

        self.compute_steady_state_expressions()
        self.auxiliary_function = njit(self.define_auxiliary_function())

    def func_rhs(self):
        raise NotImplementedError()

    def define_auxiliary_function(self, **kwargs):
        return sp.lambdify((self.y, self.p, self.v), self.auxiliary_expression)

    def compute_steady_state_expressions(self):
        raise NotImplementedError()

    def setup_sensitivities(self):
        raise NotImplementedError()

    def rhs(self, t, y, p):
        """ Evaluates the RHS of the model (including sensitivities)

        """
        return self.func_rhs(y, p, self.voltage(t))

    def get_no_state_vars(self):
        return len(self.y)

    def jrhs(self, t, y, p):
        """ Evaluates the jacobian of the RHS

            Having this function can speed up solving the system
        """
        return self.jfunc_rhs(*(*y, *p, self.voltage(t)))

    def make_Q_func(self, njitted=True):
        rates = tuple(self.rates_dict.keys())
        Q_func = sp.lambdify((rates,), self.Q, modules='numpy', cse=True)
        return njit(Q_func) if njitted else Q_func

    def get_rates_func(self, njitted=True):
        inputs = (self.p, self.v)
        rates_expr = sp.Matrix(list(self.rates_dict.values()))

        rates_func = sp.lambdify(inputs, rates_expr)

        return njit(rates_func) if njitted else rates_func

    def make_hybrid_solver_states(self, protocol_description=None,
                                  njitted=False, analytic_solver=None,
                                  strict=True, cond_threshold=None, atol=None,
                                  rtol=None, hybrid=True, crhs=None):

        if protocol_description is None:
            if self.protocol_description is None:
                raise Exception("No protocol description has been provided")
            else:
                protocol_description = self.protocol_description.copy()

        if crhs is None:
            crhs = self.get_cfunc_rhs()

        crhs_ptr = crhs.address

        no_states = self.get_no_state_vars()

        if not analytic_solver and hybrid:
            analytic_solver = self.get_analytic_solution_func(njitted=njitted,
                                                              cond_threshold=cond_threshold)
        else:
            # Define dummy function so numba doesn't fail
            @njit
            def analytic_solver(times=None, voltage=None, p=None, y0=None):
                return np.array([np.nan]), False

        rhs_inf = self.rhs_inf
        voltage = self.voltage

        if atol is None:
            atol = self.solver_tolerances[0]
        if rtol is None:
            rtol = self.solver_tolerances[1]

        times = self.times

        p = self.get_default_parameters()
        eps = np.finfo(float).eps

        def hybrid_forward_solve(p=p, times=times, atol=atol, rtol=rtol,
                                 strict=strict, hybrid=hybrid,
                                 protocol_description=protocol_description):
            y0 = rhs_inf(p, voltage(.0)).flatten()
            solution = np.full((len(times), no_states), np.nan)
            solution[0, :] = y0

            # pad protocol description to fill up 64 steps
            flat_desc = protocol_description.flatten().copy()
            if flat_desc.shape[0] < 64 * 4:
                flat_desc = \
                    np.concatenate((flat_desc,
                                    np.full(64 * 4 - flat_desc.shape[0],
                                            np.inf)))

            start_times = protocol_description[:, 0]
            for i in range(len(protocol_description) - 1):

                start_int = 0
                end_int = 0

                tstart = protocol_description[i, 0]
                tend = protocol_description[i + 1, 0]

                if i == len(start_times) - 1:
                    tend = times[-1] + 1

                istart = np.argmax(times > tstart)
                iend = np.argmax(times > tend)

                if iend == 0:
                    iend = len(times)

                vstart = protocol_description[i, 2]
                vend = protocol_description[i, 3]

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

                    data = np.concatenate((p, np.array([t_offset]),
                                           flat_desc))

                    if tend - step_times[-1] < 2 * eps * np.abs(tend):
                        end_int = -1
                        step_sol[start_int: end_int], _ = lsoda(crhs_ptr, y0,
                                                                step_times[start_int:end_int] - step_times[0],
                                                                data=data, rtol=rtol,
                                                                atol=atol,
                                                                exit_on_warning=strict,
                                                                mxstep=_lsoda_n_max_steps)
                    else:
                        end_int = 0
                        step_sol[start_int:], _ = lsoda(crhs_ptr, y0,
                                                        step_times[start_int:] - step_times[0],
                                                        data=data, rtol=rtol,
                                                        atol=atol, exit_on_warning=strict,
                                                        mxstep=_lsoda_n_max_steps)

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
        rhs = self.func_rhs
        voltage = self.voltage

        ny = self.get_no_state_vars()
        n_p = len(self.get_default_parameters())

        # Maximum steps in protocol
        n_max_steps = 64

        @cfunc(lsoda_sig)
        def crhs(t, y, dy, data):
            y = nb.carray(y, ny)
            dy = nb.carray(dy, ny)
            data = nb.carray(data, int(n_p + 1 + n_max_steps * 4))
            p = data[:n_p]
            t_offset = data[n_p]

            desc = data[n_p + 1:].reshape(-1, 4)
            res = rhs(y, p, voltage(t, offset=t_offset,
                                    protocol_description=desc)).flatten()

            dy[:] = res

        return crhs

    def make_hybrid_solver_current(self, protocol_description=None,
                                   njitted=True, strict=True,
                                   cond_threshold=None, atol=None, rtol=None,
                                   hybrid=True, crhs=None, af_kws={}, **kwargs):

        if protocol_description is None:
            protocol_description = self.protocol_description

        hybrid_solver =\
            self.make_hybrid_solver_states(protocol_description=protocol_description,
                                           njitted=njitted, strict=strict,
                                           cond_threshold=cond_threshold,
                                           atol=atol, rtol=rtol, hybrid=hybrid,
                                           crhs=crhs)

        auxiliary_function = njit(self.define_auxiliary_function(**af_kws))
        times = self.times
        atol, rtol = self.solver_tolerances
        voltage_func = self.voltage

        params = self.get_default_parameters()

        def hybrid_forward_solve(p=params, times=times, atol=atol, rtol=rtol,
                                 hybrid=hybrid,
                                 protocol_description=protocol_description):
            voltages = np.empty(len(times))
            for i in range(len(times)):
                voltages[i] = voltage_func(times[i],
                                           protocol_description=protocol_description)

            states = hybrid_solver(p, times=times, hybrid=hybrid, atol=atol,
                                   rtol=rtol,
                                   protocol_description=protocol_description)
            return (auxiliary_function(states.T, p, voltages)).flatten()

        return njit(hybrid_forward_solve) if njitted else hybrid_forward_solve

    def make_forward_solver_current(self, voltages=None, njitted=True,
                                    protocol_description=None,
                                    solver_type='lsoda', atol=None, rtol=None,
                                    **kws):

        if protocol_description is None:
            protocol_description = self.protocol_description

        if protocol_description is None:
            protocol_description = np.array([[.0, .0, .0, .0]])

        solver_states = self.make_hybrid_solver_states(njitted=njitted,
                                                       protocol_description=protocol_description,
                                                       atol=atol, rtol=rtol,
                                                       hybrid=False,
                                                       **kws)
        return self.make_solver_current(solver_states, voltages=voltages,
                                        atol=atol, rtol=rtol, njitted=njitted,
                                        protocol_description=protocol_description)

    def make_solver_current(self, solver_states, voltages=None, atol=None,
                            rtol=None, njitted=False, protocol_description=None):
        if atol is None:
            atol = self.solver_tolerances[0]

        if rtol is None:
            rtol = self.solver_tolerances[1]

        if voltages is None:
            voltages = self.GetVoltage()

        times = self.times
        default_parameters = self.get_default_parameters()
        auxiliary_function = self.auxiliary_function

        def forward_solver(p=default_parameters, times=times,
                           voltages=voltages, atol=atol, rtol=rtol,
                           protocol_description=protocol_description):
            states = solver_states(p, times, atol, rtol,
                                   protocol_description=protocol_description)

            return (auxiliary_function(states.T, p, voltages)).flatten()

        if njitted:
            forward_solver = njit(forward_solver)
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

    def get_rhs_func(self):
        return self.func_rhs

    def count_rhs_evaluations(self, p, times=None):

        if times is None:
            times = self.times

        y0 = self.rhs_inf(p, self.voltage(0)).flatten()
        rhs_func = self.rhs

        class rhs_counter():
            evals = 0

            def func(self, t, y, *args):
                self.evals += 1
                return rhs_func(t, y, p)

        rhs_count = rhs_counter()
        # Chop off RHS
        solve_ivp(lambda t, y, *args: rhs_count.func(t, y, *args),
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

    def voltage(self, t):
        raise NotImplementedError

    def SimulateForwardModel(self, p=None, times=None, **kws):
        if p is None:
            p = self.get_default_parameters()
        p = np.array(p)

        if times is None:
            times = self.times
        return self.make_forward_solver_current(njitted=False, **kws)(p, times)

    def GetVoltage(self, times=None):
        """
        Returns the voltage at every timepoint

        By default, there is a timestep every millisecond up to self.tmax
        """
        if times is None:
            times = self.times

        v = np.array([self.voltage(t) for t in times])
        return v

    def get_no_parameters(self):
        return len(self.get_default_parameters())

    def set_tolerances(self, abs_tol, rel_tol):
        self.solver_tolerances = (abs_tol, rel_tol)
