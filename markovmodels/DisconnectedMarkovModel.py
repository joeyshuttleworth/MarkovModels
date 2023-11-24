import numba as nb
import numpy as np
import sympy as sp
from numba import cfunc, njit
from numbalsoda import lsoda, lsoda_sig

from markovmodels.MarkovModel import MarkovModel


class DisconnectedMarkovModel(MarkovModel):

    def __init__(self, symbols, A, B, Qs, As, Bs, ys, connected_components,
                 rates_dict, times, parameter_labels,
                 auxiliary_expression, analytic_solution_funcs=None,
                 *args, **kwargs):

        self.analytic_solution_funcs = analytic_solution_funcs

        self.As = As
        self.Bs = Bs
        self.Qs = Qs
        self.ys = ys

        self.auxiliary_expression = auxiliary_expression

        self.connected_components = connected_components
        self.n_states = sum([Q.shape[0] for Q in Qs])
        self.parameter_labels = parameter_labels

        self.y = [var for y in ys for var in y]

        super().__init__(symbols, A, B, rates_dict, times, *args, **kwargs)
        self.n_state_vars = sum([len(y) for y in self.ys])
        self.auxiliary_function = self.define_auxiliary_function()

    def compute_steady_state_expressions(self):

        self.rhs_inf_expr_rates = []
        self.rhs_inf_expr = []
        self.rhs_infs = []
        self.current_inf_expr = []
        self.current_inf = []

        for A, B, Q in zip(self.As, self.Bs, self.Qs):
            rhs_inf_expr_rates = -A.LUsolve(B)
            self.rhs_inf_expr_rates.append(rhs_inf_expr_rates)

            rhs_inf_expr = rhs_inf_expr_rates.subs(self.rates_dict)
            self.rhs_inf_expr.append(rhs_inf_expr)
            self.rhs_infs.append(njit(sp.lambdify((self.p, self.v), rhs_inf_expr)))

        self.current_inf_expr = self.auxiliary_expression.subs(self.y, self.rhs_inf_expr)
        self.current_inf = nb.njit(sp.lambdify((self.y, self.p), self.current_inf_expr))

        # Generate function for entire RHS (all components)
        rhs_inf_expr_rates = sp.Matrix.vstack(*[expr for expr in self.rhs_inf_expr])
        rhs_inf_expr = rhs_inf_expr_rates.subs(self.rates_dict)

        self.rhs_inf = njit(sp.lambdify((self.p, self.v), rhs_inf_expr))

    def get_analytic_solution_funcs(self, cond_threshold=None):
        rates_func = self.get_rates_func()

        if self.analytic_solution_funcs is not None:
            return self.analytic_solution_funcs

        ret_funcs = []
        for A, B, comp in zip(self.As, self.Bs, self.connected_components):

            times = self.times
            voltage = self.voltage(0)

            p = self.get_default_parameters()

            A_func = njit(sp.lambdify((self.rates_dict.keys(),), A), fastmath=True)
            B_func = njit(sp.lambdify((self.rates_dict.keys(),), B), fastmath=True)

            ret_funcs.append(self.get_analytic_solution_func(A_func, B_func, A,
                                                             B, p, times,
                                                             voltage,
                                                             rates_func,
                                                             cond_threshold=cond_threshold))

        self.analytic_solution_funcs = ret_funcs
        return ret_funcs

    def get_analytic_solution_func(self, A_func, B_func, A, B, p, times,
                                   voltage, rates_func, cond_threshold=None):

        if cond_threshold is None:
            cond_threshold = 1e5

        if A.shape[0] == 1:
            # Scalar case
            def analytic_solution_func_scalar(times, voltage, p, y0):
                rates = rates_func(p, voltage).flatten()
                y0 = y0[0]
                a = A_func(rates)[0, 0]

                if a < 1 / cond_threshold:
                    return np.full((times.shape[0], 1), np.nan)

                b = B_func(rates)[0, 0]
                sol = np.expand_dims((y0 + b/a) * np.exp(a * times) - b/a, -1)
                return sol

            return analytic_solution_func_scalar

        else:
            def analytic_solution_func(times, voltage, p, y0):
                rates = rates_func(p, voltage).flatten()
                _A = A_func(rates)
                _B = B_func(rates)

                try:
                    cond_A = np.linalg.norm(_A, 2) * np.linalg.norm(np.linalg.inv(_A), 2)
                except Exception:
                    return np.full((times.shape[0], y0.shape[0]), np.nan)

                if cond_A > cond_threshold:
                    print("WARNING: cond_A = ", cond_A, " > ", cond_threshold)
                    print("matrix is poorly conditioned", cond_A, cond_threshold)
                    return np.full((times.shape[0], y0.shape[0]), np.nan)

                D, P = np.linalg.eig(_A)

                # Compute condition number doi:10.1137/S00361445024180
                try:
                    cond_P = np.linalg.norm(P, 2) * np.linalg.norm(np.linalg.inv(P), 2)
                except Exception:
                    return np.full((times.shape[0], y0.shape[0]), np.nan)

                if cond_P > cond_threshold:
                    print("WARNING: cond_P = ", cond_P, " > ", cond_threshold)
                    print("matrix is almost defective", cond_P, cond_threshold)
                    return np.full((times.shape[0], y0.shape[0]), np.nan)

                X2 = -np.linalg.solve(_A, _B)

                K = np.diag(np.linalg.solve(P, (y0 - X2).flatten()))

                solution = (P @ K @  np.exp(np.outer(D, times))).T + X2.T

                return solution

            return analytic_solution_func

    def make_hybrid_solver_states(self, protocol_description=None,
                                  njitted=False, strict=True,
                                  cond_threshold=None, atol=None, rtol=None,
                                  hybrid=True, crhs=None):
        return self.make_solver_states(protocol_description, njitted, strict,
                                       hybrid=hybrid, cond_threshold=None,
                                       atol=atol, rtol=rtol, crhs=crhs)

    def make_solver_states(self, protocol_description=None, njitted=False,
                           strict=True, hybrid=True, solver_type='lsoda',
                           atol=None, rtol=None, cond_threshold=None, crhs=None):
        if protocol_description is None:
            if self.protocol_description is None:
                raise Exception("No protocol description has been provided")
            else:
                protocol_description = self.protocol_description

        voltage = self.voltage

        if atol is None:
            atol = self.solver_tolerances[0]

        if rtol is None:
            rtol = self.solver_tolerances[1]

        times = self.times

        p = self.get_default_parameters()
        eps = np.finfo(float).eps

        if solver_type != 'lsoda':
            raise NotImplementedError()

        def hybrid_forward_solve_component(rhs_inf, analytic_solver, crhs_ptr, p=p,
                                           times=times, atol=atol, rtol=rtol,
                                           strict=strict, hybrid=hybrid, crhs=None,
                                           protocol_description=protocol_description):
            y0 = rhs_inf(p, voltage(.0)).flatten()
            no_states = y0.shape[0]
            solution = np.full((len(times), no_states), np.nan)
            solution[0, :] = y0

            # Flatten and pad out protocol description to 64 steps
            flat_desc = protocol_description.flatten().copy()
            if protocol_description.shape[0] < 64 * 4:
                flat_desc = \
                    np.append(protocol_description,
                              np.full(64 * 4 - flat_desc.shape[0],
                                      np.inf))

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
                    try:
                        # step_sol[:, :] = analytic_solver(step_times - tstart,
                        #                                  vstart, p, y0)
                        analytic_success = np.all(np.isfinite(step_sol))

                    except Exception:
                        analytic_success = False

                if not analytic_success:
                    step_times[0] = tstart
                    step_times[-1] = tend

                    if step_times[1] - tstart < 2 * eps * np.abs(step_times[1]):
                        start_int = 1
                        step_sol[0] = y0
                    else:
                        start_int = 0

                    t_offset = tstart
                    data = np.concatenate((p, np.array([t_offset]),
                                           flat_desc))

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

                if end_int == -1:
                    step_sol[-1, :] = step_sol[-2, :]

                if iend == len(times):
                    solution[istart:, ] = step_sol[1:-1, ]
                    break

                else:
                    y0 = step_sol[-1, :]
                    solution[istart:iend, ] = step_sol[1:-1, ]

            return solution

        if crhs is None:
            rhs_cfunc_ptrs = []
            for A, B, comp in zip(self.As, self.Bs, self.connected_components):
                cfunc_rhs = self.make_cfunc_rhs(A, B, comp)
                rhs_cfunc_ptrs.append(cfunc_rhs.address)

        else:
            rhs_cfunc_ptrs = [cfunc.address for cfunc in crhs]

        rhs_cfunc_ptrs = tuple(rhs_cfunc_ptrs)

        steady_state_funcs = tuple(self.rhs_infs)
        analytic_solvers = tuple([njit(func) for func in self.get_analytic_solution_funcs()])

        # Cannot loop over analytic solvers for some reason. All of the 30
        # models examples have 1 or 2 components, though
        analytic_solver1 = analytic_solvers[0]
        steady_state_func1 = steady_state_funcs[0]
        rhs_cfunc1 = rhs_cfunc_ptrs[0]
        n_state_vars = self.n_state_vars

        if len(self.connected_components) == 2:
            analytic_solver2 = analytic_solvers[1]
            steady_state_func2 = steady_state_funcs[1]
            rhs_cfunc2 = rhs_cfunc_ptrs[1]

            def hybrid_forward_solver(p=p, times=times, atol=atol, rtol=rtol,
                                      strict=strict, hybrid=hybrid,
                                      protocol_description=protocol_description):
                solution = np.full((times.shape[0], n_state_vars), np.nan)
                state_counter = 0
                component_solution = \
                    hybrid_forward_solve_component(steady_state_func1,
                                                   analytic_solver1,
                                                   rhs_cfunc1, p=p,
                                                   times=times,
                                                   atol=atol, rtol=rtol,
                                                   strict=strict,
                                                   hybrid=hybrid,
                                                   protocol_description=protocol_description)

                solution[:, :component_solution.shape[1]] = component_solution
                state_counter += component_solution.shape[1]

                component_solution = \
                    hybrid_forward_solve_component(steady_state_func2,
                                                   analytic_solver2,
                                                   rhs_cfunc2, p=p,
                                                   times=times,
                                                   atol=atol, rtol=rtol,
                                                   strict=strict,
                                                   hybrid=hybrid,
                                                   protocol_description=protocol_description
                                                   )

                solution[:, state_counter:] = component_solution

                return solution

        elif len(self.connected_components) == 1:
            def hybrid_forward_solver(p=p, times=times, atol=atol, rtol=rtol,
                                      strict=strict, hybrid=hybrid,
                                      protocol_description=protocol_description):
                solution = np.full((times.shape[0], n_state_vars), np.nan)
                component_solution = \
                    hybrid_forward_solve_component(steady_state_func1,
                                                   analytic_solver1,
                                                   rhs_cfunc1, p=p,
                                                   times=times,
                                                   atol=atol, rtol=rtol,
                                                   strict=strict,
                                                   protocol_description=protocol_description,
                                                   hybrid=hybrid)

                solution[:, :] = component_solution
                return solution

        else:
            raise NotImplementedError('Only models with a maximum of 2 connected components are implemented')

        if njitted:
            hybrid_forward_solve_component = njit(hybrid_forward_solve_component)
        return njit(hybrid_forward_solver) if njitted else hybrid_forward_solver

    def make_cfunc_rhs(self, A, B, comp):
        auxiliary_states = [str(s) for s in self.auxiliary_expression.free_symbols
                            if len(str(s)) > 6]
        auxiliary_states = [s for s in auxiliary_states if s[:6] == 'state_']
        eliminated_state = [state for state in comp if state not in auxiliary_states][0]
        y = sp.Matrix([[f"state_{state}"] for state in comp if state != eliminated_state])
        rhs_expr = (A @ y + B).subs(self.rates_dict)
        rhs_func = njit(sp.lambdify((y, self.p, self.v), rhs_expr,
                                    cse=True))
        ny = A.shape[0]
        n_p = len(self.get_default_parameters())
        voltage = self.voltage

        n_max_steps = 64

        @cfunc(lsoda_sig)
        def cfunc_rhs(t, y, dy, data):
            y = nb.carray(y, ny)
            dy = nb.carray(dy, ny)
            data = nb.carray(data, n_p + 1 + n_max_steps * 4)
            p = data[:-1 - n_max_steps*4]
            t_offset = data[-1 - n_max_steps*4]
            desc = data[-n_max_steps * 4:].reshape((-1, 4))

            v = voltage(t, offset=t_offset, protocol_description=desc)

            res = rhs_func(y, p, v).flatten()
            dy[:] = res

        return cfunc_rhs

    def get_cfunc_rhs(self):
        return [self.make_cfunc_rhs(A, B, comp) for A, B, comp in zip(self.As, self.Bs, self.connected_components)]

    def get_rhs_func(self, njitted=True):
        y = [var for y in self.ys for var in y]
        rhs_exprs = [A @ sp.Matrix(y) + B for A, B, y in zip(self.As, self.Bs, self.ys)]
        rhs_expr = sp.Matrix.vstack(*rhs_exprs).subs(self.rates_dict)
        rhs_func = sp.lambdify((y, self.p, self.v), rhs_expr, cse=True)

        return njit(rhs_func) if njitted else rhs_func

    def make_ida_residual_func():
        raise NotImplementedError()

    def define_auxiliary_function(self):
        y = [var for y in self.ys for var in y]
        aux_expr = self.auxiliary_expression.subs({'E_Kr': self.E_rev})
        return sp.lambdify((y, self.p, self.v), aux_expr)

    def get_no_state_vars(self):
        return sum([len(y) for y in self.ys])

    def get_state_labels(self):
        return [lab for c in self.connected_components for lab in c[:-1]]
