from .MarkovModel import MarkovModel

import numpy as np
import sympy as sp
import numba as nb

from numba import njit, cfunc

from numbalsoda import lsoda, lsoda_sig


class DisconnectedMarkovModel(MarkovModel):

    def __init__(self, symbols, A, B, Qs, As, Bs, ys, connected_components,
                 rates_dict, times, parameter_labels,
                 auxiliary_expression, *args, **kwargs):

        self.auxiliary_expression = auxiliary_expression

        self.connected_components = connected_components

        self.n_states = sum([Q.shape[0] for Q in Qs])

        self.parameter_labels = parameter_labels

        self.As = As
        self.Bs = Bs
        self.Qs = Qs
        self.ys = ys

        super().__init__(symbols, A, B, rates_dict, times, *args, **kwargs)

    def compute_steady_state_expressions(self):

        self.rhs_inf_expr_rates = []
        self.rhs_inf_expr = []
        self.rhs_inf = []
        self.current_inf_expr = []
        self.current_inf = []

        for A, B, Q in zip(self.As, self.Bs, self.Qs):
            rhs_inf_expr_rates = -A.LUsolve(B)
            self.rhs_inf_expr_rates.append(rhs_inf_expr_rates)

            rhs_inf_expr = rhs_inf_expr_rates.subs(self.rates_dict)
            self.rhs_inf_expr.append(rhs_inf_expr)
            self.rhs_inf.append(sp.lambdify((self.p, self.v), rhs_inf_expr))

        self.current_inf_expr = self.auxiliary_expression.subs(self.y, self.rhs_inf_expr)
        self.current_inf = nb.njit(sp.lambdify((self.y, self.p), self.current_inf_expr))

    def get_analytic_solution_funcs(self):
        rates_func = self.get_rates_func()

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
                                                             rates_func))
        return ret_funcs

    def get_analytic_solution_func(self, A_func, B_func, A, B, p, times,
                                   voltage, rates_func):
        if A.shape[0] == 1:
            # Scalar case
            def analytic_solution_func_scalar(times, voltage, p, y0):
                rates = rates_func(p, voltage).flatten()
                y0 = y0[0]
                a = A_func(rates)[0, 0]
                b = B_func(rates)[0, 0]
                sol = np.expand_dims((y0 + b/a) * np.exp(a * times) - b/a, -1)
                return sol

            return analytic_solution_func_scalar

        else:
            X_inhom = A.LUsolve(-B)
            X_inhom_func = njit(sp.lambdify((self.rates_dict.keys(),), X_inhom), fastmath=True)

            def analytic_solution_func(times, voltage, p, y0):
                rates = rates_func(p, voltage).flatten()
                A = A_func(rates)
                D, P = np.linalg.eig(A)

                # Compute condition number doi:10.1137/S00361445024180
                cond_P = np.linalg.norm(P, 2) / np.linalg.norm(np.linalg.inv(P), 2)

                if cond_P > 1e3:
                    print(f"WARNING: cond_P = {cond_P} > 1e3, matrix is almost defective")
                    print(f"{A}")
                    return np.full((times.shape[0], y0.shape[0]), np.nan)

                X2 = X_inhom_func(rates)
                y0 = np.expand_dims(y0, -1)

                K = np.diag(np.linalg.solve(P, (y0 - X2).flatten()))

                solution = (P @ K @  np.exp(np.outer(D, times))).T + X2.T

                return solution

            return analytic_solution_func

    def make_hybrid_solver_states(self, protocol_description=None, njitted=False, strict=True):
        return self.make_solver_states(protocol_description, njitted, strict, hybrid=True)

    def make_solver_states(self, protocol_description=None, njitted=False,
                           strict=True, hybrid=True, solver_type='lsoda'):
        if protocol_description is None:
            if self.protocol_description is None:
                raise Exception("No protocol description has been provided")
            else:
                protocol_description = self.protocol_description

        voltage = self.voltage
        atol, rtol = self.solver_tolerances
        times = self.times

        p = self.get_default_parameters()
        eps = np.finfo(float).eps

        start_times = [val[0] for val in protocol_description]
        intervals = tuple(zip(start_times[:-1], start_times[1:]))

        if solver_type != 'lsoda':
            raise NotImplementedError()

        def hybrid_forward_solve_component(rhs_inf, analytic_solver, crhs_ptr, p=p,
                                           times=times, atol=atol, rtol=rtol,
                                           strict=strict):
            y0 = rhs_inf(p, voltage(.0)).flatten()
            no_states = y0.shape[0]

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
                    try:
                        step_sol[:, :] = analytic_solver(step_times - tstart,
                                                         vstart, p, y0)
                        analytic_success = np.all(np.isfinite(step_sol))

                    except Exception:
                        analytic_success = False

                    # if strict and not analytic_success:
                    #     solution[:, :] = np.nan
                    #     return solution

                if not analytic_success:
                    step_times[0] = tstart
                    step_times[-1] = tend

                    if step_times[1] - tstart < 2 * eps * np.abs(step_times[1]):
                        start_int = 1
                        step_sol[0] = y0
                    else:
                        start_int = 0

                    if tend - step_times[-1] < 2 * eps * np.abs(tend):
                        end_int = -1
                        step_sol[start_int: end_int], _ = lsoda(crhs_ptr, y0,
                                                                step_times[start_int:end_int],
                                                                data=p, rtol=rtol,
                                                                atol=atol,
                                                                exit_on_warning=True)
                    else:
                        end_int = 0
                        step_sol[start_int:], _ = lsoda(crhs_ptr, y0,
                                                        step_times[start_int:],
                                                        data=p, rtol=rtol,
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

        rhs_cfunc_ptrs = []
        for A, B, comp in zip(self.As, self.Bs, self.connected_components):
            cfunc_rhs = self.make_cfunc_rhs(A, B, comp)
            rhs_cfunc_ptrs.append(cfunc_rhs.address)

        rhs_cfunc_ptrs = tuple(rhs_cfunc_ptrs)

        no_states = self.n_states
        no_components = len(self.connected_components)

        steady_state_funcs = tuple([njit(f) if njitted else f for f in self.rhs_inf])
        analytic_solvers = tuple([njit(func) for func in self.get_analytic_solution_funcs()])

        # Cannot loop over analytic solvers for some reason. All of the 30
        # models examples have 2 components, though
        analytic_solver1 = analytic_solvers[0]
        analytic_solver2 = analytic_solvers[1]
        steady_state_func1 = steady_state_funcs[0]
        steady_state_func2 = steady_state_funcs[1]
        rhs_cfunc1 = rhs_cfunc_ptrs[0]
        rhs_cfunc2 = rhs_cfunc_ptrs[1]

        if njitted:
            hybrid_forward_solve_component = njit(hybrid_forward_solve_component)

        def hybrid_forward_solver(p=p, times=times, atol=atol, rtol=rtol, strict=strict):
            solution = np.full((times.shape[0], no_states - no_components), np.nan)
            state_counter = 0
            component_solution = \
                hybrid_forward_solve_component(steady_state_func1,
                                               analytic_solver1,
                                               rhs_cfunc1, p=p,
                                               times=times,
                                               atol=atol, rtol=rtol,
                                               strict=strict)

            solution[:, :component_solution.shape[1]] = component_solution
            state_counter += component_solution.shape[1]

            component_solution = \
                hybrid_forward_solve_component(steady_state_func2,
                                               analytic_solver2,
                                               rhs_cfunc2, p=p,
                                               times=times,
                                               atol=atol, rtol=rtol,
                                               strict=strict)

            solution[:, state_counter:] = component_solution

            return solution

        return njit(hybrid_forward_solver) if njitted else hybrid_forward_solver

    def make_forward_solver_states(self, protocol_description=None,
                                   njitted=True, solver_type='lsoda'):

        return self.make_solver_states(protocol_description, njitted=njitted, hybrid=False,
                                       solver_type=solver_type)

    def make_combined_solver(self, protocol_description=None, njitted=False,
                             strict=True, solver_type='hybrid'):
        solvers = self.make_forward_solvers_of_type(protocol_description=protocol_description,
                                                    njitted=njitted,
                                                    strict=strict,
                                                    solver_type=solver_type)

        def solver(p, times, atol, rtol, strict):
            sols = []
            for solver in solvers:
                sols.append(solver(p, times, atol, rtol, strict))

            sol = np.hstack(sols)
            return sol

        return solver

    def make_cfunc_rhs(self, A, B, comp):
        auxiliary_states = [str(s) for s in self.auxiliary_expression.free_symbols\
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

        @cfunc(lsoda_sig)
        def cfunc_rhs(t, y, dy, p):
            y = nb.carray(y, ny)
            dy = nb.carray(dy, ny)
            p = nb.carray(p, n_p)
            res = rhs_func(y, p, voltage(t)).flatten()
            for i in range(ny):
                dy[i] = res[i]

        return cfunc_rhs

    def make_ida_residual_func():
        raise NotImplementedError()

    def define_auxiliary_function(self):
        y = [var for y in self.ys for var in y]
        aux_expr = self.auxiliary_expression.subs({'E_Kr': self.E_rev})
        return sp.lambdify((y, self.p, self.v), aux_expr)


