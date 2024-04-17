import logging
import os
import time

import matplotlib.pyplot as plt
import numba
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import pints
import pints.plot
import scipy
import seaborn as sns
from numba import njit
from sympy.utilities.lambdify import _TensorflowEvaluatorPrinter

import markovmodels

from markovmodels.model_generation import make_model_of_class, setup_model_for_fitting_case
from markovmodels.voltage_protocols import get_ramp_protocol_from_csv, make_voltage_function_from_description
from markovmodels.utilities import get_data
from markovmodels.voltage_protocols import remove_spikes, detect_spikes,\
    make_voltage_function_from_description

from markovmodels.ArtefactModel import ArtefactModel, no_artefact_parameters


def fit_model(mm, data, times=None, starting_parameters=None,
              fix_parameters=[], max_iterations=None, subset_indices=None,
              method=pints.CMAES, solver=None, log_transform=True, repeats=1,
              return_fitting_df=False, parallel=False,
              randomise_initial_guess=True, output_dir=None, solver_type=None,
              no_conductance_boundary=False, use_artefact_model=False,
              rng=None):
    """
    Fit a MarkovModel to some dataset using pints.

    Params:

    mm: A MarkovModel

    data: The data set to fit to: a (1,n) numpy array
    consiting of observations corresponding to the times in mm.times.

    starting_parameters: An initial guess for the optimal parameters

    fix_parameters: Which parameters (if any) should be ignored and set to fixed values

    max_iterations: An optional upper bound on the number of evaluations PINTS should perform

    method: Which optimisation method should be used

    returns: A pair containing the optimal parameters and the corresponding sum of square errors.

    """


    if not times:
        times = mm.times

    if rng is None:
        rng = np.random.default_rng()

    if log_transform:
        # Assume that the conductance is the last parameter and that the
        # parameters are arranged included

        if mm.transformations:
            transformations = [t for i, t in enumerate(mm.transformations)
                               if i not in fix_parameters]

            if use_artefact_model:
                id_transforms = [pints.IdentityTransformation(1)
                                 for i in range(len(starting_parameters) - len(transformations))]
                transformations = transformations + id_transforms

        elif not use_artefact_model:
            # Use a-space transformation (Four Ways to Fit...)
            no_rates = int((mm.get_no_parameters() - 1)/2)
            log_transformations = [pints.LogTransformation(1) for i in range(no_rates)]
            identity_transformations = [pints.IdentityTransformation(1) for i in range(no_rates)]

            # Flatten and include conductance on the end
            transformations = [w for u, v
                               in zip(log_transformations, identity_transformations)
                               for w in (u, v)]\
                                   + [pints.IdentityTransformation(1)]

        if transformations:
            transformations = [t for i, t in enumerate(transformations) if i not in fix_parameters]
            transformation = pints.ComposedTransformation(*transformations)

        else:
            raise Exception("Couldn't log transform parameters")

    else:
        transformation = None

    if starting_parameters is None:
        starting_parameters = mm.get_default_parameters()

    if transformations:
        assert len(transformations) == len(starting_parameters) - len(fix_parameters)

    if max_iterations == 0:
        return starting_parameters, np.inf

    if solver is None:
        try:
            solver = mm.make_forward_solver_of_type(solver_type)
        except numba.core.errors.TypingError as exc:
            logging.warning(f"unable to make nopython forward solver {str(exc)}")
            solver = mm.make_forward_solver_of_type(solver_type, njitted=False)

    if subset_indices is None:
        subset_indices = np.array(list(range(len(mm.times))))

    fix_parameters = np.unique(fix_parameters)

    class PintsWrapper(pints.ForwardModelS1):
        def __init__(self, mm, parameters, fix_parameters=None):
            self.mm = mm
            self.parameters = np.array(parameters)

            self.fix_parameters = fix_parameters

            unfixed_parameters = tuple([i for i in range(len(parameters)) if i not in fix_parameters])
            if fix_parameters is None:
                fix_parameters = tuple()

            if len(fix_parameters) > 0:
                def simulate(p, times):
                    sim_parameters = np.copy(parameters)
                    for i, j in enumerate(unfixed_parameters):
                        sim_parameters[j] = p[i]
                    sol = solver(sim_parameters)[subset_indices]
                    return sol
            else:
                def simulate(p, times):
                    # try:
                    return solver(p)[subset_indices]
                    # except Exception:
                    #     return np.full(times.shape, np.inf)

            self.simulate = simulate

        def n_parameters(self):
            return len(self.parameters) - len(self.fix_parameters)

        def simulateS1(self, parameters, times):
            raise NotImplementedError()

    model = PintsWrapper(mm, starting_parameters,
                         fix_parameters=fix_parameters)

    problem = pints.SingleOutputProblem(model, times[subset_indices],
                                        data[subset_indices])

    error = pints.SumOfSquaresError(problem)

    if len(fix_parameters) != 0:
        unfixed_indices = [i for i in range(
            len(starting_parameters)) if i not in fix_parameters]
        params_not_fixed = starting_parameters[unfixed_indices]
    else:
        unfixed_indices = list(range(len(starting_parameters)))
        params_not_fixed = starting_parameters

    voltages = mm.GetVoltage().flatten()
    boundaries = fitting_boundaries(starting_parameters, mm, data,
                                    voltages, rng, fix_parameters,
                                    is_artefact_model=use_artefact_model)

    if randomise_initial_guess:
        initial_guess_dist = boundaries
        starting_parameter_sets = []

    scores, parameter_sets, iterations, times_taken = [], [], [], []
    for i in range(repeats):
        if randomise_initial_guess:
            initial_guess = initial_guess_dist.sample(n=1).flatten()
            starting_parameter_sets.append(initial_guess)
            params_not_fixed = initial_guess

        if np.any(~np.isfinite(params_not_fixed)):
            raise ValueError(f"starting parameter lie outside boundary: {params_not_fixed}")

        controller = pints.OptimisationController(error, params_not_fixed,
                                                  boundaries=boundaries,
                                                  method=method,
                                                  transformation=transformation)
        if not parallel:
            controller.set_parallel(False)

        try:
            if max_iterations is not None:
                controller.set_max_iterations(max_iterations)

        except Exception as e:
            print(str(e))
            found_value = np.inf
            found_parameters = starting_parameters

        timer_start = time.process_time()
        found_parameters, found_value = controller.run()
        timer_end = time.process_time()
        time_elapsed = timer_end - timer_start

        this_run_iterations = controller.iterations()
        parameter_sets.append(found_parameters)
        scores.append(found_value)
        iterations.append(this_run_iterations)
        times_taken.append(time_elapsed)

    best_score = min(scores)
    best_index = scores.index(best_score)
    best_parameters = parameter_sets[best_index]

    if not np.all(np.isfinite(model.simulate(found_parameters, mm.times))):
        best_parameters = [p for i, p in enumerate(mm.get_default_parameters())
                           if i not in fix_parameters]
        best_score = np.inf

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        point_2 = [p for i, p in enumerate(mm.get_default_parameters()) if i not
                   in fix_parameters]
        fig, axes = pints.plot.function_between_points(error,
                                                       point_1=best_parameters,
                                                       point_2=point_2,
                                                       padding=0.1,
                                                       evaluations=100)

        fig.savefig(os.path.join(output_dir, 'best_fitting_profile_from_default'))
        plt.close(fig)

        if randomise_initial_guess:
            point_2 = starting_parameter_sets[best_index % len(starting_parameter_sets)]
            fig, axes = pints.plot.function_between_points(error,
                                                           point_1=best_parameters,
                                                           point_2=point_2,
                                                           padding=0.1,
                                                           evaluations=100)
            fig.savefig(os.path.join(output_dir, 'best_fitting_profile_from_initial_guess'))
            plt.close(fig)

    if len(fix_parameters) > 0:
        for i in np.unique(fix_parameters):
            best_parameters = np.insert(best_parameters,
                                        i,
                                        starting_parameters[i])
    if return_fitting_df:
        if len(fix_parameters) > 0:
            new_rows = parameter_sets
            for i in np.unique(fix_parameters):
                for j, row in enumerate(parameter_sets):
                    new_rows[j] = np.insert(row, i, starting_parameters[i])
            parameter_sets = np.array(new_rows)
        else:
            parameter_sets = np.vstack(parameter_sets)
        fitting_df = pd.DataFrame(parameter_sets,
                                  columns=mm.get_parameter_labels()[:parameter_sets.shape[1]])
        fitting_df['RMSE'] = np.array(scores) / len(subset_indices)
        fitting_df['iterations'] = iterations
        fitting_df['CPU_time'] = times_taken

        # Append starting parameters also
        if randomise_initial_guess:
            columns = np.array(mm.get_parameter_labels())
            initial_guess_df = pd.DataFrame(starting_parameter_sets,
                                            columns=columns[unfixed_indices])
            initial_guess_df['iterations'] = iterations
            initial_guess_df['RMSE'] = np.NaN

            fitting_df = pd.concat([fitting_df, initial_guess_df],
                                   ignore_index=True)

        return best_parameters, best_score, fitting_df
    else:
        return best_parameters, best_score


def fit_well_data(model_class_name: str, well, protocol, data_directory,
                  max_iterations, output_dir=None, T=None, K_in=None,
                  K_out=None, default_parameters: float = None,
                  removal_duration=5, repeats=1, infer_E_rev=False,
                  fit_initial_conductance=True, experiment_name='newtonrun4',
                  solver=None, E_rev=None, randomise_initial_guess=True,
                  parallel=False, solver_type=None, sweep=None,
                  scale_conductance=True, no_conductance_boundary=False,
                  use_artefact_model=False, artefact_default_kinetic_parameters=None,
                  fix_parameters=[], data_label=None, tolerance=None):

    if default_parameters is None or len(default_parameters) == 0:
        default_parameters = make_model_of_class(model_class_name).get_default_parameters()

    if not use_artefact_model:
        parameter_labels = make_model_of_class(model_class_name).get_parameter_labels()

    else:
        parameter_labels = ArtefactModel(make_model_of_class(model_class_name))\
            .get_parameter_labels()

    if max_iterations == 0 or not np.all(np.isfinite(default_parameters)):
        df = pd.DataFrame(default_parameters[None, :], columns=parameter_labels)
        df['score'] = np.inf
        return df

    data, voltage_protocol = get_data(well, protocol, data_directory, experiment_name,
                                      label=data_label, sweep=sweep)
    protocol_desc = voltage_protocol.get_all_sections()

    # Temporary solver hack
    protocol_desc = np.vstack((protocol_desc, [[protocol_desc[-1, 1], np.inf, -80.0, -80.0]]))

    voltage_func = make_voltage_function_from_description(protocol_desc)

    times = pd.read_csv(os.path.join(data_directory, f"{experiment_name}-{protocol}-times.csv"),
                        header=None).values.flatten()

    voltages = np.array([voltage_func(t) for t in times])
    spike_times, _ = detect_spikes(times, voltages, window_size=0)
    _, _, indices = remove_spikes(times, voltages, spike_times,
                                  removal_duration)

    V_off = 0.0
    if infer_E_rev:
        if output_dir:
            plot = True
            reversal_dir = os.path.join(output_dir, 'reversal_plots')
            try:
                os.makedirs(reversal_dir)
            except FileExistsError:
                pass

            output_path = os.path.join(reversal_dir,
                                        'infer_reversal_potential.png')
        else:
            output_path = None
            reversal_dir = None

        if use_artefact_model:
            # Use the artefact to forward simulate the voltages (using literature kinetics)
            params_for_Erev = default_parameters.copy()

            V_off_model_class = 'model3'

            if artefact_default_kinetic_parameters is not None:
                V_off_initial_params = \
                    np.append(artefact_default_kinetic_parameters, default_parameters[-no_artefact_parameters:])
            else:
                V_off_initial_params = \
                    np.append(make_model_of_class(V_off_model_class).get_default_parameters(),
                              default_parameters[-no_artefact_parameters:])

                print(V_off_initial_params)

            try:
                V_off = find_V_off(protocol_desc, times,
                                   data, V_off_model_class,
                                   V_off_initial_params, E_rev,
                                   forward_sim_output_dir=reversal_dir,
                                   removal_duration=removal_duration,
                                   output_path=output_path,
                                   data_label=data_label
                                   )
            except ValueError as exc:
                # Possibly non data or non-finite values in data
                logging.warning("error whilst inferring V_off: ", str(exc))
                df = pd.DataFrame(default_parameters[None, :], columns=parameter_labels)
                df['score'] = np.inf
                return df

        else:
            voltages = None
            E_obs = infer_reversal_potential(protocol_desc, data, times, plot=plot,
                                             output_path=output_path, voltages=voltages)
        if use_artefact_model:
            inferred_E_rev = E_rev
            default_parameters[-3] = V_off
            E_rev = E_rev
        else:
            inferred_E_rev = E_obs

        if inferred_E_rev < -50 or inferred_E_rev > -100:
            E_rev = inferred_E_rev

    m_model = make_model_of_class(model_class_name, voltage=voltage_func,
                                  times=times,
                                  E_rev=E_rev,
                                  protocol_description=protocol_desc,
                                  tolerances=tolerance)

    if use_artefact_model:
        model = ArtefactModel(m_model)
    else:
        model = m_model

    initial_params = default_parameters.flatten()

    columns = model.get_parameter_labels()

    if infer_E_rev:
        columns.append("E_rev")

    if solver is not None and solver_type is not None:
        raise Exception('solver and solver type provided')

    if solver is None:
        strict = False
        try:
            if use_artefact_model and data_label == 'before':
                assert solver_type is None or solver_type=='default'
                solver = model.make_hybrid_solver_current(strict=strict, return_var='I_out',
                                                          hybrid=False)
            else:
                solver = model.make_forward_solver_of_type(solver_type,
                                                           strict=strict)
            solver()

        except numba.core.errors.TypingError as exc:
            logging.warning(f"unable to make nopython forward solver {str(exc)}")
            solver = model.make_forward_solver_of_type(solver_type, njitted=False,
                                                       strict=strict)

    if not np.all(np.isfinite(solver(initial_params.flatten()))):
        if use_artefact_model:
            print(model.SimulateForwardModel(initial_params))

        bad_indices = np.argwhere(~np.isfinite(solver(initial_params.flatten())))

        print(times[bad_indices])
        print(voltages[bad_indices])

        print(len(times))

        state_solver = model.make_hybrid_solver_states(hybrid=False, njitted=False)
        print(state_solver()[bad_indices, :])

        raise Exception("Default parameters gave non-finite output \n"
                        f"{well} {protocol} {sweep} {initial_params} {E_rev}")

    fitted_params, score, fitting_df = fit_model(model, data, solver=solver,
                                                 starting_parameters=initial_params,
                                                 max_iterations=max_iterations,
                                                 subset_indices=indices,
                                                 parallel=parallel,
                                                 randomise_initial_guess=randomise_initial_guess,
                                                 return_fitting_df=True,
                                                 repeats=repeats,
                                                 output_dir=output_dir,
                                                 solver_type=solver_type,
                                                 use_artefact_model=use_artefact_model,
                                                 no_conductance_boundary=no_conductance_boundary,
                                                 fix_parameters=fix_parameters)

    fig = plt.figure(figsize=(14, 12))
    ax = fig.subplots()
    for i, row in fitting_df.iterrows():
        fitted_params = row[model.get_parameter_labels()].values.flatten()
        try:
            ax.plot(times, solver(fitted_params), label='fitted parameters')
            ax.plot(times, solver(initial_params), label='default parameters')
            ax.plot(times, data, color='grey', label='data', alpha=.5)
        except Exception:
            pass

        ax.legend()

        if infer_E_rev:
            fitted_params = np.append(fitted_params, E_rev)

        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            fname = f"{well}_{protocol}_fit_{i}" if i < repeats else f"{well}_{protocol}_initial_guess_{i}"

            fig.savefig(os.path.join(output_dir, fname))
            ax.cla()
    plt.close(fig)

    fitting_df['score'] = fitting_df['RMSE']
    fitting_df.to_csv(os.path.join(output_dir, f"{well}_{protocol}_fitted_params.csv"))
    return fitting_df


def compute_mcmc_chains(model, times, indices, data, solver=None,
                        starting_parameters=None, sigma2=1, no_chains=1,
                        chain_length=1000, burn_in=None, log_likelihood_func=None,
                        log_transform=True):
    n = len(indices)

    if solver is None:
        solver = model.make_forward_solver_current()

    if starting_parameters is None:
        starting_parameters = model.get_default_parameters().flatten()

    if log_transform:
        # Assume that the conductance is the last parameter and that the parameters are arranged included

        # log all parameters
        no_rates = int((model.get_no_parameters() - 1)/2)
        # log_transformations = [pints.LogTransformation(1) for i in range(no_rates)]
        # identity_transformations = [pints.IdentityTransformation(1) for i in range(no_rates)]

        # Flatten and include conductance on the end
        transformations = [pints.LogTransformation(1) for i in range(len(starting_parameters))]
        transformation = pints.ComposedTransformation(*transformations)

    else:
        transformation = None

    if burn_in is None:
        burn_in = int(chain_length / 10)

    if starting_parameters is None:
        starting_parameters = model.get_default_parameters()

    if log_likelihood_func is None:
        @njit
        def log_likelihood_func(p):

            if np.any(p <= 0):
                ll = -np.inf
            else:
                try:
                    output = solver(p, times)[indices]
                    error = output - data[indices]
                    SSE = np.sum(error**2)
                    ll = -n * 0.5 * np.log(2 * np.pi * sigma2) - SSE / (2 * sigma2)

                except Exception:
                    ll = -np.inf

            return ll

    class pints_likelihood(pints.LogPDF):
        def __call__(self, p):
            return log_likelihood_func(p)

        def n_parameters(self):
            return len(starting_parameters)

    class pints_prior(pints.LogPrior):
        def __init__(self, parameters=starting_parameters):
            self.parameters = parameters

        def __call__(self, parameters=starting_parameters):
            # Make sure transition rates are not too big
            for i in range(int(len(parameters)/2)):
                a = parameters[2*i]
                b = parameters[2*i + 1]

                vs = np.array([-120, 40])

                extreme_rates = np.abs(a*np.exp(b*vs))
                max_rate = np.max(extreme_rates)
                min_rate = np.min(extreme_rates)

                if max_rate > 1e7:
                    return -np.inf

                # if min_rate < 1e-8:
                #     return -np.inf

            # Ensure that all parameters > 0
            return 0 if np.all(parameters > 0) else -np.inf

        def n_parameters(self):
            return model.get_no_parameters()

    posterior = pints.LogPosterior(pints_likelihood(), pints_prior())

    initial_likelihood = log_likelihood_func(starting_parameters)

    print('initial_parameters likelihood = ', initial_likelihood)
    if not np.isfinite(initial_likelihood):
        print("{model} MCMC failed, initial parameters had non-finite log likelihood")
        return np.full((no_chains, chain_length, len(starting_parameters)), np.nan)

    print(f"initial likelihood is {initial_likelihood}")

    mcmc = pints.MCMCController(posterior, no_chains,
                                np.tile(starting_parameters, [no_chains, 1]),
                                method=pints.HaarioBardenetACMC,
                                transformation=transformation)

    mcmc.set_max_iterations(chain_length)

    try:
        samples = mcmc.run()
    except ValueError as exc:
        print(str(exc))
        return np.full((no_chains, chain_length, len(starting_parameters)), np.nan)

    return samples[:, burn_in:, :]


class fitting_boundaries(pints.Boundaries):
    def __init__(self, full_parameters, model, current, voltages, rng,
                 fix_parameters=[], is_artefact_model=False):
        self.is_artefact_model = is_artefact_model

        if is_artefact_model:
            self.mm = model.channel_model

            self.fix_parameters = [
                i for i in fix_parameters
                if (i % len(full_parameters)) < self.mm.get_no_parameters()]

            self.full_parameters = full_parameters[:self.mm.get_no_parameters()].copy()

        else:
            self.fix_parameters = fix_parameters
            self.full_parameters = full_parameters
            self.mm = model

        indices = np.argwhere(voltages - -120.0 < 1e-5)[10:200]
        conductances = (current / (voltages - self.mm.E_rev))[indices]

        self.max_conductance = np.abs(conductances.max()) * 100
        self.min_conductance = np.abs(conductances.max()) * 0.01

        self.rates_func = njit(self.mm.get_rates_func(njitted=False))

        self.rng = rng

    def check(self, parameters):
        parameters = parameters.copy()
        if len(self.fix_parameters) != 0:
            for i in np.unique(self.fix_parameters):
                # TODO repeated calls to insert are inefficient. Replace with
                # something better
                if i < len(self.full_parameters) - 1:
                    parameters = np.insert(parameters, i, self.full_parameters[i])

        parameters = parameters[:self.mm.GKr_index + 1].flatten()

        if np.any(parameters[:self.mm.GKr_index + 1] < 0):
            return False

        if max([p for i, p in enumerate(parameters) if i != self.mm.GKr_index]) > 1e5:
            return False

        if min([p for i, p in enumerate(parameters) if i != self.mm.GKr_index]) < 1e-7:
            return False

        if parameters[self.mm.GKr_index] > self.max_conductance:
            return False

        if parameters[self.mm.GKr_index] < self.min_conductance:
            return False

        Vs = [-120, 60]
        rates_func = self.rates_func
        rates_1 = rates_func(parameters, Vs[0]).flatten()
        rates_2 = rates_func(parameters, Vs[1]).flatten()

        max_transition_rates = np.max(np.vstack([rates_1, rates_2]), axis=0)

        if np.any(max_transition_rates > 1e3):
            return False

        if np.any(max_transition_rates < 1.67e-5):
            return False

        return True

    def n_parameters(self):
        return self.mm.get_no_parameters() - \
            len(self.fix_parameters) if len(self.fix_parameters) != 0 \
            else self.mm.get_no_parameters()

    def _sample_once(self, min_log_p, max_log_p):
        rng = self.rng

        # Reject samples that don't lie in the boundaries
        # try 1000 times before giving up. This should be plenty
        for i in range(1000):
            p = np.empty(self.full_parameters.shape)

            p[:self.mm.GKr_index] = 10**rng.uniform(min_log_p, max_log_p,
                                                        self.full_parameters.shape[0] - 1)
            p[self.mm.GKr_index] = 0.5 * (self.min_conductance + self.max_conductance)
            if len(self.fix_parameters) != 0:
                p = p[[i for i in range(len(self.full_parameters)) if i not in
                       self.fix_parameters]]

            # Check this lies in boundaries
            if self.check(p):
                if self.mm.GKr_index not in self.fix_parameters:
                    gkr_index = self.mm.GKr_index - np.sum(np.array(self.fix_parameters)\
                                                           < self.mm.GKr_index)

                p[gkr_index] = 10 ** (rng.uniform(np.log10(self.min_conductance),
                                                  np.log10(self.max_conductance)))
                return p

        logging.warning("Couldn't sample from boundaries")
        return np.full(p.shape, np.nan)

    def sample(self, n=1):
        min_log_p, max_log_p = [-7, 1]

        ret_vec = np.full((n, len(self.full_parameters)), np.nan)
        for i in range(n):
            ret_vec[i, :] = self._sample_once(min_log_p, max_log_p)

        params_not_fixed = [i for i in range(len(self.mm.get_default_parameters()))\
                            if i not in self.fix_parameters]

        ret_vec = ret_vec[:, params_not_fixed]
        return ret_vec


def _find_conductance(solver, data, indices, aux_func, voltages, p, Erev, gkr_index,
                      bounds=None, return_var='I_Kr'):

    if bounds is None:
        # Look at -120 step
        b_indices = np.argwhere(np.abs(voltages + 120)<1e-2)
        bounds = np.unique([0, (data[b_indices]/(voltages[b_indices] - Erev)).max() * 20])

    def find_g_opt(g):
        _p = p.copy()
        _p[gkr_index] = float(g)

        states = solver(_p)

        if not np.all(np.isfinite(states)):
            return np.inf

        prediction = aux_func(states.T, _p, voltages).flatten()

        score = np.sqrt(np.mean((prediction[indices] - data[indices])**2))
        return score

    options = {'xatol': bounds.max() * 1e-8}
    # Find conductance
    res = scipy.optimize.minimize_scalar(find_g_opt,
                                         bounds=bounds,
                                         options=options)

    if res.success:
        return res.x
    return np.nan



def find_V_off(protocol_desc, times, data,
               model_class_name,
               default_parameters, E_rev,
               forward_sim_output_dir=None,
               removal_duration=0,
               output_path=None,
               data_label=''):

    voltage_func = make_voltage_function_from_description(protocol_desc)
    voltages = np.array([voltage_func(t) for t in times])

    spike_times, _ = detect_spikes(times, voltages, window_size=0)
    _, _, indices = remove_spikes(times, voltages, spike_times,
                                  removal_duration)

    # Only use the last part of the protocol for guessing the conductance
    # Find last +40 step
    start_step = [line for line in protocol_desc if
                  line[2] == line[3] and line[2] == +40.0][-1]
    start_t = start_step[1]

    # Find end of reversal ramp
    end_ramp = [line for line in protocol_desc if line[2] != line[3]][-1]
    end_t = end_ramp[1] + 200

    indices = indices[np.argwhere((times[indices] > start_t) & (times[indices] <= end_t))]

    model = make_model_of_class(model_class_name, voltage=voltage_func,
                                times=times,
                                E_rev=E_rev,
                                protocol_description=protocol_desc,
                                tolerances=(1e-6, 1e-6))

    gkr_index = model.GKr_index
    model = ArtefactModel(model)

    a_solver = model.make_hybrid_solver_states(hybrid=False, njitted=False,
                                               strict=False)

    return_var = 'I_out' if data_label == 'before' else None

    aux_func = model.define_auxiliary_function(return_var=return_var)
    step = next(filter(lambda x: x[2] >= -74, reversed(protocol_desc)))

    if step[1] - step[0] > 200 or step[1] - step[0] < 50:
        raise Exception("Failed to find reversal ramp in protocol")
    else:
        step = step[0:2]

    # Next extract steps
    istart = np.argmax(times >= step[0])
    iend = np.argmax(times > step[1])

    p = default_parameters.copy()
    p[-3] = 0.0
    states = a_solver(p)


    if data_label == 'before':
        gleak, Eleak = p[-7:-5]
        s_data = data - gleak * (voltages - Eleak)
    else:
        s_data = data


    initial_gkr = _find_conductance(a_solver, data, indices, aux_func,
                                    voltages, p, E_rev, gkr_index)
    E_obs = infer_reversal_potential(protocol_desc, s_data, times)

    def opt_V_off_func(V_off):
        p = default_parameters.copy()

        p[gkr_index] = initial_gkr

        gkr = _find_conductance(a_solver, data, indices, aux_func, voltages, p,
                                E_rev, gkr_index)

        p[gkr_index] = gkr
        if not np.isfinite(gkr):
            return np.inf

        p[-3] = V_off
        states = a_solver(p)
        V_m = states[:, -1]
        trace = aux_func(states.T, p, voltages).flatten()

        # Do leak subtraction if handling raw trace
        if data_label == 'before':
            gleak, Eleak = p[-7:-5]
            s_trace = trace - gleak * (voltages - Eleak)
        else:
            # Leave trace untouched otherwise
            s_trace = trace

        # Get voltage where current first crosses 0 in the reversal ramp
        expected_E_obs = voltages[istart:iend][np.argmax(s_trace[istart:iend] < 0)]

        if np.any(~np.isfinite(trace)):
            return np.inf

        if expected_E_obs > voltages.max() or expected_E_obs < voltages.min():
            return np.inf

        score = (E_obs - expected_E_obs)**2
        return score

    # Set bounds for optimisation

    E_obs = infer_reversal_potential(protocol_desc, s_data, times,
                                     voltages=voltages)

    E_rev_error = E_obs - E_rev
    bounds = np.unique([-E_rev_error*2, 0])

    options = {'xatol': 1e-8}

    res = scipy.optimize.minimize_scalar(opt_V_off_func,
                                         bounds=bounds,
                                         method='bounded')
    if res.success:
        found_V_off = res.x

    else:
        return np.inf

    p[-3] = found_V_off
    gkr = _find_conductance(a_solver, data, indices, aux_func, voltages, p,
                            E_rev, gkr_index)
    p[gkr_index] = gkr

    if output_path:
        fig = plt.figure()
        ax = fig.subplots()

        states = a_solver(p)
        V_m = states[:, -1]
        trace = aux_func(states.T, p, voltages).flatten()
        # s_trace = aux_func_IKr(states.T, p, _)
        # Do leak subtraction if handling raw tra wce
        if data_label == 'before':
            gleak, Eleak = p[-7:-5]
            s_data = data - gleak * (voltages - Eleak)
            s_trace = trace - gleak * (voltages - Eleak)
        else:
            # Leave trace untouched otherwise
            s_trace = trace
            s_data = data

        ax.scatter(V_m[istart:iend], s_data[istart:iend], marker='x', color='grey',
                   s=5, label=r'$V_\text{m}$ reversal ramp')
        ax.scatter(voltages[istart:iend], s_data[istart:iend], marker='x',
                   color='pink', alpha=.5, s=5, label=r'$V_\text{cmd}$ reversal ramp')
        ax.scatter(V_m[istart:iend], s_trace[istart:iend], marker='x', s=5,
                   label=r'reference model $V_\text{m}$ reversal ramp')

        ax.scatter(voltages[istart:iend], s_trace[istart:iend], marker='x', s=5,
                   label=r'reference model with $V_\text{cmd}$ during reversal ramp')

        ax.axvline(E_rev, color='grey', linestyle='--', label=r'$E_\text{Nernst}$')

        if res.success:
            ax.axvline(E_rev - found_V_off, label=r'$E_\text{Kr} - V_\text{off}$',
                       color='red', ls='--')

        ax.axhline(0, linestyle='--', color='orange')

        ax.legend()

        fig.savefig(output_path)
        plt.close(fig)

    if forward_sim_output_dir and res.success:
        fig = plt.figure()
        axs = fig.subplots(2)

        # Find conductance
        gkr = _find_conductance(a_solver, data, indices, aux_func, voltages, p,
                                E_rev, gkr_index)

        p[-3] = found_V_off
        p[gkr_index] = gkr
        states = a_solver(p)

        prediction = aux_func(states.T, p, voltages)

        if data_label == 'before':
            gleak, Eleak = p[-7:-5]
            prediction = prediction - gleak * (voltages - Eleak)

        axs[0].plot(times, s_data, color='grey', alpha=.5, label='data')
        axs[0].plot(times, prediction, label='forward_sim')

        Vm = states[:, -1].flatten()

        axs[1].plot(times, voltages, label='Vcmd')
        axs[1].plot(times, Vm, label='Vm')

        fig.savefig(os.path.join(forward_sim_output_dir, 'simulation.png'))
        plt.close(fig)

    if res.success:
        return found_V_off

    return np.nan


def infer_reversal_potential(protocol_desc: np.array, current: np.array, times, ax=None,
                             output_path=None, plot=None, known_Erev=None, voltages=None):
    if output_path:
        dirname = os.path.dirname(output_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    if (ax or output_path) and plot is not False:
        plot = True

    # First, find the reversal ramp. Search backwards along the protocol until we find a >= 40mV step
    step = next(filter(lambda x: x[2] >= -74, reversed(protocol_desc)))


    if step[1] - step[0] > 200 or step[1] - step[0] < 50:
        raise Exception("Failed to find reversal ramp in protocol")
    else:
        step = step[0:2]

    # Next extract steps
    istart = np.argmax(times >= step[0])
    iend = np.argmax(times > step[1])

    if istart == 0 or iend == 0 or istart == iend:
        raise Exception("Couldn't identify reversal ramp")

    full_times = times

    protocol_func = make_voltage_function_from_description(protocol_desc)
    if voltages is None:
        voltages = np.array([protocol_func(t) for t in times])

    times = times[istart:iend]
    current = current[istart:iend]

    voltages = voltages[istart:iend]

    fitted_poly = poly.Polynomial.fit(voltages, current, 4)

    roots = np.unique([np.real(root) for root in fitted_poly.roots()
                       if root > np.min(voltages) and root < np.max(voltages)])

    # Take the last root (greatest voltage). This should be the first time that
    # the current crosses 0 and where the ion-channel kinetics are too slow to
    # play a role

    if len(roots) == 0:
        return np.nan

    if plot:
        created_fig = False
        if ax is None and output_path is not None:

            created_fig = True
            fig = plt.figure()
            ax = fig.subplots()

        ax.set_xlabel('voltage mV')
        ax.set_ylabel('current nA')
        # Now plot current vs voltage
        ax.plot(voltages, current, 'x', markersize=2, color='grey', alpha=.5)
        if known_Erev:
            prot_func, _, desc = markovmodels.voltage_protocols.get_ramp_protocol_from_csv('staircaseramp1')

            c_model = make_model_of_class('model3',
                                          voltage=prot_func, times=full_times, E_rev=known_Erev,
                                          protocol_description=desc,
                                          tolerances=[1e-8, 1e-8])
            IKr = c_model.SimulateForwardModel()[istart:iend]
            scaling_factor = np.max(np.abs(current)) / np.max(np.abs(IKr))
            scaled_IKr = IKr * scaling_factor

            ideal_voltages = [prot_func(t) for t in times]

            ax.plot(ideal_voltages, scaled_IKr, '--', color='red', label='ideal IKr with Nernst potential')

        ax.axvline(roots[-1], linestyle='--', color='grey', label="$E_{Kr}$")
        if known_Erev:
            ax.axvline(known_Erev, linestyle='--', color='yellow', label="known $E_{Kr}$")
        ax.axhline(0, linestyle='--', color='grey')
        ax.plot(*fitted_poly.linspace())
        ax.legend()

        if output_path is not None:
            fig = ax.figure
            fig.savefig(output_path)

        if created_fig:
            plt.close(fig)

    return roots[-1]


def compute_predictions_df(params_df, output_dir, protocol_dict, fitting_case, E_rev, subtractions_df,
                           label='predictions', model_class=None,
                           default_artefact_kinetic_parameters=None, args=None,
                           data_label=''):

    param_labels = make_model_of_class(model_class).get_parameter_labels()
    params_df = get_best_params(params_df, protocol_label='protocol')
    predictions_dir = os.path.join(output_dir, label)

    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    predictions_df = []
    protocols_list = list(subtractions_df['protocol'].unique()) + ['longap']
    protocols_list = np.array(protocols_list).astype(str)

    trace_fig = plt.figure(figsize=args.figsize)
    trace_axs = trace_fig.subplots(2)

    all_models_fig = plt.figure(figsize=args.figsize)
    all_models_axs = all_models_fig.subplots(2)

    use_artefacts = True if fitting_case in ['I', 'II'] else False

    model = make_model_of_class(model_class)
    solver = None

    prot_func = None

    for sim_protocol in np.unique(protocols_list):

        desc, full_times = protocol_dict[sim_protocol]

        # Temporary solver hack
        desc = np.vstack((desc, [[desc[-1, 1], np.inf, -80.0, -80.0]]))


        if prot_func is None:
            prot_func = make_voltage_function_from_description(desc)

        voltages = np.array([prot_func(t, protocol_description=desc) for t in full_times])

        if solver is None:
            solver = model.make_hybrid_solver_current(hybrid=False,
                                                      njitted=True,
                                                      strict=False,
                                                      protocol_description=desc,
                                                      voltage=prot_func)

        spike_times, spike_indices = markovmodels.voltage_protocols.detect_spikes(full_times, voltages,
                                                                                  threshold=10)
        _, _, indices = markovmodels.voltage_protocols.remove_spikes(full_times, voltages, spike_times,
                                                   time_to_remove=args.removal_duration)
        times = full_times[indices]

        colours = sns.color_palette('husl', len(params_df['protocol'].unique()))

        for well in params_df['well'].unique():
            for predict_sweep in params_df.sweep.unique():

                try:
                    inferred_E_rev = subtractions_df.set_index(['protocol', 'well', 'sweep']).loc[(sim_protocol, well, predict_sweep)]['E_rev']
                except KeyError:
                    # A key error is thrown if this (protocol, sweep) combination doesn't exist
                    continue

                if fitting_case == '0c':
                    adjusted_params_df = adjust_kinetics(model_class, params_df, subtractions_df,
                                                         E_rev, inferred_E_rev)
                else:
                    adjusted_params_df = params_df

                subdir_name = f"{well}_{sim_protocol}_sweep{predict_sweep}_predictions"\
                    if predict_sweep is not None else f"{well}_{sim_protocol}_predictions"
                sub_dir = os.path.join(predictions_dir, subdir_name)

                if not os.path.exists(sub_dir):
                    os.makedirs(sub_dir)

                data_label = ''
                if args:
                    if 'data_label' in args:
                        data_label = args.data_label
                try:
                    full_data, vp = markovmodels.utilities.get_data(well,
                                                                    sim_protocol,
                                                                    args.data_directory,
                                                                    experiment_name=args.experiment_name,
                                                                   label=data_label,
                                                                   sweep=predict_sweep)
                except (FileNotFoundError, StopIteration) as exc:
                    print(str(exc))
                    continue

                data = full_data[indices]
                for i, protocol_fitted in enumerate(params_df.protocol.unique()):
                    # print(protocol_fitted)
                    for fitting_sweep in params_df[params_df.protocol == protocol_fitted].sweep.unique():
                        full_prediction = make_prediction(model_class, args,
                                                          well, sim_protocol,
                                                          predict_sweep,
                                                          protocol_fitted,
                                                          fitting_sweep,
                                                          params_df,
                                                          subtractions_df,
                                                          fitting_case, E_rev,
                                                          protocol_dict,
                                                          full_data, voltages,
                                                          label=data_label,
                                                          solver=solver)

                        prediction = full_prediction[indices]

                        score = np.sqrt(np.mean((data - prediction)**2))
                        n_score = score / np.sqrt(np.mean(data**2))

                        df = adjusted_params_df[adjusted_params_df.well == well]
                        df = df[(df.protocol == protocol_fitted) & (df.sweep == fitting_sweep)]
                        if df.empty:
                            return np.full(full_times.shape, None)

                        params = df.iloc[0][param_labels].values\
                                                            .astype(np.float64)\
                                                            .flatten()


                        if not np.all(np.isfinite(prediction)):
                            logging.warning(f"running {sim_protocol} with parameters "
                                            f"from {protocol_fitted} gave non-finite values")
                            print(times[~np.isfinite(prediction)])
                        else:
                            predictions_df.append((well, protocol_fitted,
                                                   fitting_sweep,
                                                   predict_sweep, sim_protocol,
                                                   score, n_score, E_rev,
                                                   *params))
                            # Output trace
                            trace_axs[0].plot(full_times, full_prediction, label='prediction')

                            trace_axs[1].set_xlabel("time / ms")
                            trace_axs[0].set_ylabel("current / nA")
                            trace_axs[0].plot(times, data, label='data', alpha=0.25, color='grey')
                            trace_axs[0].legend()
                            trace_axs[1].plot(full_times, voltages)
                            trace_axs[1].set_ylabel('voltage / mV')
                            fname = f"fitted_to_{protocol_fitted}_{fitting_sweep}.png" if protocol_fitted != sim_protocol or \
                                fitting_sweep != predict_sweep else "fit.png"

                            handles, labels = trace_axs[1].get_legend_handles_labels()
                            by_label = dict(zip(labels, handles))
                            plt.legend(by_label.values(), by_label.keys())

                            trace_fig.savefig(os.path.join(sub_dir, fname))

                            for ax in trace_axs:
                                ax.cla()

                            all_models_axs[0].plot(full_times, full_prediction,
                                                   label=f"{protocol_fitted}_{fitting_sweep}", color=colours[i])

                all_models_axs[1].set_xlabel("time / ms")
                all_models_axs[0].set_ylabel("current / nA")
                all_models_axs[0].plot(times, data, color='grey', alpha=0.5, label='data')
                # all_models_axs[0].legend()
                all_models_axs[0].set_title(f"{well} {sim_protocol} fits comparison")
                all_models_axs[0].set_ylabel("Current / nA")

                all_models_axs[1].plot(full_times, voltages)
                all_models_axs[1].set_ylabel('voltage / mV')

                all_models_fig.savefig(os.path.join(sub_dir, "all_fits.png"))

                for ax in all_models_axs:
                    ax.cla()

    if len(predictions_df) == 0:
        raise Exception("No predictions produced")

    predictions_df = pd.DataFrame(np.array(predictions_df), columns=['well',
                                                                     'fitting_protocol',
                                                                     'fitting_sweep',
                                                                     'prediction_sweep',
                                                                     'validation_protocol',
                                                                     'score', 'n_score',
                                                                     'E_rev'] +
                                  param_labels)
    predictions_df['RMSE'] = predictions_df['score'].astype(np.float64)
    predictions_df['sweep'] = predictions_df.fitting_sweep

    plt.close(trace_fig)
    plt.close(all_models_fig)

    return predictions_df


def get_best_params(fitting_df, protocol_label='protocol'):
    best_params = []

    # Ensure score is a float - it may be read from csv file
    fitting_df['score'] = fitting_df['score'].astype(np.float64)
    fitting_df = fitting_df[np.isfinite(fitting_df['score'])].copy()

    if 'sweep' not in fitting_df.columns:
        fitting_df['sweep'] = -1

    for protocol in fitting_df[protocol_label].unique():
        for well in fitting_df['well'].unique():
            for sweep in fitting_df['sweep'].unique():
                sub_df = fitting_df[(fitting_df['well'] == well)
                                    & (fitting_df[protocol_label] == protocol)].copy()
                sub_df = sub_df[sub_df.sweep == sweep]
                sub_df = sub_df.dropna()
                # Get index of min score
                if len(sub_df.index) == 0:
                    continue
                best_params.append(sub_df[sub_df.score == sub_df.score.min()].head(1).copy())

    if not best_params:
        raise Exception(f"Couldn't find any valid parameters {fitting_df}")

    return pd.concat(best_params, ignore_index=True)


def adjust_kinetics(model_class, params_df, E_rev_df, E_rev, new_E_rev=None):
    # Assume that params_df is a datafram of parameter estimates that were
    # found under the assumption that E_obs = E_Nernst. Then adjust the
    # kinetics parameters such that E_obs = E_Nernst - V_off, instead.

    model = make_model_of_class(model_class)

    transformations = model.transformations
    assert transformations is not None

    param_labels = sorted(model.get_parameter_labels())
    param_pairs = list(zip(param_labels[:-2:2], param_labels[1:-1:2]))

    E_rev_df = E_rev_df.set_index(['protocol', 'well', 'sweep'])

    if model.name == 'WangModel':
        param_pairs = [['a_a0_a', 'a_a0_b'],
                       ['b_a0_a', 'b_a0_b'],
                       ['a_a1_a', 'a_a1_b'],
                       ['b_a1_a', 'b_a1_b'],
                       ['a_1_a', 'a_1_b'],
                       ['b_1_a', 'b_1_b']]

    # Assume that parameters listed like a1, b1, a1, b2, ..., gkr with k_i=a_i e^{b_i V}
    new_rows = []
    for _, row in params_df.iterrows():

        protocol = row['protocol']
        well = row['well']
        sweep = row['sweep']

        inferred_E_rev = E_rev_df.loc[(protocol, well, sweep)]['E_rev']

        if not new_E_rev:
            V_off = inferred_E_rev - E_rev
        else:
            V_off = inferred_E_rev - new_E_rev

        for a, b in zip(param_labels[:-3:2], param_labels[1:-1:2]):
            row[a] = row[a] * np.exp(row[b] * V_off)

        new_rows.append(row)

    new_dict = pd.DataFrame.from_records(new_rows)

    return new_dict


def make_prediction(model_class, args, well, sim_protocol, predict_sweep,
                    protocol_fitted, fitting_sweep, params_df, subtractions_df,
                    fitting_case, E_rev, protocol_dict, full_data, voltages,
                    label='', solver=None, do_spike_removal=True):

    if fitting_case in ['I', 'II']:
        use_artefacts = True
    else:
        use_artefacts = False

    model = make_model_of_class(model_class)

    if use_artefacts:
        model = ArtefactModel(model)

    param_labels = model.get_parameter_labels()

    # Protocol we use for simulation
    desc, full_times = protocol_dict[sim_protocol]

    if solver is None:
        solver= model.make_hybrid_solver_current(hybrid=False,
                                                 njitted=False,
                                                 strict=False,
                                                 protocol_description=desc)

    if do_spike_removal:
        spike_times, spike_indices = markovmodels.voltage_protocols.detect_spikes(full_times, voltages,
                                                                              threshold=10)
        _, _, indices = markovmodels.voltage_protocols.remove_spikes(full_times, voltages, spike_times,
                                                                     time_to_remove=args.removal_duration)
    else:
        indices = np.array([i for i in range(len(full_times))]).astype(int)

    times = full_times[indices]

    if fitting_case in ['I', 'II']:
        forward_sim_parameters = default_artefact_kinetic_parameters.copy()
        param_row = params_df[(params_df.well == well) &
                                (params_df.protocol == sim_protocol) &\
                                (params_df.sweep == predict_sweep)].iloc[0]

        gleak, Eleak, V_off, Rseries, Cm = param_row[['gleak, Eleak, V_off, Rseries, Cm']]
        forward_sim_parameters[-8:] = E_rev, gleak, Eleak, 0, 0, V_off, Rseries, Cm
        artefact_params = forward_sim_parameters[-no_artefact_parameters:]
        model.default_parameters[-3] = V_off

    data = full_data[indices]

    df = params_df[params_df.well == well]
    df = df[(df.protocol == protocol_fitted) & (df.sweep == fitting_sweep)]
    if df.empty:
        return np.full(full_times.shape, None)

    params = df.iloc[0][param_labels].values\
                                        .astype(np.float64)\
                                        .flatten()

    # Set artefact params
    if use_artefacts:
        params[no_artefact_parameters:] = artefact_params

    if fitting_case in ['0a', 'I']:
        pred_E_rev = E_rev
    else:
        inferred_E_rev = subtractions_df.set_index(['protocol', 'well', 'sweep']).loc[(sim_protocol, well, predict_sweep)]['E_rev']
        pred_E_rev = inferred_E_rev

    if fitting_case in ['I', 'II']:
        params[-no_artefact_params] = inferred_E_rev
        return solver(params, times=full_times, protocol_description=desc)
    else:
        return solver(params, times=full_times, protocol_description=desc,
                      E_rev=pred_E_rev)

