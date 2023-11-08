#!/usr/bin/env python3

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
from numba import njit

import markovmodels

from markovmodels.model_generation import make_model_of_class
from markovmodels.voltage_protocols import get_ramp_protocol_from_csv
from markovmodels.utilities import get_data
from markovmodels.voltage_protocols import remove_spikes
from markovmodels.ArtefactModel import ArtefactModel


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
        # Assume that the conductance is the last parameter and that the parameters are arranged included

        if mm.transformations:
            transformations = [t for i, t in enumerate(mm.transformations)
                               if (i % len(starting_parameters)) not in fix_parameters]
            transformation = pints.ComposedTransformation(*transformations)

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

            transformations = [t for i, t in enumerate(transformations) if i not in fix_parameters]
            transformation = pints.ComposedTransformation(*transformations)

        else:
            raise Exception("Couldn't log transform parameters")

    else:
        transformation = None

    if starting_parameters is None:
        starting_parameters = mm.get_default_parameters()

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
                    try:
                        return solver(p)[subset_indices]
                    except Exception:
                        return np.full(times.shape, np.inf)

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

    boundaries = fitting_boundaries(starting_parameters, mm,
                                    fix_parameters, is_artefact_model=use_artefact_model)

    if randomise_initial_guess:
        initial_guess_dist = fitting_boundaries(starting_parameters, mm,
                                                fix_parameters, is_artefact_model=use_artefact_model)
        starting_parameter_sets = []

    scores, parameter_sets, iterations, times_taken = [], [], [], []
    for i in range(repeats):
        if randomise_initial_guess:
            initial_guess = initial_guess_dist.sample(n=1).flatten()
            starting_parameter_sets.append(initial_guess)
            params_not_fixed = initial_guess

        if not boundaries.check(params_not_fixed):
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
        best_parameters = mm.get_default_parameters()
        best_score = np.inf

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        point2 = [p for i, p in enumerate(mm.get_default_parameters()) if i not
                  in fix_parameters]
        fig, axes = pints.plot.function_between_points(error,
                                                       point_1=best_parameters,
                                                       point_2=point2,
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
        fitting_df['RMSE'] = scores
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
                  fix_parameters=[], data_label=None):

    if default_parameters is None or len(default_parameters) == 0:
        default_parameters = make_model_of_class(model_class_name).get_default_parameters()

    parameter_labels = make_model_of_class(model_class_name).get_parameter_labels()

    if max_iterations == 0:
        df = pd.DataFrame(default_parameters[None, :], columns=parameter_labels)
        df['score'] = np.inf
        return df

    # Ignore files that have been commented out
    voltage_func, times, protocol_desc = get_ramp_protocol_from_csv(protocol)

    data = get_data(well, protocol, data_directory, experiment_name,
                    label=data_label, sweep=sweep)

    times = pd.read_csv(os.path.join(data_directory, f"{experiment_name}-{protocol}-times.csv"),
                        float_precision='round_trip')['time'].values

    voltages = np.array([voltage_func(t) for t in times])
    spike_times, _ = detect_spikes(times, voltages, window_size=0)
    _, _, indices = remove_spikes(times, voltages, spike_times,
                                  removal_duration)

    if infer_E_rev:
        try:
            if output_dir:
                plot = True
                output_path = os.path.join(output_dir, 'infer_reversal_potential.png')

            if use_artefact_model:
                # Use the artefact to forward simulate the voltages (using literature kinetics)
                model = make_model_of_class(model_class_name, voltage=voltage_func,
                                            times=times,
                                            E_rev=E_rev,
                                            protocol_description=protocol_desc)

                model = ArtefactModel(model)

                params_for_Erev = default_parameters.copy()
                no_kinetic_params = len(parameter_labels) - 1
                params_for_Erev[:no_kinetic_params] = artefact_default_kinetic_parameters[:no_kinetic_params]

                E_obs = infer_reversal_potential_with_artefact(protocol, times, data,
                                                               'model3',
                                                               params_for_Erev,
                                                               E_rev,
                                                               removal_duration=removal_duration,
                                                               plot=True,
                                                               output_path=output_dir,
                                                               forward_sim_output_dir=output_dir,
                                                               )

            else:
                voltages = None
                E_obs = infer_reversal_potential(protocol, data, times, plot=plot,
                                                 output_path=output_path, voltages=voltages)
            if use_artefact_model:
                inferred_E_rev = E_rev
                V_off = E_rev - E_obs
                default_parameters[-3] = V_off
            else:
                inferred_E_rev = E_obs

            if inferred_E_rev < -50 or inferred_E_rev > -100:
                E_rev = inferred_E_rev

        except None as exc:
            print(str(exc))
            pass

    model = make_model_of_class(model_class_name, voltage=voltage_func,
                                times=times,
                                E_rev=E_rev,
                                protocol_description=protocol_desc)

    if use_artefact_model:
        model = ArtefactModel(model)

    if default_parameters is not None:
        initial_params = default_parameters.copy()
    else:
        model.default_parameters = default_parameters.copy()

    columns = model.get_parameter_labels()

    if infer_E_rev:
        columns.append("E_rev")

    if solver is not None and solver_type is not None:
        raise Exception('solver and solver type provided')

    if solver is None:
        try:
            solver = model.make_forward_solver_of_type(solver_type)
            solver()
        except numba.core.errors.TypingError as exc:
            logging.warning(f"unable to make nopython forward solver {str(exc)}")
            solver = model.make_forward_solver_of_type(solver_type, njitted=False)

    if not np.all(np.isfinite(solver())):
        raise Exception('Default parameters gave non-finite output')

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
    def __init__(self, full_parameters, model, fix_parameters=[], is_artefact_model=False):
        self.is_artefact_model = is_artefact_model

        if is_artefact_model:
            self.mm = model.channel_model
            self.fix_parameters = [i for i in fix_parameters if ((i % len(full_parameters)) < self.mm.get_no_parameters())]
            self.full_parameters = full_parameters[:self.mm.get_no_parameters()].copy()

        else:
            self.fix_parameters = fix_parameters
            self.full_parameters = full_parameters
            self.mm = model

    def check(self, parameters):
        parameters = parameters.copy()
        if len(self.fix_parameters) != 0:
            for i in np.unique(self.fix_parameters):
                parameters = np.insert(parameters, i, self.full_parameters[i])

        # rates function
        rates_func = self.mm.get_rates_func(njitted=False)

        Vs = [-120, 60]
        rates_1 = rates_func(parameters, Vs[0])
        rates_2 = rates_func(parameters, Vs[1])

        max_transition_rates = np.max(np.vstack([rates_1, rates_2]), axis=0)

        if not np.all(max_transition_rates < 1e3):
            return False

        if not np.all(max_transition_rates > 1.67e-5):
            return False

        if max([p for i, p in enumerate(parameters) if i != self.mm.GKr_index]) > 1e5:
            return False

        if min([p for i, p in enumerate(parameters) if i != self.mm.GKr_index]) < 1e-7:
            return False

        # Ensure that all parameters > 0
        return np.all(parameters > 0)

    def n_parameters(self):
        return self.mm.get_no_parameters() - \
            len(self.fix_parameters) if len(self.fix_parameters) != 0 \
            else self.mm.get_no_parameters()

    def _sample_once(self, min_log_p, max_log_p):
        for i in range(1000):
            p = np.empty(self.full_parameters.shape)
            p[-1] = self.full_parameters[-1]
            p[:-1] = 10**np.random.uniform(min_log_p, max_log_p,
                                           self.full_parameters[:-1].shape)

            if len(self.fix_parameters) != 0:
                p = p[[i for i in range(len(self.full_parameters)) if i not in
                       self.fix_parameters]]
            # Check this lies in boundaries
            if self.check(p):
                return p
        logging.warning("Couldn't sample from boundaries")
        return np.NaN

    def sample(self, n=1):
        min_log_p, max_log_p = [-7, 1]

        ret_vec = np.full((n, len(self.full_parameters)), np.nan)
        for i in range(n):
            ret_vec[i, :] = self._sample_once(min_log_p, max_log_p)

        params_not_fixed = [i for i in range(len(self.mm.get_default_parameters()))\
                            if i not in self.fix_parameters]

        ret_vec = ret_vec[:, params_not_fixed]
        return ret_vec


def infer_reversal_potential_with_artefact(protocol, times, data,
                                           model_class_name,
                                           default_parameters, E_rev,
                                           forward_sim_output_dir,
                                           removal_duration=0, **kwargs):
    voltage_func, _, protocol_desc = get_ramp_protocol_from_csv(protocol)
    voltages = np.array([voltage_func(t) for t in times])
    spike_times, _ = detect_spikes(times, voltages, window_size=0)
    _, _, indices = remove_spikes(times, voltages, spike_times,
                                  removal_duration)

    model = make_model_of_class(model_class_name, voltage=voltage_func,
                                times=times,
                                E_rev=E_rev,
                                protocol_description=protocol_desc)

    model = ArtefactModel(model)
    forward_sim_params = default_parameters.copy()

    # Set V_off to 0
    forward_sim_params[-3] = 0
    a_solver = model.make_hybrid_solver_current(njitted=False, hybrid=False)

    # Rough estimate of conductance
    # Maybe make it so this only looks at the reversal ramp, or leak step
    def find_conductance_func(g):
        p = forward_sim_params.copy()
        p[-8] = g
        return np.sum((a_solver(p)[indices] - data[indices])**2)

    res = scipy.optimize.minimize_scalar(find_conductance_func,
                                         bounds=(data[indices].max()*1e-4, data[indices].max()*1e1))
    found_conductance = res.x

    # set conductance parameter
    forward_sim_params[-8] = found_conductance

    fig = plt.figure()
    axs = fig.subplots(2)

    normalised_current = a_solver(forward_sim_params)   # plot current
    axs[0].plot(times, data, color='grey', alpha=.5)
    axs[0].plot(times, normalised_current)

    a_state_solver = model.make_hybrid_solver_states(hybrid=False, njitted=False)
    # plot Vm
    axs[1].plot(times, a_state_solver(forward_sim_params)[:, -1])

    if not os.path.exists(forward_sim_output_dir):
        os.makedirs(forward_sim_output_dir)

    fig.savefig(os.path.join(forward_sim_output_dir,
                             "infer_reversal_potential_forward_sim"))
    plt.close(fig)

    voltages = model.make_hybrid_solver_states(hybrid=False)(forward_sim_params)[:, -1]

    return infer_reversal_potential(protocol, data, times, voltages=voltages,
                                    **kwargs)


def infer_reversal_potential(protocol: str, current: np.array, times, ax=None,
                             output_path=None, plot=None, known_Erev=None, voltages=None):

    if output_path:
        dirname = os.path.dirname(output_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    if (ax or output_path) and plot is not False:
        plot = True

    protocol_func, _, protocol_desc = get_ramp_protocol_from_csv(protocol)

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

