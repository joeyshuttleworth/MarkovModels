#!/usr/bin/env python3

from MarkovModels.BeattieModel import BeattieModel
from MarkovModels import common
import argparse
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import pandas as pd
import myokit as mk
import cProfile
from numba import njit
import sympy as sp

this_dir = os.path.dirname(os.path.realpath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_simulations', '-n', type=int, default=500)
    parser.add_argument('--output', '-o', type=str, default="output")
    parser.add_argument('--tolerance_plot', '-t', action='store_true', default=False)
    parser.add_argument('--dt', default=0.1, type=float)

    args = parser.parse_args()

    print(f"Using a timestep of {args.dt:.2f}ms, running {args.no_simulations} simulations")

    output_dir = os.path.join(args.output, "benchmark_fitting")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mean_params = np.array([2.07E-3, 7.17E-2, 3.44E-5, 6.18E-2, 4.18E-1, 2.58E-2,
                            4.75E-2, 2.51E-2, 3.33E-2])

    params_cov = np.array([[5.09324351e-12, 1.06985687e-10, -1.34170447e-14, 4.12063294e-12,
                            7.93019250e-11, -1.53579401e-12, -5.91328798e-12, -1.80148798e-12,
                            3.58550560e-12, ],
                           [1.06985687e-10, 3.35858728e-09, -4.56663043e-13, 1.19997420e-10,
                            9.78999959e-10, -1.28159569e-12, -1.20704368e-10, -1.46695722e-11,
                            4.12117387e-11],
                           [-1.34170447e-14, -4.56663043e-13, 2.08508221e-15, -7.71053807e-13,
                            -3.21866748e-12, 2.54720193e-13, -3.26388833e-13, -5.12107040e-14,
                            3.49054070e-14],
                           [4.12063294e-12, 1.19997420e-10, -7.71053807e-13, 3.60968781e-10,
                            1.14969564e-09, -9.14683023e-11, -1.20206502e-10, -2.24064950e-11,
                            1.34409986e-10],
                           [7.93019250e-11, 9.78999959e-10, -3.21866748e-12, 1.14969564e-09,
                            2.44584725e-06, 6.20608480e-08, 2.75070933e-07, -6.34168364e-08,
                            1.58426566e-09],
                           [-1.53579401e-12, -1.28159569e-12, 2.54720193e-13, -9.14683023e-11,
                            6.20608480e-08, 3.57801616e-09, 7.02365678e-09, -3.48983103e-09,
                            1.05810627e-11],
                           [-5.91328798e-12, -1.20704368e-10, -3.26388833e-13, -1.20206502e-10,
                            2.75070933e-07, 7.02365678e-09, 3.29041280e-08, -6.77904303e-09,
                            -9.66666120e-10],
                           [-1.80148798e-12, -1.46695722e-11, -5.12107040e-14, -2.24064950e-11,
                            -6.34168364e-08, -3.48983103e-09, -6.77904303e-09, 3.61024335e-09,
                            -2.79099415e-10],
                           [3.58550560e-12, 4.12117387e-11, 3.49054070e-14, 1.34409986e-10,
                            1.58426566e-09, 1.05810627e-11, -9.66666120e-10, -2.79099415e-10,
                            7.07475893e-10]])

    t_max = 15400
    times = np.linspace(0, t_max, int(t_max / args.dt))
    print(f"Evaluating at {len(times)} timesteps")

    global Erev
    Erev = common.calculate_reversal_potential(310.15)

    # Set up protocol
    mk_protocol = mk.load_protocol(os.path.join(this_dir, 'MarkovModels', 'benchmarks', 'resources', 'simplified-staircase.mmt'))

    protocol = []
    for event in mk_protocol.events():
        duration = event.duration()
        end_t = event.characteristic_time()
        start_t = end_t - duration
        level = event.level()
        protocol.append((start_t, end_t, level, level))

    protocol = tuple(protocol)
    holding_potential = -80

    @njit
    def protocol_func(t):
        if t < 0 or t >= protocol[-1][1]:
            return holding_potential

        for i in range(len(protocol)):
            if t <= protocol[i][1]:
                if np.abs(protocol[i][3] - protocol[i][2]) > 0.0001:
                    return protocol[i][2] + (t - protocol[i][0]) * (protocol[i][3] - protocol[i][2]) / (protocol[i][1] - protocol[i][0])
                else:
                    return protocol[i][2]

    voltages = np.array([protocol_func(t) for t in times])

    model = BeattieModel(protocol_func, times, Erev=Erev)
    model.window_locs = [tstart for tstart, _, _, _ in protocol]
    model.protocol_description = protocol

    # Do tolerance plots
    if args.tolerance_plot:
        make_tolerance_plot(model, mk_protocol, mean_params, protocol, output_dir)

    mk_solver = get_mk_solver(mk_protocol, times, atol=model.solver_tolerances[0], rtol=model.solver_tolerances[1])

    inputs = list(model.y) + list(model.p) + [model.v]
    frhs = np.array([e for e in model.rhs_expr])

    rhs = njit(sp.lambdify(inputs, frhs))

    @njit
    def crhs(t, y, dy, p):
        res = rhs(y[0], y[1], y[2],
                  p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8],
                  protocol_func(t))
        for i in range(3):
            dy[i] = res[i]

    Erev = model.Erev

    atol = model.solver_tolerances[0]
    rtol = model.solver_tolerances[1]

    mms_solver_func = model.make_forward_solver_current()
    mms_ida_solver_func = model.make_solver_current(model.make_dae_solver_states())

    @njit
    def mms_solver(p):
        return mms_solver_func(p, times, voltages=voltages, atol=atol, rtol=rtol)

    hybrid_solve = model.make_hybrid_solver_current()

    @njit
    def hybrid_solver(p):
        return hybrid_solve(p, times, atol, rtol)


    # Use less tolerance
    @njit
    def ida_solver(p):
        return hybrid_solve(p, times, 1e-5, rtol)

    samples = np.random.multivariate_normal(mean_params, params_cov, args.no_simulations)

    n_subsamples = int(args.no_simulations / 10) + 1
    subsamples = samples[0:n_subsamples + 1, :]

    mms_res = simulate_samples(subsamples, mms_solver)
    mk_res = simulate_samples(subsamples, mk_solver)
    hybrid_res = simulate_samples(subsamples, hybrid_solver)
    dae_res = simulate_samples(subsamples, ida_solver)

    print("compiled hybrid solver")

    fig = plt.figure(figsize=(16, 14))
    ax = fig.subplots()
    ax.plot(times, np.array(mk_res[0]['membrane.I_Kr']), label="myokit solution")
    ax.plot(times, mms_res[0], label="MarkovModels LSODA solution")
    ax.plot(times, hybrid_res[0], label="hybrid solver solution")
    ax.plot(times, dae_res[0], label="IDA solver solution")
    ax.legend()
    fig.savefig(os.path.join('benchmark_solver_comparison'))
    ax.cla()
    plt.close(fig)

    fig = plt.figure(figsize=(16, 14))
    ax = fig.subplots()

    # compare voltages
    ax.plot(mk_res[0]['engine.time'], np.array(mk_res[0]['membrane.V']))
    ax.plot(times, voltages)
    fig.savefig(os.path.join(output_dir, 'voltage_comparison'))
    ax.cla()

    mk_errors = np.array([((res2 - res1['membrane.I_Kr'])**2).sum() for res1, res2 in zip(mk_res, hybrid_res)])
    mms_errors = np.array([((res2 - res1)**2).sum() for res1, res2 in zip(mms_res, hybrid_res)])
    errors_df = pd.DataFrame(np.column_stack((subsamples, mk_errors[:, None], mms_errors[:, None])), columns=[
                             f"p{i+1}" for i in range(model.get_no_parameters())] + ['Myokit SSE', "MarkovModels SSE"])

    arg_max_error = np.argmax(mms_errors)
    plt.plot(times, hybrid_solver(subsamples[arg_max_error]), label="hybrid")
    plt.plot(times, mms_solver(subsamples[arg_max_error]), label="mms")
    plt.plot(times, mk_solver(subsamples[arg_max_error])['membrane.I_Kr'], label="myokit")
    plt.plot(times, ida_solver(subsamples[arg_max_error]), label="NumbaIDA")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "biggest_mms_error"))

    logging.info(errors_df)

    errors_df.to_csv(os.path.join(output_dir, 'benchmark_errors.csv'))

    print("Running mk simulations")
    global func1

    def func1():
        return simulate_samples(samples, mk_solver)
    cProfile.run('func1()')

    print("Running MarkovModels simulations")

    global func2

    def func2():
        return simulate_samples(samples, mms_solver)
    cProfile.run('func2()')

    global func3

    def func3():
        return simulate_samples(samples, hybrid_solver)
    cProfile.run('func3()')

    global func4

    def func4():
        return simulate_samples(samples, mms_ida_solver_func)
    cProfile.run('func4()')


def get_mk_solver(mk_protocol, times, atol, rtol):
    model = mk.load_model(os.path.join(this_dir, 'MarkovModels', 'benchmarks', 'beattie_model.mmt'))
    sim = mk.Simulation(model, mk_protocol)

    sim.set_constant('membrane.Erev', Erev)

    end_time = int(mk_protocol.characteristic_time())+1

    sim.set_tolerance(atol, rtol)

    # set states to steady state
    sim.pre(20000)
    model.set_state(sim.state())

    sim = mk.Simulation(model, mk_protocol)
    sim.set_constant('membrane.Erev', Erev)

    def solver(parameters):
        # TODO: set parameters to steady state
        sim.reset()
        for i, p in enumerate(parameters):
            sim.set_constant('markov_chain.p' + str(i), p)
        log = sim.run(end_time, log_times=times, log=['membrane.I_Kr',
                      'engine.time', 'membrane.V', 'engine.evaluations'])
        return log
    return solver


def simulate_samples(samples, solver):
    return [solver(sample) for sample in samples]


def make_tolerance_plot(model, mk_protocol, mean_params, protocol, output_dir):
    tolerances = [(1E-3, 1E-5), (1E-5, 1E-7), (1E-7, 1E-9), (1E-9, 1E-12)]
    times = model.times[::100]
    voltages = model.GetVoltage()[::100]
    fig = plt.figure(figsize=(16, 14))
    axs = fig.subplots(len(tolerances))

    mms_solver_func = model.make_forward_solver_current()

    for ax, (abs_tol, rel_tol) in zip(axs, tolerances):
        mk_res = get_mk_solver(mk_protocol, times, abs_tol, rel_tol)(mean_params)
        mms_res = mms_solver_func(mean_params, times, voltages, abs_tol, rel_tol)
        hybrid_res = model.make_hybrid_solver_current(protocol)(mean_params, times, voltages)

        ax.plot(times, np.log(np.abs(np.array(mk_res['membrane.I_Kr']) - hybrid_res)), label="myokit errors")
        ax.plot(times, np.log(np.abs(mms_res - hybrid_res)), label="LSODA errors")
        ax.legend()
        ax.set_title(f"abs_tol = {abs_tol}, rel_tol = {rel_tol}")
        ax.set_ylabel("log absolute errors ")

    axs[-1].set_xlabel("time /ms")
    model.set_tolerances(1e-3, 1e-5)

    fig.savefig(os.path.join(output_dir, 'benchmark_solver_errors'))
    plt.close(fig)

if __name__ == "__main__":
    main()
