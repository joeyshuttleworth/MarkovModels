from BeattieModel import BeattieModel
import argparse
import os
import common
import numpy as np
import logging
import matplotlib.pyplot as plt
import regex as re
import pandas as pd
import common
import myokit as mk
import cProfile
from BeattieModel import BeattieModel
from numba import njit, cfunc
from NumbaLSODA import lsoda_sig
import sympy as sp

this_dir = os.path.dirname(os.path.realpath(__file__))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_simulations', '-n', type=int, default=500)

    args = parser.parse_args()

    mean_params = np.array([2.07E-3, 7.17E-2, 3.44E-5, 6.18E-2, 4.18E-1, 2.58E-2,
                       4.75E-2, 2.51E-2, 3.33E-2])

    params_cov = np.array([[ 5.09324351e-12,  1.06985687e-10, -1.34170447e-14,  4.12063294e-12,
                             7.93019250e-11, -1.53579401e-12, -5.91328798e-12, -1.80148798e-12,
                             3.58550560e-12,],
                           [ 1.06985687e-10,  3.35858728e-09, -4.56663043e-13,  1.19997420e-10,
                             9.78999959e-10, -1.28159569e-12, -1.20704368e-10, -1.46695722e-11,
                             4.12117387e-11],
                           [-1.34170447e-14, -4.56663043e-13,  2.08508221e-15, -7.71053807e-13,
                            -3.21866748e-12,  2.54720193e-13, -3.26388833e-13, -5.12107040e-14,
                            3.49054070e-14],
                           [ 4.12063294e-12,  1.19997420e-10, -7.71053807e-13,  3.60968781e-10,
                             1.14969564e-09, -9.14683023e-11, -1.20206502e-10, -2.24064950e-11,
                             1.34409986e-10],
                           [ 7.93019250e-11,  9.78999959e-10, -3.21866748e-12,  1.14969564e-09,
                             2.44584725e-06,  6.20608480e-08,  2.75070933e-07, -6.34168364e-08,
                             1.58426566e-09],
                           [-1.53579401e-12, -1.28159569e-12,  2.54720193e-13, -9.14683023e-11,
                            6.20608480e-08,  3.57801616e-09,  7.02365678e-09, -3.48983103e-09,
                            1.05810627e-11],
                           [-5.91328798e-12, -1.20704368e-10, -3.26388833e-13, -1.20206502e-10,
                            2.75070933e-07,  7.02365678e-09,  3.29041280e-08, -6.77904303e-09,
                            -9.66666120e-10],
                           [-1.80148798e-12, -1.46695722e-11, -5.12107040e-14, -2.24064950e-11,
                            -6.34168364e-08, -3.48983103e-09, -6.77904303e-09,  3.61024335e-09,
                            -2.79099415e-10],
                           [ 3.58550560e-12,  4.12117387e-11,  3.49054070e-14,  1.34409986e-10,
                             1.58426566e-09,  1.05810627e-11, -9.66666120e-10, -2.79099415e-10,
                             7.07475893e-10]])

    t_max = 15400
    times = np.linspace(0, t_max, t_max*10)

    global Erev
    Erev = common.calculate_reversal_potential(310.15)

    # Set up protocol
    mk_protocol = mk.load_protocol(os.path.join(this_dir, 'benchmarks', 'resources', 'simplified-staircase.mmt'))

    protocol = []
    for event in mk_protocol.events():
        duration = event.duration()
        end_t = event.characteristic_time()
        start_t = end_t - duration
        level = event.level()
        protocol.append((start_t, end_t, level, level))

    protocol = tuple(protocol)
    holding_potential = mk_protocol.events()[-1].level()

    @njit
    def protocol_func(t):
        if t < 0 or t >= protocol[-1][1]:
            return holding_potential

        for i in range(len(protocol)):
            if t <= protocol[i][1]:
                if np.abs(protocol[i][3] - protocol[i][2]) > 0.0001:
                    return protocol[i][2] + (t - protocol[i][0])*(protocol[i][3]-protocol[i][2])/(protocol[i][1] - protocol[i][0])
                else:
                    return protocol[i][3]

    for step in protocol:
        mk_protocol.add_step(step[-1], step[1]-step[0])

    voltages = np.array([protocol_func(t) for t in times])

    model = BeattieModel(protocol_func, times, Erev=Erev)

    mk_solver = get_mk_solver(mk_protocol, times, atol=model.solver_tolerances[0], rtol=model.solver_tolerances[1])
    mms_solver_func = model.make_forward_solver_current()
    mms_solver_func_states = model.make_forward_solver_states()

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

    @njit
    def mms_solver(p):
        sol = mms_solver_func_states(p, times, len(times))

        dy = np.array([0, 0, 0])
        # for i in range(len(times)):
        #     crhs(times[i], sol[i,:], dy, p)
        return p[-1]*sol[:,1]*(voltages - Erev)

    hybrid_solve = model.make_hybrid_solver_current(protocol)
    @njit
    def hybrid_solver(p):
        return hybrid_solve(p, times, voltages)

    samples = np.random.multivariate_normal(mean_params, params_cov, args.no_simulations)

    subsamples = samples[0:100,:]
    mms_res = simulate_samples(subsamples, mms_solver)
    mk_res  = simulate_samples(subsamples, mk_solver)
    hybrid_res  = simulate_samples(subsamples, hybrid_solver)

    print("compiled hybrid solver")

    fig = plt.figure(figsize=(16,14))
    ax = fig.subplots()
    ax.plot(times, np.array(mk_res[0]['membrane.I_Kr']), label="myokit solution")
    ax.plot(times, mms_res[0], label="numerical solver solution")
    ax.plot(times, hybrid_res[0], label="hybrid solver solution")
    fig.savefig('benchmark_solver_comparison')
    ax.legend()
    ax.cla()

    # compare voltages
    ax.plot(mk_res[0]['engine.time'], np.array(mk_res[0]['membrane.V']))
    ax.plot(times, voltages)
    fig.savefig('voltage_comparison')
    ax.cla()


    mk_errors = np.array([((res2-res1['membrane.I_Kr'])**2).sum() for res1, res2 in zip(mk_res, hybrid_res)])
    mms_errors = np.array([((res2-res1)**2).sum() for res1, res2 in zip(mms_res, hybrid_res)])
    errors_df = pd.DataFrame(np.column_stack((subsamples, mk_errors[:,None], mms_errors[:,None])), columns=[f"p{i+1}" for i in range(model.get_no_parameters())] + ['Myokit SSE', "MarkovModels SSE"])

    arg_max_error = np.argmax(mms_errors)
    plt.plot(times, hybrid_solver(subsamples[arg_max_error]), label="hybrid")
    plt.plot(times, mms_solver(subsamples[arg_max_error]), label="mms")
    plt.legend()
    plt.show()

    print(errors_df)

    errors_df.to_csv('benchmark_errors')

    print("Running mk simulations")
    global func1
    func1 = lambda : simulate_samples(samples, mk_solver)
    cProfile.run('func1()')

    print("Running MarkovModels simulations")
    global func2
    func2 = lambda : simulate_samples(samples, mms_solver)
    cProfile.run('func2()')

    global func3
    func3 = lambda : simulate_samples(samples, hybrid_solver)
    cProfile.run('func3()')

def get_mk_solver(mk_protocol, times, atol, rtol):
    model = mk.load_model(os.path.join(this_dir, 'benchmarks', 'beattie_model.mmt'))
    sim = mk.Simulation(model, mk_protocol)

    sim.set_constant('membrane.Erev', Erev)

    end_time = mk_protocol.characteristic_time()
    sim.set_tolerance(atol, rtol)

    # set states to steady state
    log=sim.run(20000)

    states = ["C", "s_O", "s_I"]

    for i in range(len(states)):
        state = f"markov_chain.{states[i]}"
        model.get(state).set_state_value(log[state][-1])

    sim = mk.Simulation(model, mk_protocol)
    sim.set_tolerance(atol, rtol)

    def solver(parameters):
        # TODO: set parameters to steady state
        sim.reset()
        for i, p in enumerate(parameters):
            sim.set_constant('markov_chain.p' + str(i), p)
            sim.set_constant('membrane.Erev', Erev)
        log = sim.run(end_time, log_times=times, log=['membrane.I_Kr', 'engine.time', 'membrane.V', 'engine.evaluations'])
        return log
    return solver

def simulate_samples(samples, solver):
    return [solver(sample) for sample in samples]

if __name__ == "__main__":
    main()
