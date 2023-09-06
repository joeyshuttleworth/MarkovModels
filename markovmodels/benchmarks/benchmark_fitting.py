#!/usr/bin/env python3

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

def main():
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
    times = np.arange(0, t_max, 0.1)

    global Erev
    Erev = calculate_reversal_potential(310.15)

    # Set up protocol
    mk_protocol = myokit.load_protocol('simplified-staircase.mmt')

    protocol = []
    for event in mk_protocol_events():
        duration = event.duration()
        end_t = mk.characteristic_time()
        start_t = end_t - duration
        level = event.level()
        protocol.append((start_t, end_t, level, level))

    @njit
    def protocol_func(t):
        if t < 0 or t >= protocol[-1][1]:
            return holding_potential

        for i in range(len(protocol)):
            if t <= protocol[i][1]:
                if np.abs(protocol[i][3] - protocol[i][2]) > threshold:
                    return protocol[i][2] + (t - protocol[i][0])*(protocol[i][3]-protocol[i][2])/(protocol[i][1] - protocol[i][0])
                else:
                    return protocol[i][3]

    for step in protocol:
        mk_protocol.add_step()

    voltages = np.array([protocol_func(t) for t in times])

    model = BeattieModel(protocol_func, times, Erev=Erev)
    mk_solver = get_myokit_solver()
    mms_solver_func = model.make_forward_solver_states()

    @njit
    def mms_solver(p):
        return mms_solver_func(p, times, len(times), voltages)

    parameter_sets = np.random.multivariate_normal(mean_params, params_cov, 1000)

    print("Running myokit simulations")
    cProfile.run(simulate_samples(samples, mk_solver))

    print("Running markovmodels simulations")
    cProfile.run(simulate_samples(samples, solver))

def get_myokit_solver(mk_protocol):
    model = myokit.load_model('beattie_model.mmt')
    sim = mk.Simulation(model, mk_protocol)

    sim.set_constant('Erev', Erev)

    end_time = mk_protocol.characteristic_time()

    def solver(parameters):
        # TODO: set parameters to steady state
        sim.reset()
        for i, p in enumerate(parameters):
            self.sim.set_constant('markov_chain.p' + str(i), p)
        log = self.sim.run(tmax, log_times=times, log=['membrane.current'])
        return log
    return solver

def simulate_samples(samples, solver):
    for sample in samples:
        solver(sample)


if __name__ == "__main__":
    main()
