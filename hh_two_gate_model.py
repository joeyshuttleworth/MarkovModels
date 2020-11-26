#!/bin/env python3

import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np

class TwoGateChannelModel:
    # Voltage as a function of time
    V = lambda t: 0

    #state probabilities
    m = 0
    h = 0

    #Parameters. Maximal conductance is P[8].
    P = [0] * 9

    #transition rates
    k1 = 0
    k2 = 0
    k3 = 0
    k4 = 0

    #Constructor to set Parameters values
    def __init__(self, params, voltage_function = lambda x: 0):
        self.P = params
        self.V = voltage_function

    def setTransitionRates(self, t):
        V = self.V(t)
        self.k1 = self.P[0] * np.exp(self.P[1]*V)
        self.k2 = self.P[2] * np.exp(-self.P[3]*V)
        self.k3 = self.P[4] * np.exp(self.P[5]*V)
        self.k4 = self.P[6] * np.exp(-self.P[7] * V)

    def getDerivatives(self, t, X):
        self.setTransitionRates(t)
        [k1, k2, k3, k4] = self.getTransitionRates(t)
        return [(1 - X[0]) * k1 - k2 * X[0], (1 - X[1])*k3 - X[1]*k4]

    def getTransitionRates(self, t=0):
        self.setTransitionRates(t)
        return [self.k1, self.k2, self.k3, self.k4]

    def calculateDerivatives(self, times, states):
        return [getDerivatives(times[i], states[i]) for i in range(0, len(times))]

    def calculateCurrent(self, t, ProbOpen):
        # G is maximal conductance
        G = self.P[-1]

        V = self.V(t)

        # E is the Nernst potential for potassium ions across the membrane
        # Gas constant R, temperature T, Faradays constat F
        R = 8314.55
        T = 293
        F = 96485

        # Intracellular and extracellular concentrations of potassium.
        K_out = 4
        K_in  = 130

        # valency of ions (1 in the case of K^+)
        z = 1


        #Nernst potential
        E = R*T/(z*F) * np.log(K_out/K_in)
        return G * ProbOpen *(V - E)

def main():
    t = 5000
    params = [2.26E-04, 0.0699, 3.45E-05, 0.05462, 0.0873, 8.92E-03, 5.150E-3, 0.03158, 0.1524]
    amplitudes = [54,26,10]
    frequencies = [0.007,0.037, 0.19]
    V =  lambda t: -30 + sum([amplitudes[i]*np.sin(frequencies[i]*(t-2500)) for i in range(0,3)])
    model = TwoGateChannelModel(params, V)
    print(model.getTransitionRates())
    solution = integrate.solve_ivp(model.getDerivatives, [0,t], [0,0])

    y = solution.y
    IVec = [y[0,t]*(1 - y[1,t]) for t in range(0,len(solution.t))]

    plt.plot(np.linspace(0,5000,1000), V(np.linspace(0,5000,1000)))
    plt.plot(solution.t, solution.y[1])
    plt.plot(solution.t, IVec)

    plt.xlabel("time / ms")
    plt.legend(["V", "[O]", "I / A"])
    plt.show()

if __name__ == '__main__':
    main()
