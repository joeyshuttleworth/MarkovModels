#!/bin/env python3

import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np

class ChannelModel:
    # Voltage as a function of time
    V = lambda t: 0

    #state probabilities
    ProbabilityOpen = 0
    ProbabilityClosed = 1
    ProbabilityInactive = 0

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

    def getProbabilityICState(self):
        return 1 - self.ProbabilityOpen - self.ProbabilityClosed - self.ProbabilityInactive


    def getDPrClosedDt(self):
        return -(self.k1 + self.k3)*self.ProbabilityClosed + self.k2*self.ProbabilityOpen + self.k4*self.getProbabilityICState()

    def getDPrOpenDt(self):
        return -(self.k2 + self.k3)*self.ProbabilityOpen + self.k1*self.ProbabilityClosed + self.k4 * self.ProbabilityInactive

    def getDPrInactiveDt(self):
        return -(self.k2 + self.k4) * self.ProbabilityInactive + self.k3*self.ProbabilityOpen + self.k1*self.getProbabilityICState()

    def getStates(self, t):
        self.setTransitionRates(t)
        return np.ndarray(shape=(3,), buffer=np.array([self.ProbabilityClosed, self.ProbabilityOpen, self.ProbabilityInactive]))

    def setStates(self,X):
        self.ProbabilityClosed = X[0]
        self.ProbabilityOpen = X[1]
        self.ProbabilityInactive = X[2]

    def getDerivatives(self, t, X):
        self.setStates(X)
        self.setTransitionRates(t)
        IC = self.getProbabilityICState()
        return [1 - self.ProbabilityOpen - self.ProbabilityClosed - IC,  -(self.k2 + self.k3)*self.ProbabilityOpen + self.k1*self.ProbabilityClosed + self.k4 * self.ProbabilityInactive, -(self.k2 + self.k4) * self.ProbabilityInactive + self.k3*self.ProbabilityOpen + IC]


    def getTransitionRates(self, t=0):
        self.setTransitionRates(t)
        return [self.k1, self.k2, self.k3, self.k4]

    def calculateCurrent(self, probO, t=0):
        self.ProbabilityOpen = probO
        return self.getCurrent(self.V(t))

    #Calculate the current through the membrane in nano amps
    def getCurrent(self, V):
        # G is maximal conductance
        G = self.P[-1]

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
        return G * self.ProbabilityOpen*(V - E)

    def getSystemOfOdes(self, time=0):
        ''' Return [A,B] where A is a 3x3 matrix and B is a 3x1 vector
            satisfying the non-homogeneous linear system of odes
            dX/dt = AX + B where X ]
            (ProbabilityClosed, ProbabilityOpen, ProbabilityInactive)^T
            assuming that the voltage remains constant.
        '''
        self.setTransitionRates(time)
        A=np.matrix([[-self.k1 -self.k3 -self.k4, self.k2 - self.k4, -self.k4], [self.k1, -self.k2 - self.k3, self.k4], [-self.k1, self.k3-self.k1, -self.k2 - self.k4 - self.k1]])
        B = np.matrix([self.k4,0,self.k1]).T
        return [A, B]

    def calculateDerivatives(self, times, states):
        [getDerivatives(times[i], states[i]) for i in range(0, len(times))]

def main():
    t = 5000
    params = [2.26E-04, 0.0699, 3.45E-05, 0.05462, 0.0873, 8.92E-03, 5.150E-3, 0.03158, 0.1524]
    amplitudes = [54,26,10]
    frequencies = [0.007,0.037, 0.19]
    V =  lambda t: -30 + sum([amplitudes[i]*np.sin(frequencies[i]*(t-2500)) for i in range(0,3)])
    model = ChannelModel(params, V)
    print(model.getTransitionRates())
    initial_conditions = model.getStates(0)
    print(model.getStates(0))
    solution = integrate.solve_ivp(model.getDerivatives, [0,t], model.getStates(0))

    y = solution.y
    IVec = [model.calculateCurrent(y[1,t]) for t in range(0,len(solution.t))]

    plt.plot(np.linspace(0,5000,1000), V(np.linspace(0,5000,1000)))
    plt.plot(solution.t, solution.y[1])
    plt.plot(solution.t, IVec)

    plt.xlabel("time / ms")
    plt.legend(["V", "[O]", "I / A"])
    plt.show()

if __name__ == '__main__':
    main()
