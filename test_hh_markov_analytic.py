from hh_markov_model import ChannelModel
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import scipy.integrate as integrate

def main():
    #Setup the model with an example parameter set and constant voltage
    # params = [1, 1, 1, 1, 1, 1, 1, 0.1524]
    params = [2.26E-04, 0.0699, 3.45E-05, 0.05462, 0.0873, 8.92E-03, 5.150E-3, 0.03158, 0.1524]
    model = ChannelModel(params, lambda t: -80)

    #Get the transition rates of the model
    [k1, k2, k3, k4] = model.getTransitionRates()

    #Use the initial conditions to set C1 and C2 assuming x1,x2 = 0 (equivalently, [C] = 1).
    C1 = -k1/(k1+k2)
    C2 = -k3/(k3+k4)

    # Calculate the current at time t analytically
    t = 5
    x1 = C1*np.exp(t*(-k1-k2)) + k1/(k1+k2)
    x2 = C2*np.exp(t*(-k3-k4)) + k3/(k3+k4)

    analyticI = model.calculateCurrent((1-x1)*(1-x2), x1*(1-x2), x1*x2)

    # Run the model numerically
    numerical_model = ChannelModel(params, lambda t : -80)
    solution = integrate.solve_ivp(model.getDerivatives, [0,t], model.getStates(0))
    y = solution.y

    #Get the current for the final timestep (at t)
    numericalI = model.calculateCurrent(y[0,-1], y[1, -1], y[2,-1])

    print(numericalI, analyticI)
    assert((numericalI - analyticI)/analyticI < 0.1)


if __name__ == '__main__':
    main()
