from hh_markov_model import ChannelModel
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import scipy.integrate as integrate

def main():
    #Setup the model with an example parameter set and constant voltage
    params = [1, 1, 1, 1, 1, 1, 1, 0.1524]
    model = ChannelModel(params, lambda t: -80)

    #Get the transition rates of the model
    [k1, k2, k3, k4] = model.getTransitionRates()

    print(k1, k4)

    A =np.matrix([[-k1 -k3 -k4, k2 - k4, -k4], [k1, -k2 - k3, k4], [-k1, k3-k1, -k2 - k4 - k1]])

    print("Det(A)", np.linalg.det(A))
    print(A)
    eigenvalues, C = np.linalg.eig(A)
    D = np.diag(eigenvalues)
    print(D)

    # Non-homogeneous solution
    X2 = -np.linalg.inv(A).dot(np.matrix([k4,0,k1]).T)
    print("X2", X2)

    #Calculate the constant vector of constants, K, using the initial conditions
    K = np.linalg.inv(C)*np.matrix(model.getStates(0)).T - np.linalg.inv(C) * X2
    print("K ", K)

    t = 100

    #Calculate the value at time t
    z = np.matrix([K[i,0] * np.exp(t*eigenvalues[i]) for i in range(0,3)]).T
    X = C.dot(z)
    print(X)
    X = X + X2
    print("X {}".format(X))

    analyticI = model.calculateCurrent(X[0], X[1], X[2])
    print("current ", analyticI)

    # Run the model numerically
    numerical_model = ChannelModel(params, lambda t : -80)
    solution = integrate.solve_ivp(model.getDerivatives, [0,t], model.getStates(0))
    y = solution.y

    #Get the current for the final timestep (at 1s)
    numericalI = model.calculateCurrent(y[0,-1], y[1, -1], y[2,-1])
    print("last probOpen", y[1, -1])
    print(numericalI, analyticI)

    assert((numericalI - analyticI)/analyticI < 0.1)


if __name__ == '__main__':
    main()
