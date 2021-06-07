import pytest
import numpy as np
from hh_markov_model import ChannelModel
import scipy.integrate as integrate

def compare_solutions(params, max_t=10):
    #Setup the model with an example parameter set and constant voltage

    assert len(params) == 9, "Parameter vector has the wrong dimensions"

    if any(x < 0 for x in params) == True:
        raise ValueError("Negative parameters not allowed")

    model = ChannelModel(params, lambda t: -80)

    #Get the transition rates of the model
    [k1, k2, k3, k4] = model.getTransitionRates()

    #Use the initial conditions to set C1 and C2 assuming x1,x2 = 0 (equivalently, [C] = 1).
    C1 = -k1/(k1+k2)
    C2 = -k3/(k3+k4)

    # Calculate the current at time t analytically
    t = np.linspace(0, max_t, 10000)
    x1 = C1*np.exp(t*(-k1-k2)) + k1/(k1+k2)
    x2 = C2*np.exp(t*(-k3-k4)) + k3/(k3+k4)

    analyticI = model.calculateCurrent((1-x1[-1])*(1-x2[-1]), x1[-1]*(1-x2[-1]), x1[-1]*x2[-1])

    # Run the model numerically
    numerical_model = ChannelModel(params, lambda t : -80)
    solution = integrate.solve_ivp(model.getDerivatives, [0,t[-1]], model.getStates(0))
    y = solution.y

    #Get the current for the final timestep (at t)
    numericalI = model.calculateCurrent(y[0,-1], y[1, -1], y[2,-1])

    print(numericalI, analyticI)
    if not analyticI == 0:
        assert np.abs(numericalI - analyticI)/analyticI < 0.1, "Solutions don't match"


def test_solutions_agree():
    params = [2.26E-04, 0.0699, 3.45E-05, 0.05462, 0.0873, 8.92E-03, 5.150E-3, 0.03158, 0.1524]
    for t in np.linspace(0, 1000, 101):
        compare_solutions(params, 1000)


def test_nonsensical_parameters():
    params = -1 * np.ones(9)
    with pytest.raises(ValueError, match="Negative parameters not allowed"):
        compare_solutions(params, 1000)


def test_incorrect_parameter_vector():
    params = [2.26E-04, 0.0699, 3.45E-05, 0.05462, 0.0873, 8.92E-03, 5.150E-3, 0.03158]
    with pytest.raises(AssertionError, match="Parameter vector has the wrong dimensions"):
        compare_solutions(params, 1000)

