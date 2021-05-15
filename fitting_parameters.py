import pints
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.integrate as integrate
from hh_markov_model import ChannelModel


def SineWaveProtocol(t):
    """The voltage protocol used to produce the Sine Wave data in
    https://github.com/mirams/sine-wave. This code is translated from the
    function, static int f in
    https://github.com/mirams/sine-wave/blob/master/Code/MexAslanidi.c.
    """

    # This shift is needed for simulated protocol to match the protocol recorded in experiment, which is shifted by 0.1ms as compared to the original input protocol. Consequently, each step is held for 0.1ms longer in this version of the protocol as compared to the input.
    shift = 0.1

    C = [54.0, 26.0, 10.0, 0.007/(2*np.pi), 0.037/(2*np.pi), 0.19/(2*np.pi)]
    if t>=0 and t<250+shift:
        return -80
    elif t>=250+shift and t<300+shift:
        return -120
    elif t>=300+shift and t<500+shift:
        return -80
    elif t>=500+shift and t<1500+shift:
        return 40
    elif t>=1500+shift and t<2000+shift:
        return -120
    elif t>=2000+shift and t<3000+shift:
        return -80
    elif t>=3000+shift and t<6500+shift:
        v=-30+C[0]*(np.sin(2*np.pi*C[3]*(t-2500-shift))) + C[1]*(np.sin(2*np.pi*C[4]*(t-2500-shift))) + C[2]*(np.sin(2*np.pi*C[5]*(t-2500-shift)))
        # print(v)
        return(v)
    elif t>=6500+shift and t<7000+shift:
        return -120
    elif t>= 7000+shift and t<8000+shift:
        return -80
    else:
        # print("voltage out of bounds")
        return -999


class ChannelModelPintsWrapper(pints.ForwardModelS1):
    def n_parameters(self):
        return 9

    def simulate(self, parameters, times):
        mdl = ChannelModel(parameters, SineWaveProtocol)
        # Solve the IVP at the required times
        solution = integrate.solve_ivp(mdl.getDerivatives, [times[0], times[-1]], mdl.getStates(0), t_eval=times, rtol=1E-8, atol=1E-8, method = "LSODA")
        y = solution.y
        #Now calculate the corresponding currents
        IVec = [mdl.calculateCurrent(y[1,t], times[t]) for t in range(0,len(solution.t))]
        print("plotting [Open]")
        plt.plot(times, y[1,:])
        plt.plot(times, IVec)
        plt.show()
        return IVec

    def simulateS1(self, parameters, time):
        mdl = ChannelModel(parameters, SineWaveProtocol)
        # Solve the IVP at the required times
        solution = integrate.solve_ivp(mdl.getDerivatives, [times[0], times[-1]], mdl.getStates(0), t_eval=times, rtol=1E-8, atol=1E-8, method = "LSODA")
        y = solution.y
        #Now calculate the corresponding currents
        IVec = [mdl.calculateCurrent(y[1,t], times[t]) for t in range(0,len(solution.t))]

        return [IVec, mdl.calculateDerivatives(times, y)]



class MarkovModelBoundaries(pints.Boundaries):
    def check(self, parameters):
        '''Check that each rate constant lies in the range 1.67E-5 < A*exp(B*V) < 1E3
        '''

        for i in range(0, 4):
            alpha = parameters[2*i]
            beta  = parameters[2*i + 1]

            vals = [0,0]
            vals[0] = alpha * np.exp(beta * -90 * 1E-3)
            vals[1] = alpha * np.exp(beta *  50 * 1E-3)

            for val in vals:
                if val < 1.67E-5 or val > 1E3:
                    return False
        # Check maximal conductance
        if parameters[8] > 0 and parameters[8] < 2:
            return True
        else:
            return False

    def n_parameters(self):
        return 9

def extract_time_ranges(lst, time_ranges):
    """
    Take values from a list, lst which are indexes between the upper and lower
    bounds provided by time_ranges. Each element of time_ranges specifies an
    upper and lower bound.

    Returns a 2d numpy array containing all of the relevant data points
    """
    ret_lst = []
    for time_range in time_ranges:
        ret_lst.extend(lst[time_range[0]:time_range[1]].tolist())
    return np.array(ret_lst)

def main():
    #constants
    timeRangesToUse = [[1,2499], [2549,2999], [3049,4999], [5049,14999], [15049,19999], [20049,29999], [30049,64999], [65049,69999], [70049,-1]]
    starting_parameters = [3.87068845e-04, 5.88028759e-02, 6.46971727e-05, 4.87408447e-02, 8.03073893e-02, 7.36295506e-03, 5.32908518e-03, 3.32254316e-02, 6.56614672e-02]

    model = ChannelModelPintsWrapper()
    data  = pd.read_csv("data/averaged-data.txt", delim_whitespace=True)
    dat = extract_time_ranges(data.values, timeRangesToUse)
    times=dat[:,0]
    values=dat[:,1]
    current = model.simulate(starting_parameters, times)
    problem = pints.SingleOutputProblem(model, times, values)
    error = pints.SumOfSquaresError(problem)
    boundaries  = MarkovModelBoundaries()
    x0 = np.array([0.1]*9)
    found_parameters, found_value = pints.optimise(error, starting_parameters, boundaries=boundaries)
    print(found_parameters, found_value)

if __name__ == "__main__":
    main()
    print("done")
