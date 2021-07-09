#!/usr/bin/env python3

import argparse
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import numpy as np
import pandas as pd
import math
import os
import pints

def get_args(data_reqd=False):
    # Check input arguments
    parser = get_parser(data_reqd)
    args = parser.parse_args()

    # Create output directory
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    return args

def get_parser(data_reqd=False):
    # Check input arguments
    parser = argparse.ArgumentParser(
        description='Plot sensitivities of the Beattie model')
    if data_reqd:
        parser.add_argument("data_file_path", help="path to csv data for the model to be fit to")

    parser.add_argument("-s", "--sine_wave", action='store_true', help="whether or not to use sine wave protocol",
        default=False)
    parser.add_argument("-p", "--plot", action='store_true', help="whether to plot figures or just save",
        default=False)
    parser.add_argument("--dpi", type=int, default=100, help="what DPI to use for figures")
    parser.add_argument("-o", "--output", type=str, default="output", help="The directory to output figures and data to")
    return parser



def calculate_reversal_potential(temp = 20):
    # E is the Nernst potential for potassium ions across the membrane
    # Gas constant R, temperature T, Faradays constat F
    R = 8314.55
    T = temp+273.15
    F = 96485

    # Intracellular and extracellular concentrations of potassium.
    K_out = 4
    K_in  = 130

    # valency of ions (1 in the case of K^+)
    z = 1

    # Nernst potential
    E = R*T/(z*F) * np.log(K_out/K_in)
    return E


def cov_ellipse(cov, offset=[0,0], q=None, nsig=None, **kwargs):
    """
    Parameters
    ----------
    copied from stackoverflow


    cov : (2, 2) array
        Covariance matrix.
    q : float, optional
        Confidence level, should be in (0, 1)
    nsig : int, optional
        Confidence level in unit of standard deviations.
        E.g. 1 stands for 68.3% and 2 stands for 95.4%.

    Returns
    -------
    width, height, rotation :
         The lengths of two axises and the rotation angle in degree
    for the ellipse.
    """

    if q is not None:
        q = np.asarray(q)
    elif nsig is not None:
        q = 2 * scipy.stats.norm.cdf(nsig) - 1
    else:
        raise ValueError('One of `q` and `nsig` should be specified.')

    qs = np.sort(q)

    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    for q in qs:
        r2 = scipy.stats.chi2.ppf(q, 2)

        val, vec = np.linalg.eigh(cov)
        width, height = 2 * np.sqrt(val[:, None] * r2)
        rotation = np.arctan2(*vec[::-1, 0])

        # print("width, height, rotation = {}, {}, {}".format(width, height, math.degrees(rotation)))

        e = matplotlib.patches.Ellipse(offset, width[0], height[0], math.degrees(rotation), color=np.random.rand(3), fill=False, label="{}% confidence region".format(int(q*100)))
        ax.add_patch(e)
        e.set_clip_box(ax.bbox)

        window_width = np.abs(width[0]*np.cos(rotation)*1.5)
        window_height= np.abs(height[0]*np.sin(rotation)*1.5)
        max_dim = max(window_width, window_height)

    ax.set_xlim(offset[0]-max_dim, offset[0]+max_dim)
    ax.set_ylim(offset[1]-max_dim, offset[1]+max_dim)
    return fig, ax

# TODO Add some processing so it outputs the system in AX + B form
def equations_from_adjacency_matrix(A, index_to_elim):
    # TODO Check that the graph is connected using a library such as py_graph
    # TODO Check types
    A = np.array(A)
    y = se.Matrix([[se.symbols("y%d" % i)] for i in range(0, A.shape[0])])

    eqns = []
    for row in A:
        element = -sum([el*v for el, v in zip(row, y)])
        for el, x in zip(A,y):
            element += x * el
        eqns.append(element)
    eqns = np.array(eqns).T.tolist()[0]
    del eqns[index_to_elim]
    eqns = se.Matrix([row.simplify() for row in eqns])
    eqns=eqns.subs(y[index_to_elim], 1 - sum([x for x in y if x != y[index_to_elim]]))

    # Output as a list of equations
    print(eqns)
    return eqns

def extract_times(lst, time_ranges, step):
    """
    Take values from a list, lst which are indexes between the upper and lower
    bounds provided by time_ranges. Each element of time_ranges specifies an
    upper and lower bound.

    Returns a 2d numpy array containing all of the relevant data points
    """
    if time_ranges == None:
        return lst
    ret_lst = []
    for time_range in time_ranges:
        ret_lst.extend(lst[time_range[0]:time_range[1]:step].tolist())
    return np.array(ret_lst)

def remove_indices(lst, indices_to_remove):
    if indices_to_remove == None:
        return lst

    first_lst = lst[0:indices_to_remove[0][0]]

    lsts = []
    for i in range(1, len(indices_to_remove)):
        lsts.append(lst[indices_to_remove[i-1][1] : indices_to_remove[i][0]+1])

    lsts.append(lst[indices_to_remove[-1][1]:-1])

    lst = first_lst + [index for lst in lsts for index in lst]
    return lst

def detect_spikes(x, y):
    dx = np.diff(x)
    dy = np.diff(y)

    deriv = dy/dx
    spike_indices = np.argwhere(np.abs(deriv)>10000)[:,0]

    return x[spike_indices]

def beattie_sine_wave(t):
        # This shift is needed for simulated protocol to match the protocol recorded in experiment, which is shifted by 0.1ms compared to the original input protocol. Consequently, each step is held for 0.1ms longer in this version of the protocol as compared to the input.
        shift = 0.1
        C = [54.0, 26.0, 10.0, 0.007/(2*np.pi), 0.037/(2*np.pi), 0.19/(2*np.pi)]

        if t >= 250+shift and t < 300+shift:
            V = -120
        elif t >= 500+shift and t < 1500+shift:
            V = 40
        elif t >= 1500+shift and t < 2000+shift:
            V = -120
        elif t >= 3000+shift and t < 6500+shift:
            V = -30 + C[0] * (np.sin(2*np.pi*C[3]*(t-2500-shift))) + C[1] * \
            (np.sin(2*np.pi*C[4]*(t-2500-shift))) + C[2] * (np.sin(2*np.pi*C[5]*(t-2500-shift)))
        elif t >= 6500+shift and t < 7000+shift:
            V = -120
        else:
            V = -80
        return V

def get_staircase_protocol(par):
    protocol = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "protocols", "protocol-staircaseramp.csv"))

    times = 1000*protocol["time"].values
    voltages = protocol["voltage"].values

    spikes = 1000*detect_spikes(protocol["time"], protocol["voltage"])

    staircase_protocol = scipy.interpolate.interp1d(times, voltages, kind="linear")
    staircase_protocol_safe = lambda t : staircase_protocol(t) if t < times[-1] else par.holding_potential
    return staircase_protocol_safe



def draw_cov_ellipses(para, settings, S1n=None, sigma2=None, cov=None, plot_dir=None):
    if S1n is None and cov is None:
        raise
    elif S1n is not None and cov is not None:
        raise

    for j in range(0, settings.n_params):
        for i in range(j+1, settings.n_params):
            parameters_to_view = np.array([i,j])
            if S1n is not None:
                if sigma2 is None:
                    raise
                sub_sens = S1n[:,[i,j]]
                sub_cov = sigma2*np.linalg.inv(np.dot(sub_sens.T, sub_sens))
            # Else use cov
            else:
                sub_cov = cov[parameters_to_view[:,None], parameters_to_view]
            eigen_val, eigen_vec = np.linalg.eigh(sub_cov)
            eigen_val=eigen_val.real
            if eigen_val[0] > 0 and eigen_val[1] > 0:
                cov_ellipse(sub_cov, q=[0.5, 0.95], offset=[1,1]) # Parameters have been normalised to 1
                plt.ylabel("parameter {}".format(i+1))
                plt.xlabel("parameter {}".format(j+1))
                plt.legend()
                if plot_dir is None:
                    plt.show()
                else:
                    plt.savefig(os.path.join(plot_dir, "covariance_for_parameters_{}_{}".format(j+1,i+1)))
                plt.clf()
            else:
                print("COV_{},{} : negative eigenvalue: {}".format(i,j, eigen_val))

def fit_model(funcs, times, values, starting_parameters, par, max_iterations=None):
    class Boundaries(pints.Boundaries):
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
                    if val < 1E-7 or val > 1E3:
                        return False
            # Check maximal conductance
            if parameters[8] > 0 and parameters[8] < 2:
                return True
            else:
                return False

        def n_parameters(self):
            return 9

    class PintsWrapper(pints.ForwardModelS1):

        def __init__(self, settings, funcs):
            self.funcs = funcs
            self.settings = settings

        def n_parameters(self):
            return self.settings.n_params

        def simulate(self, parameters, times):
            ret = self.funcs.SimulateForwardModel(parameters)
            # print(ret.shape)
            return ret

        def simulateS1(self, parameters, times):
            return self.funcs.SimulateForwardModelSensitivites(parameters, data), self.times_to_use, 1

    model = PintsWrapper(par, funcs)
    problem = pints.SingleOutputProblem(model, times, values)
    error = pints.SumOfSquaresError(problem)
    boundaries  = Boundaries()
    controller=pints.OptimisationController(error, starting_parameters, boundaries=boundaries)
    if max_iterations is not None:
        print("Setting max iterations = {}".format(max_iterations))
        controller.set_max_iterations(max_iterations)
    found_parameters, found_value = controller.run()
    return found_parameters, found_value

