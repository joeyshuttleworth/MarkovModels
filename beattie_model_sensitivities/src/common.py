#!/usr/bin/env python3

import argparse
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import numpy as np
import math
import os

def get_args(data_reqd=False):
    # Check input arguments
    parser = get_parser()
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



def calculate_resting_potential(temp = 20):
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

