#!/usr/bin/env python3

import numpy as np
import symengine as se
import matplotlib.pyplot as plt
plt.rcParams['axes.axisbelow'] = True
import argparse

from settings import Params
from sensitivity_equations import GetSensitivityEquations, GetSymbols

# Check input arguments
parser = argparse.ArgumentParser(
    description='Plot sensitivities of the Beattie model')
parser.add_argument("-s", "--sine_wave", action='store_true', help="whether or not to use sine wave protocol",
    default=False)
parser.add_argument("-p", "--plot", action='store_true', help="whether to plot figures or just save",
    default=False)
parser.add_argument("--dpi", type=int, default=100, help="what DPI to use for figures")
args = parser.parse_args()

par = Params()

# Choose starting parameters (from J Physiol paper)
para = [2.26E-04, 0.0699, 3.45E-05, 0.05462, 0.0873, 8.92E-03, 5.150E-3, 0.03158, 0.1524]

# Create symbols for symbolic functions
p, y, v = GetSymbols(par)

# Define system equations and initial conditions
k1 = p[0] * se.exp(p[1] * v)
k2 = p[2] * se.exp(-p[3] * v)
k3 = p[4] * se.exp(p[5] * v)
k4 = p[6] * se.exp(-p[7] * v)

k  = np.array([k1,k2,k3,k4]).T

# Write in matrix form taking y = ([C], [O], [I])^T

A=se.Matrix([[-k1 -k3 -k4, k2 - k4, -k4], [k1, -k2 - k3, k4], [-k1, k3-k1, -k2 - k4 - k1]])
B = se.Matrix([k4,0,k1])

rhs = np.array(A*y + B)[0]

times = np.linspace(0, par.tmax, par.tmax+1)

funcs = GetSensitivityEquations(par, p, y, v, A, B, para, times, sine_wave=args.sine_wave)

S1 = funcs.SimulateForwardModelSensitivities(para)
S1n = funcs.NormaliseSensitivities(S1, para)

state_variables = funcs.GetStateVariables(para)
state_labels = ['C', 'O', 'I', 'IC']

param_labels = ['S(p' + str(i + 1) + ',t)' for i in range(par.n_params)]

fig = plt.figure(figsize=(8, 8), dpi=args.dpi)
ax1 = fig.add_subplot(411)
ax1.plot(funcs.GetVoltage())
ax1.grid(True)
ax1.set_xticklabels([])
ax1.set_ylabel('Voltage (mV)')
ax2 = fig.add_subplot(412)
ax2.plot(funcs.SimulateForwardModel(para))
ax2.grid(True)
ax2.set_xticklabels([])
ax2.set_ylabel('Current (nA)')
ax3 = fig.add_subplot(413)
for i in range(par.n_state_vars + 1):
    ax3.plot(state_variables[:, i], label=state_labels[i])
ax3.legend(ncol=4)
ax3.grid(True)
ax3.set_xticklabels([])
ax3.set_ylabel('State occupancy')
ax4 = fig.add_subplot(414)
for i in range(par.n_params):
    ax4.plot(S1n[:, i], label=param_labels[i])
ax4.legend(ncol=3)
ax4.grid(True)
ax4.set_xlabel('Time (ms)')
ax4.set_ylabel('Sensitivities')
plt.tight_layout()

if not args.plot:
    plt.savefig('ForwardModel_SW_' + str(args.sine_wave) + '.png')

H = np.dot(S1n.T, S1n)
evals = np.linalg.eigvals(H)
print('Eigenvalues of H:')
eigvals = evals/np.max(evals)
print(eigvals)

fig = plt.figure(figsize=(6, 6), dpi=args.dpi)
ax = fig.add_subplot(111)
for i in eigvals:
    ax.axhline(y=i, xmin=0.25, xmax=0.75)
ax.set_yscale('log')
ax.set_xticks([])
ax.grid(True)

if not args.plot:
    plt.savefig('Eigenvalues_SW_' + str(args.sine_wave) + '.png')

if args.plot:
    plt.show()
