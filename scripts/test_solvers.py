from markovmodels import common
from markovmodels.BeattieModel import BeattieModel
from markovmodels.WangModel import WangModel
from markovmodels.KempModel import KempModel

import matplotlib.pyplot as plt
import argparse

import numpy as np
import pandas as pd
import os
from matplotlib.gridspec import GridSpec

from matplotlib import rc

from numba import njit

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 24})
rc('text', usetex=True)
rc('figure', dpi=500)

relabel_dict = {# 'longap': '$d_0$',
                # 'hhbrute3gstep': '$d_1$',
                'sis': '$d_2$',
                'spacefill19': '$d_3$',
                'staircaseramp1': '$d_4$',
                # 'staircaseramp2': '$d_4^*$',
                # 'wangbrute3gstep': '$d_5$'
                }

protocols_list = [k for k, v in relabel_dict.items()]


def setup_axes(fig):
    gs = GridSpec(6, 3, figure=fig, width_ratios=[.05, 1, 1])
    axes = [fig.add_subplot(cell) for cell in gs]
    return axes


def main():

    parser = argparse.ArgumentParser('--figsize')
    parser.add_argument('--output', '-o')
    parser.add_argument('--figsize', nargs=2, type=float, default=[10, 8])
    parser.add_argument('--noise', default=0.03)
    parser.add_argument('--well', '-w', default='B24')
    parser.add_argument('--experiment_name', default='fluoride_free')

    global args
    args = parser.parse_args()

    global output_dir
    output_dir = common.setup_output_directory(args.output, 'test_solvers')

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    axes = setup_axes(fig)

    axes[1].set_title(r'$V$ (mV)')
    axes[2].set_title(r'$I_{\textrm{Kr}}$ (nA)')

    for i, protocol in enumerate([p for p in protocols_list if p != 'staircaseramp2']):
        print(protocol)
        sim_protocol(protocol)


default_params = BeattieModel().get_default_parameters()


@njit
def step_solver(times, V, p, y0):

    ret_vec = np.full((times.shape[0], 3), np.nan)
    alpha_a = p[0] * np.exp(p[1] * V)
    beta_a = p[2] * np.exp(- p[3] * V)

    IC = 1 - y0.sum()
    C = y0[0]
    O = y0[1]
    I = y0[2]

    old_a = O + I
    old_r = I + IC

    a_inf = alpha_a / (alpha_a + beta_a)

    a = a_inf + (old_a - a_inf) * np.exp((beta_a + alpha_a) * -times)

    alpha_r = p[4] * np.exp(p[5] * V)
    beta_r = p[6] * np.exp(-p[7] * V)
    r_inf = alpha_r / (alpha_r + beta_r)

    r = r_inf + (old_r - r_inf) * np.exp((beta_r + alpha_r) * -times)

    ret_vec[:, 0] = (1 - a) * (1 - r)
    ret_vec[:, 1] = a * (1 - r)
    ret_vec[:, 2] = a * r

    return ret_vec, True


def sim_protocol(protocol, times=None, solver_type=''):
    voltage_func, times, protocol_desc = common.get_ramp_protocol_from_csv(protocol)
    model = BeattieModel(voltage=voltage_func,
                         protocol_description=protocol_desc,
                         times=times.flatten())

    res1 = model.make_ida_solver_current(njitted=True)()
    res2 = model.make_forward_solver_current(njitted=True)(atol=1e-12, rtol=1e-12)
    res3 = model.make_hybrid_solver_current(njitted=False)()

    reference_solution = model.make_hybrid_solver_current(analytic_solver=step_solver,
                                                          njitted=False)()

    fig = plt.figure()
    ax = fig.subplots()

    ax.plot(times, res1, label='ida')
    ax.plot(times, res2, label='lsoda')
    ax.legend()
    fig.savefig(os.path.join(output_dir, 'solver_comparison_ida_lsoda_%s' % protocol))

    ax.cla()
    ax.plot(times, res2, label='lsoda')
    ax.plot(times, res3, label='hybrid')
    ax.plot(times, reference_solution, label='HH')
    fig.savefig(os.path.join(output_dir, 'solver_comparison_lsoda_hybrid_%s' % protocol))
    ax.legend()
    ax.cla()

    ax.plot(times, res2-res1, label='ida error')
    fig.savefig(os.path.join(output_dir, 'ida_lsoda_errors_%s' % protocol))
    ax.cla()

    ax.plot(times, np.log10(np.abs(res3 - reference_solution)), label='hybrid error')
    ax.plot(times, np.log10(np.abs(res2 - reference_solution)), label='LSODA error')
    ax.axhline(0, linestyle='--')
    ax.legend()
    fig.savefig(os.path.join(output_dir, 'hybrid_lsoda_errors_%s' % protocol))

    hybrid_errors = res3 - res2

    print('hybrid errors', times[np.argmax(hybrid_errors)])


if __name__ == '__main__':
    main()
