import numpy as np
import pandas as pd
import os
import logging
import multiprocessing

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import seaborn as sns

from markovmodels.fitting import fit_leak_parameters_with_artefact, find_V_off, _find_conductance
from quality_control.leak_fit import fit_leak_lr
from markovmodels.model_generation import make_model_of_class
from markovmodels.voltage_protocols import get_ramp_protocol_from_csv, make_voltage_function_from_description
from markovmodels.ArtefactModel import ArtefactModel, no_artefact_parameters

from markovmodels.utilities import setup_output_directory

import matplotlib
matplotlib.use('agg')
pool_kws = {
    'maxtasksperchild': 1,
    }



def main():

    parser = ArgumentParser()
    parser.add_argument('--n_samples', '-n', type=int, default=10)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output')
    parser.add_argument('--figsize', nargs=2, type=float, default=[5.4, 3.5])
    parser.add_argument('--sampling_rate', type=float, default=2.0)
    parser.add_argument('--reversal', type=float, default=-91.71)
    parser.add_argument('--noise_sigma', type=float, default=10.0)
    parser.add_argument('--pool_size', '-c', type=int, default=1)
    parser.add_argument('--protocol', default='staircaseramp')
    parser.add_argument('--model_class', default='model3')
    parser.add_argument('--default_params_file')

    global args
    args = parser.parse_args()

    if args.default_params_file is None:
        args.default_params_file = os.path.join('data', 'Beattie_Sinusoidal_params.csv')

    if args.seed is None:
        args.seed = np.random.randint(0, 9223372036854775807, dtype=np.int64)

    rng = np.random.default_rng(seed=args.seed)
    artefacts_df = generate_artefact_parameters(n_samples=args.n_samples,
                                                rng=rng)

    global output_dir
    output_dir = setup_output_directory(args.output, 'analyse_offset_inference')

    with open(os.path.join(output_dir, 'seed.txt'), 'w') as fout:
        fout.write(f"{args.seed}\n")

    # Save parameters to file
    artefacts_df.to_csv(os.path.join(output_dir, 'artefact_parameters.csv'))

    protocol_dir = os.path.join('markovmodels', 'protocols')
    voltage_func, _times, protocol_desc = get_ramp_protocol_from_csv(
        args.protocol,
        protocol_dir
    )

    times = np.arange(_times[0], _times[-1], 1.0/args.sampling_rate)

    default_parameters = np.loadtxt(args.default_params_file, delimiter=',').flatten()
    model = make_model_of_class(args.model_class, times=times,
                                protocol_description=protocol_desc,
                                E_rev=args.reversal,
                                voltage=voltage_func,
                                default_parameters=default_parameters)

    a_model = ArtefactModel(model)
    a_solver_states = a_model.make_hybrid_solver_states(hybrid=False,
                                                        njitted=False,
                                                        strict=False,
                                                        times=times)

    a_solver_current = a_model.make_hybrid_solver_current(hybrid=False, return_var='I_out',
                                                          njitted=False)

    a_solver_states = a_model.make_hybrid_solver_states(hybrid=False,
                                                        njitted=False)

    res = []
    for index, row in artefacts_df.iterrows():
        res.append(generate_data(index, **row,
                                 default_parameters=default_parameters,
                                 rng=rng, a_solver_current=a_solver_current,
                                 times=times))

    res = np.vstack(res)

    # artefacts_df['Rs_est'] = np.nan
    # artefacts_df['Cm_est'] = np.nan
    artefacts_df['gkr_est'] = np.nan
    artefacts_df['gleak_est'] = np.nan
    artefacts_df['Eleak_est'] = np.nan
    artefacts_df['V_off_est'] = np.nan

    tasks = []
    for current, (index, row) in zip(res, artefacts_df.iterrows()):
        V_off_plots_dir = os.path.join(output_dir, 'V_off_plots_dir',
                                       f"index_{index}")
        if not os.path.exists(V_off_plots_dir):
            os.makedirs(V_off_plots_dir)
        tasks.append([args, current,
                      protocol_desc,
                      times,
                      args.model_class,
                      os.path.join(V_off_plots_dir,
                                   f"sample_{index}.pdf"),
                      args.reversal,
                      default_parameters,
                      V_off_plots_dir])

    with multiprocessing.Pool(min(args.pool_size, len(tasks)), **pool_kws) as pool:
        infer_res = pool.starmap(infer_artefact_values, tasks, chunksize=1)

    for index, new_vals in zip(artefacts_df.index, infer_res):
        for key, val in new_vals.items():
            artefacts_df.loc[index][key] = val

    artefacts_df.to_csv(os.path.join(output_dir, "fitted_artefact_parameters.csv"))

    scatterplot_estimates(artefacts_df)


def scatterplot_estimates(artefacts_df):
    fig = plt.figure(figsize=args.figsize)
    ax = fig.subplots()

    ax.spines[['top', 'right']].set_visible(False)

    param_pairs = [
        ['gleak', 'gleak_est'],
        ['Eleak', 'Eleak_est'],
        ['V_off', 'V_off_est']
    ]

    pretty_vars_dict = {
        'gleak': r'$g_\text{l}$',
        'gleak_est': r'$\hat g_\text{l}$',
        'Eleak': r'$E_\text{l}$',
        'Eleak_est': r'$\hat E_\text{l}$',
        'V_off': r'$V_\text{off}$',
        'V_off_est': r'$\hat V_\text{off}$',
    }

    for var, var_est in param_pairs:
        lam = np.linspace(artefacts_df[[var, var_est]].min(),
                        artefacts_df[[var, var_est]].max(),
                        3)
        ax.plot(lam, lam, ls='--', color='grey', alpha=.2)
        sns.scatterplot(data=artefacts_df, x=var, y=var_est)

        ax.set_xlabel(pretty_vars_dict[var])
        ax.set_ylabel(pretty_vars_dict[var_est])

        ax.spines[['top', 'right']].set_visible(False)
        fig.savefig(os.path.join(output_dir, f"{var}_est_scatter"))
        ax.cla()

    plt.close(fig)



def infer_artefact_values(args, current, protocol_desc, times, model_class,
                          output_path, E_rev, default_parameters,
                          forward_sim_output_dir=None, a_solver_states=None,
                          a_solver_current=None, voltage_func=None):

    if not voltage_func:
        voltage_func = make_voltage_function_from_description(protocol_desc)

    Vcmd = np.array([voltage_func(t) for t in times])

    # Infer gleak and Eleak from trace (ignoring artefacts)
    leak_ramp_i = [i for i, l in enumerate(protocol_desc) if l[2] != l[3]][0]
    ramp_end = protocol_desc[leak_ramp_i + 1, 1] - 50.0
    ramp_start = protocol_desc[leak_ramp_i - 1, 0] + 50.0

    dt = times[1] - times[0]
    g_leak_est, E_leak_est, _, _, _, _, _ = fit_leak_lr(
        Vcmd, current, dt=dt,
        ramp_start=ramp_start,
        ramp_end=ramp_end
    )

    model = make_model_of_class(model_class, times=times, E_rev=args.reversal,
                                protocol_description=protocol_desc,
                                voltage=voltage_func,
                                default_parameters=default_parameters)

    a_model = ArtefactModel(model)
    aux_func = a_model.define_auxiliary_function(return_var='I_out')
    default_artefact_parameters = a_model.get_default_parameters()

    if a_solver_states is None:
        a_solver_states = a_model.make_hybrid_solver_states(hybrid=False,
                                                            strict=False)

    if a_solver_current is None:
        a_solver_current = a_model.make_hybrid_solver_current(hybrid=False,
                                                              strict=False,
                                                              return_var='I_out')

    try:
        V_off, success = find_V_off(protocol_desc, times, current, model_class,
                                    default_artefact_parameters, E_rev, pp_gleak=g_leak_est,
                                    pp_Eleak=E_leak_est, output_path=output_path,
                                    forward_sim_output_dir=forward_sim_output_dir,
                                    a_solver_states=a_solver_states,
                                    a_solver_current=a_solver_current,
                                    data_label='before',
                                    leak_parameters_initial_guess=[g_leak_est, E_leak_est],
                                    aux_func=aux_func)

    except ValueError as exc:
        logging.warning(f"find_V_off failed: {str(exc)}")
        ret_dict = {
            'gkr_est': np.nan,
            'gleak_est': np.nan,
            'Eleak_est': np.nan,
            'V_off_est': np.nan
        }

        return ret_dict

    if voltage_func is None:
        voltage_func = model.voltage

    voltages = np.array([voltage_func(t) for t in times])
    gleak, Eleak = fit_leak_parameters_with_artefact(a_model,
                                                     protocol_desc, times, current,
                                                     voltages,
                                                     default_artefact_parameters,
                                                     a_solver_current=a_solver_current)

    gkr_index = model.GKr_index

    # Find end of reversal ramp
    ramp = [line for line in protocol_desc if line[2] != line[3]][-1]
    start_t, end_t = ramp[0:2]

    # Fit for middle of ramp
    ramp_length = end_t - start_t
    start_t += 0.35 * ramp_length
    end_t -= 0.35 * ramp_length

    indices = np.argwhere((times > start_t) & (times < end_t))

    g_leak_index = -no_artefact_parameters + 1
    E_leak_index = -no_artefact_parameters + 2
    default_artefact_parameters[g_leak_index] = gleak
    default_artefact_parameters[E_leak_index] = Eleak

    gkr = _find_conductance(a_solver_current, protocol_desc, times, current, indices,
                            voltages, default_artefact_parameters, E_rev, gkr_index,
                            a_model, bounds=None)

    ret_dict = {
        'gkr_est': gkr,
        'gleak_est': gleak,
        'Eleak_est': Eleak,
        'V_off_est': V_off
    }

    return ret_dict


def generate_data(index, Rseries, Cm, gkr, gleak, Eleak, V_off, default_parameters, rng,
                  a_solver_current, times=None):
    params = default_parameters.copy()
    params[-1] = gkr
    params = np.append(params, [args.reversal, gleak, Eleak, 0, 0, V_off, Rseries, Cm])

    current = a_solver_current(params, times=times).flatten()
    assert current.shape == times.shape
    current += rng.normal(0, args.noise_sigma, size=current.shape)

    # Plot and save data
    sub_dir = os.path.join(output_dir, 'generated_data')

    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    fig = plt.figure(figsize=args.figsize)
    ax = fig.subplots()

    ax.plot(times, current, color='grey')
    ax.set_ylabel(r'$I_\text{out}$ (pA)')
    ax.set_xlabel(r'$t$ (ms)')

    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(os.path.join(sub_dir, f"current_{index}"))

    plt.close(fig)

    return current


def generate_artefact_parameters(n_samples, rng):

    # Rseries in MOhm
    Rseries = 10**rng.normal(np.log10(12.5), np.log10(2), size=n_samples) * 1e-3

    # Cm in pF
    Cm = 10**rng.normal(np.log10(12.5), np.log10(2), size=n_samples) * 1e-3

    gkr = 10**rng.normal(np.log10(200), 0.5, size=n_samples)

    # gleak in pS
    gleak = 10**rng.normal(np.log10(3), 1, size=n_samples)

    # Eleak in mV
    Eleak = rng.uniform(-50, 25, size=n_samples)

    # V_off in mV
    V_off = rng.normal(0, 5, size=n_samples)

    df = pd.DataFrame(np.vstack([Rseries,
                                Cm,
                                gkr,
                                gleak,
                                Eleak,
                                 V_off]).T,
                      columns = ['Rseries', 'Cm', 'gkr', 'gleak', 'Eleak', 'V_off'])
    print(df)
    return df


if __name__ == '__main__':
    main()
