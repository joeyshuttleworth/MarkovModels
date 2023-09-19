import argparse
import os
import matplotlib.pyplot as plt
from markovmodels import common
from markovmodels.ArtefactModel import ArtefactModel


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('subtraction_summary_file')
    parser.add_argument('--figsize', default=[12, 9], nargs=2, type=int)
    parser.add_argument('--output')

    global args
    args = parser.parse_args()

    global output_dir
    output_dir = common.setup_output_directory(args.output, 'show_artefact_variability')

    model_class = 'BeattieModel'
    protocol = 'staircaseramp1'
    voltage_func, times, desc = common.get_ramp_protocol_from_csv(protocol)
    c_model = common.make_model_of_class(model_class, times, voltage=voltage_func,
                                         protocol_description=desc)

    artefact_model = ArtefactModel(c_model)

    _p = artefact_model.get_default_parameters()
    p = _p.copy()

    fig = plt.figure(figsize=args.figsize)
    axs = fig.subplots(2, 2)

    axs[0, 0].plot(times*1e-3, c_model.SimulateForwardModel(), color='grey')
    axs[0, 0].set_title('no artefacts')
    axs[0, 1].plot(times*1e-3, c_model.SimulateForwardModel(), color='grey')
    axs[1, 0].plot(times*1e-3, c_model.SimulateForwardModel(), color='grey')
    axs[1, 1].plot(times*1e-3, c_model.SimulateForwardModel(), color='grey')

    axs[0, 1].plot(times*1e-3, artefact_model.SimulateForwardModel())
    axs[0, 1].set_title(f"C_m={p[-2]}, R_s={p[-1]}")

    p = _p.copy()
    p[-1] = 20 #MOhm
    p[-2] = 20e-3
    axs[1, 0].plot(times*1e-3, artefact_model.SimulateForwardModel(p))
    axs[1,0].set_title(f"C_m={p[-2]}nF, R_s={p[-1]}MOhm")
    p[-7] = 2e-3 #gS

    print(artefact_model.SimulateForwardModel(p))
    axs[1, 1].plot(times*1e-3, artefact_model.SimulateForwardModel(p))
    axs[1, 1].set_title(f"C_m={p[-2]}nF, R_s={p[-1]}MOhm, " \
                        f"g_leak=2GS E_leak=0")

    fig.savefig(os.path.join(output_dir, "artefact_impact"))


if __name__ == '__main__':
    main()
