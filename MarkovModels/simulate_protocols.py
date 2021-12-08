
from BeattieModel import BeattieModel
import argparse
import os
import common
import numpy as np
import logging
import matplotlib.pyplot as plt
import multiprocessing

def simulate_protocol(model, name, output_dir):
    fig = plt.figure(figsize=(14,12))
    axs = fig.subplots(4)
    print(f"Plotting {name}")
    current, S1 = model.SimulateForwardModelSensitivities()
    S1n = S1 * np.array(model.get_default_parameters())[None, :]

    param_labels = ['S(p' + str(i + 1) + ',t)' for i in range(model.n_params)]
    state_occupancies = model.GetStateVariables()

    interp_voltage, _, _, _ = common.get_protocol_from_csv(name, directory=common.get_protocol_directory())
    axs[0].plot(model.times, model.GetVoltage(), label='generated voltage function')
    axs[0].plot(model.times, [interp_voltage(t) for t in model.times], label='actual voltage')
    axs[0].plot()
    axs[0].grid(True)
    axs[0].set_xticklabels([])
    axs[0].set_ylabel('Voltage (mV)')

    axs[1].plot(model.times, current)
    axs[1].grid(True)
    axs[1].set_xticklabels([])
    axs[1].set_ylabel('Current (nA)')
    axs[2].plot(model.times, state_occupancies, label=model.state_labels + ['IC'])
    axs[2].legend(ncol=4)
    axs[2].grid(True)
    axs[2].set_xticklabels([])
    axs[2].set_ylabel('State occupancy')
    for i in range(model.n_params):
        axs[3].plot(model.times, S1n[:, i], label=param_labels[i])
    axs[3].legend(ncol=3)
    axs[3].grid(True)
    axs[3].set_xlabel('Time (ms)')
    axs[3].set_ylabel('Sensitivities')
    plt.tight_layout()

    fig.savefig(os.path.join(output_dir, f"{name}_sensitivities"))

    for ax in axs:
        ax.cla()
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Plot output from different protocols")
    output_dir = os.path.join('output', 'simulate_protocols')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tasks = []
    for fname in os.listdir(common.get_protocol_directory()):
        fname, ext = os.path.splitext(fname)
        if fname[0] == '#':
            continue
        tasks.append(fname)


    global func
    def func(protocol_name):
        if ext != ".csv":
            logging.warning(f"Using file with extension {ext}")

        protocol, t_start, t_end, t_step = common.get_protocol(protocol_name)
        times = np.linspace(t_start, t_end, int((t_end-t_start)/t_step))
        model = BeattieModel(protocol, times)
        # model.default_parameters = np.array([2.07E-3, 7.17E-2, 3.44E-5, 6.18E-2, 4.18E-1, 2.58E-2, 4.75E-2, 2.51E-2, 3.33E-2])
        model.default_parameters = np.array([0.00023215680795174809,0.07422110165735675,2.477501557744992e-05,0.04414799725791213,0.11023652619943154,0.015996823969951217,0.015877336172564104,0.027816696279347616,49.70368237942998])
        model.Erev = common.calculate_reversal_potential(298, K_out=130, K_in=5)
        simulate_protocol(model, protocol_name, output_dir)

    pool = multiprocessing.Pool(processes=min(os.cpu_count(), len(tasks))
    pool.map(func, tasks)




if __name__ == "__main__":
    main()
