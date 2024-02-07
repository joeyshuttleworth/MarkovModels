import matplotlib.pyplot as plt
import numpy as np
from markovmodels.model_generation import make_model_of_class
from markovmodels.voltage_protocols import get_ramp_protocol_from_csv


def main():

    protocol_func, times, protocol_desc = get_ramp_protocol_from_csv('staircaseramp1')
    model_class_name = 'model3'
    m_model = make_model_of_class(model_class_name, voltage=protocol_func,
                                  times=times,
                                  protocol_description=protocol_desc)

    params = m_model.get_default_parameters()

    voltages = np.linspace(-120, 40, 1000)
    steady_states = np.vstack([m_model.rhs_inf(params, voltage).flatten() for voltage in voltages])

    print(steady_states)
    aux_func = m_model.define_auxiliary_function()
    steady_state_currents = aux_func(steady_states.T, params, voltages)

    fig = plt.figure()
    axs = fig.subplots(4)

    axs[0].plot(voltages, steady_states[:, 0])
    axs[1].plot(voltages, steady_states[:, 1])
    axs[2].plot(voltages, steady_state_currents)
    Rs = 20 # MOhm
    axs[3].plot(voltages[:-1], 1 - np.diff(steady_state_currents) * Rs * 1e-3)
    plt.show()


if __name__ == '__main__':
    main()
