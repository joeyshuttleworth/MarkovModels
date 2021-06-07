import matplotlib.pyplot as plt
import numpy as np
from hh_markov_model import ChannelModel

def main():
    params = [2.26E-04, 0.0699, 3.45E-05, 0.05462, 0.0873, 8.92E-03, 5.150E-3, 0.03158, 0.1524]
    dets = []
    volts = np.linspace(-80,50,1000)
    for volt in volts:
        model = ChannelModel(params, lambda t: volt)
        [A,B] = model.getSystemOfOdes()
        dets.append(np.linalg.det(A))


    plt.plot(volts, dets)
    plt.xlabel("Voltage / mV")
    plt.ylabel("det(A)")
    plt.show()

if __name__ == "__main__":
    main()
