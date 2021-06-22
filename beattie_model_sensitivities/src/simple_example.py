#!/usr/bin/env python3
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from fit_model import cov_ellipse
from common import get_args

# Compute sensitivities and Fisher information matrix for a simple
# 1-dimensional linear model, y = alpha * t + beta with i.i.d Gaussian noise,
# sigma

def main():

    args = get_args()
    if not os.path.exists(os.path.join(args.output, "simple_example")):
        os.mkdir(os.path.join(args.output, "simple_example"))

    # [alpha, beta, sigma]
    true_parameters = [2, 3, 0.25]
    forward_model   = lambda t : true_parameters[0] + t * true_parameters[1]

    # Generate data
    n_points = 100
    times = np.linspace(0,1,n_points)
    data  = forward_model(times) + np.random.normal(0, true_parameters[2], len(times))

    # Plot data against true model
    plt.plot(times, data, "o")
    plt.plot(times, forward_model(times))
    if args.plot:
        plt.show()
    else:
        plt.savefig(os.path.join(args.output, "simple_example", "synthetic_data"))

    # Sensitivities are easy to write down (use pen and paper)
    sens1 = np.ones(n_points)
    sens2 = times

    # Estimate sigma
    sigma2 = sum(data - forward_model(times))**2/(n_points-1)

    sens = np.matrix(np.stack((sens1, sens2)))

    H = sens @ sens.T

    # Compute FIM
    FIM = H/(sigma2)
    print(FIM)
    print("estimate of sigma is {}".format(math.sqrt(sigma2)))

    # Compute covariance matrix
    cov = np.linalg.inv(FIM)
    print("Covariance matrix is {}".format(cov))

    inferred_params = scipy.stats.linregress(x=times, y=data)
    inferred_params = np.array((inferred_params[1], inferred_params[0]))

    # Plot 1 s.d ellipse
    cov_ellipse(cov, offset=inferred_params)
    plt.xlabel("intercept")
    plt.ylabel("gradient")

    if args.plot:
        plt.show()
    else:
        plt.savefig(os.path.join(args.output, "simple_example", "1sd_ellipse_parameter_dist"))

if __name__ == "__main__":
    main()
