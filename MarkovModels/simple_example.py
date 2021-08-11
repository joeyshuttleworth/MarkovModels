#!/usr/bin/env python3
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from common import get_args, cov_ellipse

# Compute sensitivities and Fisher information matrix for a simple
# 1-dimensional linear model, y = alpha * t + beta with i.i.d Gaussian noise,
# sigma


def main():

    args = get_args()
    if not os.path.exists(os.path.join(args.output, "simple_example")):
        os.mkdir(os.path.join(args.output, "simple_example"))

    # [alpha, beta, sigma]
    true_parameters = [2, 3, 0.25]
    def forward_model(t): return true_parameters[0] + t * true_parameters[1]

    # Generate data
    # Use a lot of data points - the approximations used only hold asymptotically
    n_points = 1000
    times = np.linspace(0, 1, n_points)
    data = forward_model(times) + np.random.normal(0,
                                                   true_parameters[2], len(times))

    # Sensitivities are easy to write down (use pen and paper)
    sens1 = np.ones(n_points)
    sens2 = times

    # Compute inferred parameters
    inferred_params = scipy.stats.linregress(x=times, y=data)
    inferred_params = np.array((inferred_params[1], inferred_params[0]))

    # Estimate sigma
    sigma2 = sum(
        (inferred_params[0] + inferred_params[1]*times - data)**2)/(n_points-1)
    print("observed sigma^2 vs true value\t{}, {}".format(
        sigma2, true_parameters[2]**2))

    sens = np.matrix(np.stack((sens1, sens2)))

    H = sens @ sens.T

    # Compute FIM
    FIM = H/(sigma2)

    # Compute observed covariance matrix
    cov = np.linalg.inv(FIM)
    print("Covariance matrix is {}".format(cov))

    print("Fisher information matrix is:\n{}".format(FIM))

    # Add samples to the ellipse
    n_samples = 1000
    samples = np.random.multivariate_normal(inferred_params, cov, n_samples)

    # Plot data against true model
    plt.plot(times, data, "o")
    plt.plot(times, forward_model(times))

    # plot the output from the sampled parameters
    for sample in samples:
        plt.plot(times, sample[0] + sample[1]*times, color="grey", alpha=0.1)

    if args.plot:
        plt.show()
    else:
        plt.savefig(os.path.join(
            args.output, "simple_example", "synthetic_data"))

    # Plot 1 s.d ellipse
    # Does this plot make sense? What does it mean?
    fig, ax = cov_ellipse(cov, offset=inferred_params,
                          q=[0.75, 0.9, 0.95, 0.99])
    plt.xlabel("intercept")
    plt.ylabel("gradient")

    ax.plot(*samples.T, "x", color="grey", alpha=0.25)
    ax.legend()
    if args.plot:
        plt.show()
    else:
        plt.savefig(os.path.join(args.output, "simple_example",
                                 "1sd_ellipse_parameter_dist"))


if __name__ == "__main__":
    main()
