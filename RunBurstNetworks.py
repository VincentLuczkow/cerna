#!/usr/bin/python3

import random
from ceRNA import Networks
from pdb import set_trace
from sys import argv


def main():
    gamma_relative_error = 0
    gamma_sim_re = 0
    gamma_rmse = 0
    number_of_tests = 1
    for test in range(number_of_tests):
        x = random.randrange(5, 10)
        y = random.randrange(5, 10)

        # Set up network
        ks, mus, gammas = Networks.Network.generate_parameters(x, y)
        alphas, betas = Networks.BurstNetwork.generate_burst_parameters_basic(x, y)
        current_network = Networks.BurstNetwork(x, y, ks, mus, gammas, alphas, betas)
        current_network.add_test("gamma")
        current_network.add_simple_decay_estimator("gamma", "ode", test)
        gamma_relative_error += current_network.common_decay_estimators["gamma"]["ode"][test][1].average_relative_error
        gamma_rmse += current_network.common_decay_estimators["gamma"]["ode"][test][1].average_rmse

    gamma_sim_re /= number_of_tests
    gamma_relative_error /= number_of_tests
    gamma_rmse /= number_of_tests
    print("Gamma PE: {0}".format(gamma_relative_error))
    print("Gamma RMSE: {0}".format(gamma_rmse))
    print("{0}".format(gamma_sim_re))


if __name__ == "__main__":
    main()
