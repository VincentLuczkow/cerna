#!/usr/bin/python3

import random
from ceRNA import Networks
from pdb import set_trace


def main():
    # test_type = argv[1]
    gamma_relative_error = 0
    gamma_rmse = 0
    lambda_relative_error = 0
    lambda_rmse = 0
    number_of_tests = 1
    for test in range(number_of_tests):
        x = random.randrange(5, 25)
        y = random.randrange(5, 25)

        # Set up network
        ks, mus, gammas = Networks.Network.generate_parameters(x, y)
        current_network = Networks.Network(x, y, ks, mus, gammas)
        current_network.add_test("gamma")
        current_network.add_simple_decay_estimator("gamma", "ode", test)
        gamma_relative_error += current_network.common_decay_estimators["gamma"]["ode"][test][1].average_relative_error
        gamma_rmse += current_network.common_decay_estimators["gamma"]["ode"][test][1].average_rmse

    gamma_relative_error /= number_of_tests
    gamma_rmse /= number_of_tests
    print("Gamma PE: {0}".format(gamma_relative_error))
    print("Gamma RMSE: {0}".format(gamma_rmse))


if __name__ == "__main__":
    main()
