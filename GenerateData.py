#!/usr/bin/python3

import random
from ceRNA import Network

def main():
    # test_type = argv[1]
    gamma_pe = 0
    gamma_rmse = 0
    lambda_pe = 0
    lambda_rmse = 0
    number_of_tests = 1
    for test in range(number_of_tests):
        x = random.randrange(5, 10)
        y = random.randrange(5, 10)

        # Set up network
        ks, mus, gammas = Network.Network.generate_parameters(x, y)
        current_network = Network.Network(x, y, ks, mus, gammas)
        gamma_pe += current_network.gamma_test.average_relative_error["ode"]
        gamma_rmse += current_network.gamma_test.average_rmse["ode"]
        lambda_pe += current_network.lambda_test.average_relative_error["ode"]
        lambda_rmse += current_network.lambda_test.average_rmse["ode"]
    gamma_pe /= number_of_tests
    gamma_rmse /= number_of_tests
    lambda_pe /= number_of_tests
    lambda_rmse /= number_of_tests
    print("Gamma PE: {0}".format(gamma_pe))
    print("Gamma RMSE: {0}".format(gamma_rmse))
    print("Lambda PE: {0}".format(lambda_pe))
    print("Lambda RMSE: {0}".format(lambda_rmse))


if __name__ == "__main__":
    main()
