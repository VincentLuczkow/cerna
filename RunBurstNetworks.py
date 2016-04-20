#!/usr/bin/python3

import random
from ceRNA import Network
from pdb import set_trace


def main():
    # test_type = argv[1]
    gamma_relative_error = 0
    gamma_rmse = 0
    lambda_relative_error = 0
    lambda_rmse = 0
    number_of_tests = 1
    for test in range(number_of_tests):
        x = random.randrange(5, 10)
        y = random.randrange(5, 10)

        # Set up network
        ks, mus, gammas = Network.Network.generate_parameters(x, y)
        alphas, betas = Network.BurstNetwork.generate_burst_parameters_basic(x, y)
        current_network = Network.BurstNetwork(x, y, ks, mus, gammas, alphas, betas)
        current_network.add_test("wild")
        current_network.add_test("lambda")
        current_network.add_test("gamma")
        current_network.add_test("knockout")
        current_network.add_simple_decay_estimator("gamma", "ode", test)
        current_network.add_simple_decay_estimator("lambda", "ode", test)

    gamma_relative_error /= number_of_tests
    gamma_rmse /= number_of_tests
    #lambda_relative_error /= number_of_tests
    #lambda_rmse /= number_of_tests
    print("Gamma PE: {0}".format(gamma_relative_error))
    print("Gamma RMSE: {0}".format(gamma_rmse))
    #print("Lambda PE: {0}".format(lambda_relative_error))
    #print("Lambda RMSE: {0}".format(lambda_rmse))


if __name__ == "__main__":
    main()
