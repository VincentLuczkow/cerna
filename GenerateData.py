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
        current_network = Network.Network(x, y, ks, mus, gammas)
        current_network.add_test("gamma")
        current_network.add_decay_estimator("gamma", test, "ode")
        #set_trace()
        current_network.add_test("lambda")
        current_network.add_decay_estimator("lambda", test, "ode")
        gamma_relative_error += current_network.estimates[("gamma", "ode", test)].average_relative_error
        gamma_rmse += current_network.estimates[("gamma", "ode", test)].average_rmse
        lambda_relative_error += current_network.estimates[("lambda", "ode", test)].average_rmse
        lambda_rmse += current_network.estimates[("lambda", "ode", test)].average_rmse
    gamma_relative_error /= number_of_tests
    gamma_rmse /= number_of_tests
    lambda_relative_error /= number_of_tests
    lambda_rmse /= number_of_tests
    print("Gamma PE: {0}".format(gamma_relative_error))
    print("Gamma RMSE: {0}".format(gamma_rmse))
    print("Lambda PE: {0}".format(lambda_relative_error))
    print("Lambda RMSE: {0}".format(lambda_rmse))


if __name__ == "__main__":
    main()
