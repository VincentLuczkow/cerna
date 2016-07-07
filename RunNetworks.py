import random
import sys

import numpy as np

import ceRNA.Constants as Constants
import ceRNA.Networks as Networks
import ceRNA.NetworkTopologies as NetworkTopologies


def main():
    run_tests(test_type=Constants.RateTests.LAMBDA)
    pass


def record_results(output_file_name: str):
    pass


def run_tests(number_of_networks: int=50,
              record: bool=False,
              network_type: str=Constants.NetworkModels.BASE,
              network_topology_name: int=Constants.NetworkTopologies.SIMPLE,
              test_type: str=Constants.RateTests.GAMMA,
              output_file_name: str= ""):

    relative_error = np.zeros(number_of_networks)
    for test in range(number_of_networks):
        print("Test {}".format(test))
        x = random.randrange(5, 25)
        y = random.randrange(5, 25)

        # Set up network
        network_topology = NetworkTopologies.network_topology_mapping[network_topology_name]
        current_network = network_topology(x, y, network_type)
        current_network.add_test(test_type)
        current_network.add_simple_decay_estimator(test_type, "ode", 0)
        relative_error[test] = current_network.common_decay_estimators[test_type]["ode"][0][1].average_relative_error
        if record:
            record_results(output_file_name)

    if not record:
        average_relative_error = np.mean(relative_error)
        print("Average relative error: {}".format(average_relative_error))


# Simulates the specified network.
def simulate_network(x: int, y: int, test_type: str, number_of_tests: int):
    pass


if __name__ == "__main__":
    main()