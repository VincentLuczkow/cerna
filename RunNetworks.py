import random
import sys

import numpy as np

import ceRNA.Constants as Constants
import ceRNA.NetworkTypes as NetworkTypes
import ceRNA.NetworkTopologies as NetworkTopologies
import ceRNA.TestTypes as RateTests


def main():
    network_type = int(sys.argv[1])
    network_topology = int(sys.argv[2])
    test_type = int(sys.argv[3])
    run_tests(network_type=network_type,
              network_topology_name=network_topology,
              test_type=Constants.RateTests.LAMBDA.value)
    pass


def record_results(output_file_name: str):
    pass


def run_tests(number_of_networks: int=5,
              record: bool=False,
              network_type: int=Constants.NetworkModels.BASE.value,
              network_topology_name: int=Constants.NetworkTopologies.SIMPLE.value,
              test_type: int=Constants.RateTests.GAMMA.value,
              output_file_name: str= ""):

    relative_error = np.zeros(number_of_networks)
    for test in range(number_of_networks):
        print("Test {}".format(test))
        x = random.randrange(5, 20)
        y = random.randrange(5, 20)

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