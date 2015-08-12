#!/usr/bin/python3

from sys import argv
from getpass import getuser
import pickle

from ceRNA import Network, RateTests


def run_tests(x: int, y: int, test_type: str, start: int, stop: int):
    path_to_stochpy = "/home/{0}/Stochpy/".format(getuser())
    network_file_name = path_to_stochpy + "Network{0},{1}".format(x, y)
    network = pickle.load(open(network_file_name, "rb"))
    if test_type == "k":
        test = network.knockout_test
    elif test_type == "g":
        test = network.gamma_test
    else:
        test = network.lambda_test
        stop *= 2

    test.run_simulations(start, stop)

    return 0


def read_files(x: int, y: int, test_type: str):
    path_to_stochpy = "/home/{0}/Stochpy/".format(getuser())
    network_file_name = path_to_stochpy + "Network{0},{1}".format(x, y)
    network = pickle.load(open(network_file_name, "rb"))

    if test_type == "g":
        test = network.gamma_test
        test.collate_sim_data()
    elif test_type == "k":
        test = network.lambda_test
        test.collate_sim_data()
    else:
        if test_type == "kl":
            test = network.knockout_lambda_test
            knockout = network.select_usable_knockouts("sim")
            decay = network.lambda_test.tests["sim"]
        else:
            test = network.knockout_gamma_test
            knockout = network.knockout_test.tests["sim"]
            decay = network.gamma_test.tests["sim"]
        wild_type = network.wild_type_test.tests["sim"]
        test.set_matrix(wild_type, knockout, decay, "sim")

    test.post_processing("sim")
    test.print_results("sim")

    pickle.dump(network, open(network_file_name, "wb"))


def retrieve_data() -> dict:
    try:
        rate_data_file = open("/home/{0}/Stochpy/AggregateData".format(getuser()), "rb")
        rate_data = pickle.load(rate_data_file)
        rate_data_file.close()
    except IOError:
        rate_data = {}
    try:
        estimate_data_file = open("/home/{0}/Stochpy/EstimateData".format(getuser()), "rb")
        estimate_data = pickle.load(estimate_data_file)
        estimate_data_file.close()
    except IOError:
        estimate_data = {}
    return rate_data, estimate_data


def setup(x: int, y: int):
    ks, mus, gammas = Network.Network.generate_parameters(x, y)
    network = Network.Network("Model", ks, mus, gammas, x, y)
    file_name = "/home/{0}/Stochpy/Network{1},{2}".format(getuser(), x, y)
    network_file = open(file_name, "wb")
    pickle.dump(network, network_file)


def record_rate_data(rate_test: RateTests.RateTest, test_type: str):
    x = rate_test.x
    y = rate_test.y
    rate_data = retrieve_data()[0]
    key = "BaseNet{0},{1}{2}Rates".format(x, y, test_type)
    ks = rate_test.ks
    mus = rate_test.mus
    gammas = rate_test.gammas
    tests = rate_test.tests["sim"]
    rate_data[key] = (ks, mus, gammas, tests)

    rate_data_file = open("/home/{0}/Stochpy/AggregateData".format(getuser()), "wb")
    pickle.dump(rate_data, rate_data_file)



def record_estimate_data(estimate: RateTests.Estimator, test_type: str, x: int, y: int):
    size = estimate.estimate_size
    estimate_data = retrieve_data()[1]
    key = "BaseNet{0},{1}{2}Estimates".format(x, y, test_type)
    estimate_data[key] = (size)


def main():
    # Read data files
    x = int(argv[2])
    y = int(argv[3])
    if argv[1] == "r":
        run_tests(x, y, argv[4], int(argv[5]), int(argv[6]))
    if argv[1] == "d":
        read_files(x, y, argv[4])
    if argv[1] == "c":
        ks, mus, gammas = Network.Network.generate_parameters(x, y)
        network = Network.Network("Model", ks, mus, gammas, x, y)
        pickle.dump(network, open("/home/{0}/Stochpy/Network{1},{2}".format(getuser(), x, y), "wb"))

    return 0


if __name__ == "__main__":
    main()
