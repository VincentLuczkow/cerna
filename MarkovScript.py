#!/usr/bin/python3

from sys import argv
from getpass import getuser
import pickle
from pdb import set_trace

from ceRNA import Network, RateTests


def main():
    if argv[1] == "averages":
        get_averages()
        exit(0)

    x = int(argv[2])
    y = int(argv[3])

    if argv[1] == "run":
        run_tests(x, y, argv[4], int(argv[5]), int(argv[6]))
    elif argv[1] == "post":
        read_files(x, y, argv[4])
    elif argv[1] == "create":
        ks, mus, gammas = Network.Network.generate_parameters(x, y)
        network = Network.Network("Model", ks, mus, gammas, x, y)
        pickle.dump(network, open("/home/{0}/Stochpy/Network{1},{2}".format(getuser(), x, y), "wb"))

    return 0


def setup(x: int, y: int):
    ks, mus, gammas = Network.Network.generate_parameters(x, y)
    network = Network.Network("Model", ks, mus, gammas, x, y)
    file_name = "/home/{0}/Stochpy/Network{1},{2}".format(getuser(), x, y)
    network_file = open(file_name, "wb")
    pickle.dump(network, network_file)


def run_tests(x: int, y: int, test_type: str, start: int, stop: int):
    path_to_stochpy = "/home/{0}/Stochpy/".format(getuser())
    network_file_name = path_to_stochpy + "Network{0},{1}".format(x, y)
    network = pickle.load(open(network_file_name, "rb"))
    if test_type == "k":
        test = network.knockout_test
    elif test_type == "g":
        test = network.gamma_test
    elif test_type == "l":
        test = network.lambda_test
        stop *= 2
    else:
        test = network.wild_type_test

    test.run_simulations(start, stop)

    return 0


def read_files(x: int, y: int, test_type: str):
    path_to_stochpy = "/home/{0}/Stochpy/".format(getuser())
    network_file_name = path_to_stochpy + "Network{0},{1}".format(x, y)
    network = pickle.load(open(network_file_name, "rb"))

    if test_type == "g":
        test = network.gamma_test
        test.collate_sim_data()
    elif test_type == "l":
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
    record_rate_data(test, test_type)
    if test_type in ["g", "l"]:
        record_estimate_data(test, test_type, x, y)


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

    rate_data_file = open("/home/{0}/Stochpy/SimulationData".format(getuser()), "wb")
    pickle.dump(rate_data, rate_data_file)


def record_estimate_data(estimate: RateTests.Estimator, test_type: str, x: int, y: int):
    size = estimate.estimate_size
    estimate_data = retrieve_data()[1]
    key = "BaseNet{0},{1}{2}Estimates".format(x, y, test_type)
    estimate_data[key] = (size, estimate.estimates, estimate.average_relative_error, estimate.average_rmse)
    estimate_data_file = open("/home/{0}/Stochpy/EstimateData".format(getuser()), "wb")
    pickle.dump(estimate_data, estimate_data_file)


def retrieve_data() -> dict:
    try:
        rate_data_file = open("/home/{0}/Stochpy/SimulationData".format(getuser()), "rb")
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


def get_averages():
    estimate_data_file = open("/home/{0}/Stochpy/EstimateData".format(getuser()), "rb")
    estimate_data = pickle.load(estimate_data_file)
    number_of_estimates = len(estimate_data)
    total_rmse = 0
    total_re = 0
    set_trace()
    for estimate in estimate_data:
        estimate_re = estimate_data[estimate][2]["sim"]
        estimate_rmse = estimate_data[estimate][3]["sim"]
        total_re += estimate_re
        total_rmse += estimate_rmse
    average_re = total_re / number_of_estimates
    average_rmse = total_rmse / number_of_estimates
    print(average_re)
    print(average_rmse)


if __name__ == "__main__":
    main()