import random
from copy import deepcopy
from pdb import set_trace
from typing import Dict, List, Callable, Tuple
import scipy.stats as stats
import numpy as np
from itertools import product

import ceRNA.TestTypes as TestTypes
import ceRNA.Estimators as Estimators
import ceRNA.NetworkFiles as NetworkFiles
import ceRNA.Simulate as Simulate
import ceRNA.Constants as Constants


class Network:

    test_types_mapping = {
        Constants.RateTests.WILD.value: TestTypes.WildTypeTest,
        Constants.RateTests.GAMMA.value: TestTypes.GammaTest,
        Constants.RateTests.LAMBDA.value: TestTypes.LambdaTest,
        Constants.RateTests.KNOCKOUT.value: TestTypes.KnockoutTest
    }                                                                               # type: Dict[str, RateTests.RateTest]

    def __init__(self, x: int, y: int, ks: np.ndarray, mus: np.ndarray, gammas: dict):
        self.x = x
        self.y = y
        self.n = x + y
        self.ks = ks
        self.mus = mus
        self.gammas = gammas

        self.deterministic_solver = self.get_deterministic_solver()
        self.file_writer = self.get_file_writer()

        # A vector containing (k_1,...,mu_1...), i.e. the true birth and death rates for the network.
        self.real_vector = np.concatenate((self.ks, self.mus), axis=0)
        self.rate_tests = {
            Constants.RateTests.WILD.value: [],
            Constants.RateTests.GAMMA.value: [],
            Constants.RateTests.LAMBDA.value: [],
            Constants.RateTests.KNOCKOUT.value: []
        }
        # type: Dict[str: List[RateTests.RateTest]]
        self.full_estimators = []
        # type: List[Tuple[List[Tuple[str, str, int, int, int]], Estimators.Estimator]]
        self.decay_estimators = []
        # type: List[Tuple[List[Tuple[str, str, int, int, int]], Estimators.Estimator]]
        self.first_unsimulated_network = 0

        #self.gamma_tests = {"sim": [], "ode": []}
        #self.lambda_tests = {"sim": [], "ode": []}\

        self.common_decay_estimators = {Constants.RateTests.GAMMA.value: {"sim": [], "ode": []},
                                        Constants.RateTests.LAMBDA.value: {"sim": [], "ode": []}}
        # type: Dict[str, List[Tuple[Tuple[str, str, int, int, int], Estimators.Estimator]]]
        self.common_full_estimators = {"kg": [], "kl": []}
        # type: Dict[str, List[Tuple[Tuple[str, str, int, int, int], Estimators.Estimator]]]

    # Returns a function that solves  the coupled equations d/dt <m_i> for all species i.
    def get_deterministic_solver(self) -> Callable[..., type(None)]:
        return Simulate.base_network_ode_solution

    # Return a function which writes a description of a network to a file.
    # Used to generate psc files for simulating a rate test.
    def get_file_writer(self) -> Callable[..., type(None)]:
        return NetworkFiles.get_base_network_writer(self.x, self.y)

    # Adds a test to the current network.
    def add_test(self, test_type: str, simulation_type: str="ode"):
        rate_test_class = TestTypes.test_types_mapping[test_type]
        rate_test = rate_test_class(self.x, self.y, self.ks, self.mus, self.gammas,
                                    self.deterministic_solver,
                                    self.file_writer)  # type: TestTypes.RateTest
        # rate_test.setup()
        rate_test.run_deterministic_test()
        self.rate_tests[test_type].append(rate_test)

    # Takes a set of test runs, and combines some of the runs from each test in a specified way.
    def combine_tests(self, number_of_rows: int, tests_to_use: List[Tuple[str, str, int, int, int]]) -> np.ndarray:
        combined_tests = np.empty([number_of_rows, 2 * self.n])
        current_index = 0
        for (test_type, simulation_type, test_index, start_row, end_row) in tests_to_use:
            next_start_index = current_index + (end_row - start_row)
            rate_test = self.rate_tests[test_type][test_index]
            rate_test_rows = rate_test.prepare_rows_for_matrix(simulation_type, start_row, end_row)
            combined_tests[current_index:next_start_index] = rate_test_rows
            current_index = next_start_index
        return combined_tests

    def add_full_estimator(self, number_of_rows: int, tests_to_use: List[Tuple[str, str, int, int, int]]) -> Estimators.Estimator:

        combined_tests = self.combine_tests(number_of_rows, tests_to_use)

        estimator = Estimators.Estimator(self.real_vector, combined_tests)
        estimator.calculate_accuracy()
        self.full_estimators.append((tests_to_use, estimator))
        return estimator

    # Takes a set of tests, and for each of those a set of rows as input.
    # Creates a decay estimator using the selected rows from each test.
    # Returns the estimator and adds it to the Networks set of decay estimators.
    # Number of t
    def add_decay_estimator(self, number_of_rows: int, tests_to_use: List[Tuple[str, str, int, int, int]]) -> Estimators.DecayEstimator:

        combined_tests = self.combine_tests(number_of_rows, tests_to_use)

        decay_estimator = Estimators.DecayEstimator(self.real_vector[self.n:], combined_tests)
        decay_estimator.calculate_accuracy()
        self.decay_estimators.append((tests_to_use, decay_estimator))
        return decay_estimator

    # TODO: Testing
    # Creates a decay estimator using a single run of a single RateTest.
    def add_simple_decay_estimator(self, test_type: str, simulation_type: str, test_index: int) -> Estimators.DecayEstimator:
        number_of_rows = 2 * self.n
        tests_to_use = [(test_type, simulation_type, test_index, 0, 2 * self.n)]
        decay_estimator = self.add_decay_estimator(number_of_rows, tests_to_use)
        self.common_decay_estimators[test_type][simulation_type].append((tests_to_use, decay_estimator))
        return decay_estimator

    # Runs the specified number of simulations
    def run_simulations(self, test_type: str, network_to_simulate: int=-1):
        if network_to_simulate == -1:
            network_to_simulate = self.first_unsimulated_network
        # While the networks being worked with have already been simulated
        self.rate_tests[test_type][network_to_simulate].run_simulations()

    # Generates
    @staticmethod
    def generate_parameters(x: int, y: int):
        n = x + y
        gammas = {}

        ks = np.random.normal(10, 0.5, size=n)
        mus = np.random.normal(10, 0.5, size=n)

        for i in range(x):
            mus[i] = np.random.normal(10, 0.5)
            # mus[i] = 5
            gammas[i] = {}
            for j in range(x, n):
                value = np.random.normal(1.25, 0.1)
                # value = 2.6
                gammas[i][j] = value
                if j not in gammas:
                    gammas[j] = {}
                gammas[j][i] = value

        #  Remove some gamma values
        legal = False
        while not legal:
            new_gammas = Network.cull_gammas(x, y, deepcopy(gammas))
            legal = Network.check_network_legality(x, y, new_gammas)

        gammas = Network.generate_gammas(x, y)

        return ks, mus, gammas

    @staticmethod
    def create_test_containers() -> Dict:
        tests = {
            "wild" : {"ode": [], "sim": []},
            "gamma": {"ode": [], "sim": []},
            "lambda": {"ode": [], "sim": []},
            "knockout": {"ode": [], "sim": []}
        }
        return tests

    @staticmethod


    # Generate gammas from the given distribution.
    @staticmethod
    def generate_gammas(x: int, y: int,
                        distribution: stats.rv_continuous=stats.truncnorm,
                        parameters: np.ndarray=np.array([5, 15])) -> np.ndarray:
        gammas = np.zeros([x+y, x+y])
        mRNA_to_miRNA = np.zeros([x, y])
        for row in range(x):
            mRNA_to_miRNA[row] = distribution.rvs(*parameters, size=y)

        #  Remove some gamma values
        legal = False
        while not legal:
            culled_gammas = Network.new_cull_gammas(x, y, deepcopy(mRNA_to_miRNA))
            legal = Network.new_check_network_legality(x, y, culled_gammas)

        gammas[:x, x:] = culled_gammas
        gammas[x:, :x] = culled_gammas.T

        return gammas

    # Generates gamma parameters for models with very large gamma values.
    @staticmethod
    def generate_large_gamma_parameters(x, y):
        n = x + y
        gammas = {}
        ks = np.zeros(x + y)
        mus = np.zeros(x + y)

        for i in range(x):
            ks[i] = np.random.normal(10, 0.5)
            # ks[i] = 6
            mus[i] = np.random.normal(10, 0.5)
            # mus[i] = 5
            gammas[i] = {}
            for j in range(x, n):
                value = np.random.normal(125, 10)
                # value = 2.6
                gammas[i][j] = value
                if j not in gammas:
                    gammas[j] = {}
                gammas[j][i] = value

        for i in range(x, n):
            ks[i] = np.random.normal(10, 0.5)
            # ks[i] = 5
            mus[i] = np.random.normal(10, 0.5)
        # mus[i] = 5

        #  Remove some gamma values
        legal = False
        while not legal:
            new_gammas = Network.cull_gammas(x, y, deepcopy(gammas))
            legal = Network.check_network_legality(x, y, new_gammas)

        return ks, mus, gammas

    @staticmethod
    def new_cull_gammas(x, y, mRNA_to_miRNA: np.ndarray) -> np.ndarray:
        legal = False
        while not legal:
            temp_rates = deepcopy(mRNA_to_miRNA)
            number_to_cull = int(.1 * x * y)
            gamma_pairs = list(product(range(x), range(y)))
            interactions_to_remove = random.sample(gamma_pairs, number_to_cull)
            for interactions in interactions_to_remove:
                temp_rates[interactions[0]][interactions[1]] = 0.0
            legal = Network.new_check_network_legality(x, y, temp_rates)
        return temp_rates

    @staticmethod
    def new_check_network_legality(x: int, y: int, rates: np.ndarray):
        not_reached = list(range(x+y))
        shadowed = [0]
        while shadowed:
            for node in shadowed:
                shadowed.remove(node)
                if node >= x:
                    neighbors = np.nonzero(rates[:, node-x])
                else:
                    neighbors = np.add(np.nonzero(rates[node]), x)
                for neighbor in neighbors[0]:
                    if neighbor in not_reached:
                        shadowed.append(neighbor)
                        not_reached.remove(neighbor)
        return len(not_reached) == 0

    @staticmethod
    def cull_gammas(x: int, y: int, gammas: dict) -> dict:
        legal = False
        temp_gammas = {}
        while not legal:
            temp_gammas = deepcopy(gammas)
            gammas_list = []
            for i in range(x):
                for j in range(x, x + y):
                    gammas_list.append((i, j))

            # Choose how many and which connections to cull
            number_to_cull = int(.1 * (x * y))
            connections_to_cull = random.sample(gammas_list, number_to_cull)

            # Cull the chosen connections
            for connection in connections_to_cull:
                del temp_gammas[connection[0]][connection[1]]
                del temp_gammas[connection[1]][connection[0]]
            legal = Network.check_network_legality(x, y, temp_gammas)
        return temp_gammas

    @staticmethod
    def check_network_legality(x: int, y: int, gammas: dict):
        added = []
        shadowed = [0]
        while shadowed:
            for node in shadowed:
                shadowed.remove(node)
                added.append(node)
                for neighbor in gammas[node].keys():
                    if neighbor not in shadowed and neighbor not in added:
                        shadowed.append(neighbor)
        added.sort()
        return added == list(range(x + y))


class BurstNetwork(Network):

    def __init__(self, x: int, y: int, ks: np.ndarray, mus: np.ndarray, gammas: np.ndarray,
                 alphas: np.ndarray, betas: np.ndarray):
        self.alphas = alphas
        self.betas = betas

        # The equilibrium probability of each promoter state being on.
        # vec[i] is the probability that species i is in the on state at equilibrium
        self.equilibrium_probability = self.alphas / (self.alphas + self.betas)

        Network.__init__(self, x, y, ks, mus, gammas)

        # Multiply all birth rates by their equilibrium probabilities. Leave death rates alone.
        np.multiply(self.real_vector, np.concatenate((self.equilibrium_probability, np.ones(self.n))), self.real_vector)

    def get_deterministic_solver(self):
        return Simulate.get_burst_network_solver(self.equilibrium_probability, self.alphas, self.betas)

    def get_file_writer(self):
        return NetworkFiles.get_burst_network_writer(self.x, self.y, self.alphas, self.betas)

    @staticmethod
    def generate_burst_parameters_basic(x, y) -> Tuple[np.ndarray, np.ndarray]:
        alphas = stats.truncnorm.rvs(0, 1, size=(x+y))
        betas = stats.truncnorm.rvs(0, 1, size=(x+y))
        return alphas, betas


class StoichiometricComplexNetwork(BurstNetwork):
    def __init__(self, x: int, y: int, ks: np.ndarray, mus: np.ndarray, gammas: np.ndarray,
                 alphas: np.ndarray, betas: np.ndarray,
                 deltas: np.ndarray, zetas: np.ndarray):
        self.deltas = deltas
        self.zetas = zetas
        BurstNetwork.__init__(self, x, y, ks, mus, gammas, alphas, betas)


    @staticmethod
    def generate_complex_parameters(x, y) -> tuple:
        pass


class CatalyticComplexNetwork(StoichiometricComplexNetwork):
    def __init__(self, x: int, y: int, ks: np.ndarray, mus: np.ndarray, gammas: np.ndarray,
                 alphas: np.ndarray, betas: np.ndarray,
                 deltas: np.ndarray, zetas: np.ndarray,
                 etas: np.ndarray):
        StoichiometricComplexNetwork.__init__(self, x, y, ks, mus, gammas, alphas, betas, deltas, zetas)


network_mapping = {
    Constants.NetworkModels.BASE.value: Network,
    Constants.NetworkModels.BURST.value: BurstNetwork,
    Constants.NetworkModels.SCOM.value: StoichiometricComplexNetwork,
    Constants.NetworkModels.CCOM.value: CatalyticComplexNetwork
}  # type: Dict[int, Network]




