import random
from copy import deepcopy
from pdb import set_trace
from typing import Dict, List, Callable, Tuple
import numpy as np

from .Calculations import get_sub_matrix
import ceRNA.RateTests as RateTests
import ceRNA.Estimators as Estimators
import ceRNA.NetworkFiles as NetworkFiles
from .Simulate import base_network_ode_solution, burst_network_ode_solution


class Network:

    test_types_mapping = {
        "wild": RateTests.WildTypeTest,
        "gamma": RateTests.GammaTest,
        "lambda": RateTests.LambdaTest,
        "knockout": RateTests.KnockoutTest
    } # type: Dict[str, RateTests.RateTest]

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

        self.full_estimators = []  # type: List[Tuple[]]
        self.decay_estimators = []

        self.rate_tests = {"wild": [], "gamma": [], "lambda": [], "knockout": []}  # type: Dict[str: List[RateTests.RateTest]]

        self.simple_decay_estimators = {}

    # Returns a function that solves  the coupled equations d/dt <m_i> for all species i.
    def get_deterministic_solver(self):
        return base_network_ode_solution

    # TODO: Testing
    def get_file_writer(self) -> Callable[..., type(None)]:
        return NetworkFiles.get_base_network_writer(self.x, self.y)

    # TODO: Testing
    def add_test(self, test_type_name: str):
        test_type = self.test_types_mapping[test_type_name]
        new_test = test_type(self.x, self.y, self.ks, self.mus, self.gammas, self.real_vector[self.n:],
                             self.deterministic_solver)
        new_test.setup()
        self.rate_tests[test_type_name].append(new_test)

    # TODO: Testing
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

    # TODO: Testing
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
        self.decay_estimators.append((decay_estimator, tests_to_use))
        return decay_estimator

    # TODO: Testing
    # Creates a decay estimator using a single run of a single RateTest.
    def add_simple_decay_estimator(self, test_type: str, simulation_type: str, test_index: int):
        number_of_rows = 2 * self.n
        tests_to_use = [(test_type, simulation_type, test_index, 0, 2 * self.n)]
        decay_estimator = self.add_decay_estimator(number_of_rows, tests_to_use)

    def setup(self):
        self.wild_type_test.setup()
        self.knockout_test.setup()

    # TODO: Testing
    # Generates both arrival and decay rate parameters.
    @staticmethod
    def generate_arrival_and_decay_parameters(x: int, y: int,
                                              m_mean: float=10,
                                              m_var: float=0.5,
                                              s_mean: float=10,
                                              s_var: float=0.5) -> np.ndarray:
        parameters = np.zeros(x + y)
        parameters[:x] = np.random.normal(m_mean, m_var, x)
        parameters[:y] = np.random.normal(s_mean, s_var, y)

        return parameters

    # TODO: Testing
    # Generates interaction rate parameters
    @staticmethod
    def generate_gamma_parameters(x: int, y: int):
        gammas = {}

        for i in range(x):
            gammas[i] = {}
            for j in range(x, x+y):
                value = np.random.normal(1.25, 0.1)
                # value = 2.6
                gammas[i][j] = value
                if j not in gammas:
                    gammas[j] = {}
                gammas[j][i] = value

        return gammas

    # Generates
    @staticmethod
    def generate_parameters(x: int, y: int):
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
                value = np.random.normal(1.25, 0.1)
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
    def create_test_containers() -> Dict:
        tests = {}
        tests["wild"] = {"ode": [], "sim": []}
        tests["gamma"] = {"ode": [], "sim": []}
        tests["lambda"] = {"ode": [], "sim": []}
        tests["knockout"] = {"ode": [], "sim": []}

        return tests

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
    def cull_gammas(x, y, gammas):
        gammas_list = []
        for i in range(x):
            for j in range(x, x + y):
                gammas_list.append((i, j))

        # Choose how many and which connections to cull
        number_to_cull = int(.1 * (x * y))
        connections_to_cull = random.sample(gammas_list, number_to_cull)

        # Cull the chosen connections
        for connection in connections_to_cull:
            del gammas[connection[0]][connection[1]]
            del gammas[connection[1]][connection[0]]
        return gammas

    @staticmethod
    def check_network_legality(x, y, gammas):
        added = []
        shadowed = [0]
        while shadowed:
            for node in shadowed:
                shadowed.remove(node)
                added.append(node)
                for neighbor in gammas[node].keys():
                    if not neighbor in shadowed and not neighbor in added:
                        shadowed.append(neighbor)
        added.sort()
        return added == list(range(x + y))


class BurstNetwork(Network):

    def __init__(self, x: int, y: int, ks: np.ndarray, mus: np.ndarray, gammas: np.ndarray,
                 alphas: np.ndarray, betas: np.ndarray):
        self.alphas = alphas
        self.betas = betas
        Network.__init__(self, x, y, ks, mus, gammas)

    def get_deterministic_solver(self):
        return base_network_ode_solution

    # TODO: return the actual file writer
    def get_file_writer(self):
        return NetworkFiles.get_burst_network_writer(self.x, self.y, self.alphas, self.betas)

    def add_test(self, test_type_name: str):
        pass

    def add_full_estimator(self, real_vector: np.ndarray, number_of_rows: int, tests_to_use: list, estimate_name: str):
        return super().add_full_estimator(real_vector, number_of_rows, tests_to_use, estimate_name)

    @staticmethod
    def generate_burst_parameters(x, y) -> tuple:
        pass


class ComplexNetwork(BurstNetwork):
    def __init__(self, x: int, y: int, ks: np.ndarray, mus: np.ndarray, gammas: np.ndarray,
                 alphas: np.ndarray, betas: np.ndarray):
        self.alphas = alphas
        self.betas = betas
        BurstNetwork.__init__(self, x, y, ks, mus, gammas)

    def add_test(self, test_type_name: str):
        pass

    @staticmethod
    def generate_complex_parameters(x, y) -> tuple:
        pass
