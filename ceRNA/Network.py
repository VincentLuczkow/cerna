import random
from copy import deepcopy
from pdb import set_trace
import numpy as np

from .Calculations import get_sub_matrix
from .RateTests import WildTypeTest, GammaTest, LambdaTest, KnockoutTest
from .Estimators import *
from .Simulate import base_network_ode_solution, burst_network_ode_solution

class Network:

    test_types = {
        "wild": WildTypeTest,
        "gamma": GammaTest,
        "lambda": LambdaTest,
        "knockout": KnockoutTest
    }

    def __init__(self, x: int, y: int, ks: np.ndarray, mus: np.ndarray, gammas: dict):
        self.x = x
        self.y = y
        self.n = x + y
        self.ks = ks
        self.mus = mus
        self.gammas = gammas

        n = x + y

        self.wild_type_test = np.ones(2 * n)
        # This section is for calculating accuracy of tests
        self.real_vector = np.concatenate((self.ks, self.mus), axis=0)

        # Core set of tests.
        self.wild_type_test = WildTypeTest(x, y, ks, mus, gammas)
        self.wild_type_test.setup()

        self.knockout_test = KnockoutTest(x, y, ks, mus, gammas)
        self.knockout_test.setup()

        #self.tests = [self.wild_type_test, self.gamma_test, self.lambda_test, self.knockout_test]

        #usable_knockout_tests = self.select_usable_knockouts("ode")

        self.rate_tests = {"wild": [], "gamma": [], "lambda": [], "knockout": []}

        self.deterministic_solver = self.get_deterministic_solver()

        self.estimates = {}

        # TODO : Move imports around, point this to base_network_file_writer.
        self.file_writer = self.get_file_writer()

    def get_deterministic_solver(self):
        return base_network_ode_solution

    # TODO: return the actual file writer
    def get_file_writer(self):
        return 0

    # TODO: Testing
    def add_test(self, test_type: str):
        new_test = self.test_types[test_type](self.x, self.y, self.ks, self.mus, self.gammas, self.real_vector[self.n:],
                                              self.deterministic_solver)
        new_test.setup()
        self.rate_tests[test_type].append(new_test)

    # TODO: Testing
    def add_estimator(self, real_vector: np.ndarray, number_of_tests: int, included_tests: list, estimate_name: str):
        size_of_tests = len(real_vector)
        matrix = np.empty([number_of_tests, 2 * self.n])
        current_index = 0
        for test in included_tests:
            test_type, test_index, simulation_type, start_row, end_row = test
            next_index = current_index + (end_row - start_row)
            rate_test = self.rate_tests[test_type][test_index]
            rate_test_rows = rate_test.prepare_rows_for_matrix(simulation_type, start_row, end_row)
            matrix[current_index:next_index] = rate_test_rows
            current_index = next_index + 1
        set_trace()
        return 0

    # TODO: Testing
    def add_decay_estimator(self, test_type: str, test_index: int, simulation_type:str):
        real_vector = self.real_vector[self.n:]
        number_of_tests = 2 * self.n
        decay_test = (test_type, test_index, simulation_type, 0, number_of_tests)
        included_tests = [decay_test]
        estimate_name = "{0} estimator for {1} test: {2}".format(test_type, simulation_type, test_index)
        decay_tests = self.combine_tests(number_of_tests, included_tests)
        decay_estimator = DecayEstimator(real_vector, decay_tests)
        decay_estimator.calculate_accuracy()
        self.estimates[(test_type, simulation_type, test_index)] = decay_estimator

    def combine_tests(self, number_of_tests: int, included_tests: list):
        combined_tests = np.empty([number_of_tests, 2 * self.n])
        current_index = 0
        for test in included_tests:
            test_type, test_index, simulation_type, start_row, end_row = test
            next_index = current_index + (end_row - start_row)
            rate_test = self.rate_tests[test_type][test_index]
            rate_test_rows = rate_test.prepare_rows_for_matrix(simulation_type, start_row, end_row)
            combined_tests[current_index:next_index] = rate_test_rows

            current_index = next_index + 1
        return combined_tests

    def setup(self):
        self.wild_type_test.setup()
        self.knockout_test.setup()

    def post_processing(self):
        pass

    def select_usable_knockouts(self, knockout_test_index: int, lambda_test_index: int, run_type: str) -> np.ndarray:
        column_map = range(2 * self.n)
        knockout_tests = self.rate_tests["knockout"][knockout_test_index].tests[run_type]
        lambda_test = self.rate_tests["lambda"][lambda_test_index]
        if lambda_test.knockout != self.n - 1:
            row_map = range(self.n - 1)
        else:
            row_map = dict(zip(list(range(1, self.n)), list(range(self.n - 1))))
        usable_knockout_tests = get_sub_matrix(row_map, column_map, knockout_tests)
        return usable_knockout_tests


    # TODO: Add function generate_gamma_parameters
    # TODO: Testing
    @staticmethod
    def generate_simple_parameters(x: int, y: int,
                                     m_mean: float=10, m_var: float=0.5, s_mean: float=10, s_var: float=0.5) -> np.ndarray:
        ks = np.zeros(x + y)
        for mRNA in range(x):
            ks[mRNA] = np.random.normal(m_mean, m_var)
        for miRNA in range(x, x + y):
            ks[miRNA] = np.random.normal(s_mean, s_var)

        return ks

    # Generates
    @staticmethod
    def generate_parameters(x, y):
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

    def add_test(self, test_type: str):
        pass

    def add_estimator(self, real_vector: np.ndarray, number_of_tests: int, included_tests: list, estimate_name: str):
        return super().add_estimator(real_vector, number_of_tests, included_tests, estimate_name)

    @staticmethod
    def generate_burst_parameters(x, y) -> tuple:
        pass


class ComplexNetwork(BurstNetwork):
    def __init__(self, x: int, y: int, ks: np.ndarray, mus: np.ndarray, gammas: np.ndarray,
                 alphas: np.ndarray, betas: np.ndarray):
        self.alphas = alphas
        self.betas = betas
        BurstNetwork.__init__(self, x, y, ks, mus, gammas)

    def add_test(self, test_type: str):
        pass

    @staticmethod
    def generate_complex_parameters(x, y) -> tuple:
        pass
