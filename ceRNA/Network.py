import random
from copy import deepcopy
from pdb import set_trace
import numpy as np

from .Calculations import get_sub_matrix
from .RateTests import WildTypeTest, GammaTest, LambdaTest, KnockoutTest
from .Estimators import CombinedTest


class Network:
    def __init__(self, x, y, ks, mus, gammas):
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

        self.gamma_test = GammaTest(x, y, self.real_vector[n:], ks, mus, gammas)
        self.gamma_test.setup()
        self.gamma_test.post_processing("ode")

        self.lambda_test = LambdaTest(x, y, self.real_vector[n:], ks, mus, gammas)
        self.lambda_test.setup()
        self.lambda_test.post_processing("ode")

        self.knockout_test = KnockoutTest(x, y, ks, mus, gammas)
        self.knockout_test.setup()

        self.tests = [self.wild_type_test, self.gamma_test, self.lambda_test, self.knockout_test]

        usable_knockout_tests = self.select_usable_knockouts("ode")

        self.knockout_lambda_estimator = CombinedTest(self.real_vector)
        self.knockout_lambda_estimator.set_matrix(self.wild_type_test.tests["ode"],
                                             usable_knockout_tests,
                                             self.lambda_test.tests["ode"], "ode")
        self.knockout_lambda_estimator.post_processing("ode")

        self.knockout_gamma_estimator = CombinedTest(self.real_vector)
        self.knockout_gamma_estimator.set_matrix(self.wild_type_test.tests["ode"],
                                            self.knockout_test.tests["ode"],
                                            self.gamma_test.tests["ode"], "ode")

        self.knockout_gamma_estimator.post_processing("ode")

        self.rate_tests = {"wild": [], "gamma": [], "lambda": [], "knockout": []}
        self.estimates = {}
        self.wild_type_tests = []
        self.gamma_tests = []
        self.lambda_tests = []
        self.knockout_tests = []

    # TODO
    def add_test(self, test_type: str):
        if test_type == "wild":
            new_test = WildTypeTest(self.x, self.y, self.ks, self.mus, self.gammas)
        elif test_type == "gamma":
            new_test = GammaTest(self.x, self.y, self.ks, self.mus, self.gammas)
        elif test_type == "lambda":
            new_test = LambdaTest(self.x, self.y, self.ks, self.mus, self.gammas)
        elif test_type == "knockout":
            new_test = KnockoutTest(self.x, self.y, self.ks, self.mus, self.gammas)
        self.rate_tests[test_type].append(new_test)

    # TODO
    def add_estimator(self, real_vector: np.ndarray, number_of_tests: int, included_tests: list, estimate_name: str):
        size_of_tests = len(real_vector)
        matrix = np.empty([number_of_tests, size_of_tests])
        current_index = 0
        for test in included_tests:
            test_type, test_index, simulation_type, start_row, end_row = test
            next_index = current_index + (end_row - start_row)
            rate_test = self.rate_tests[test_type][test_index]
            rate_test_rows = rate_test.prepare_rows_for_matrix()
            matrix[current_index:next_index] = rate_test_rows
            current_index = next_index + 1
        self.estimates[estimate_name] = 5
        return 0

    def setup(self):
        self.wild_type_test.setup()
        self.gamma_test.setup()
        self.lambda_test.setup()
        self.knockout_test.setup()

    def post_processing(self):
        self.gamma_test.post_processing("ode")
        self.lambda_test.post_processing("ode")
        self.knockout_gamma_estimator.post_processing("ode")
        self.knockout_lambda_estimator.post_processing("ode")

    def select_usable_knockouts(self, run_type) -> np.ndarray:
        column_map = range(2 * self.n)
        tests = self.knockout_test.tests[run_type]
        if self.lambda_test.knockout != self.n - 1:
            row_map = range(self.n - 1)
        else:
            row_map = dict(zip(list(range(1, self.n)), list(range(self.n - 1))))
        usable_knockout_tests = get_sub_matrix(row_map, column_map, tests)
        return usable_knockout_tests

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


class ComplexNetwork(BurstNetwork):
    def __init__(self, x: int, y: int, ks: np.ndarray, mus: np.ndarray, gammas: np.ndarray,
                 alphas: np.ndarray, betas: np.ndarray):
        self.alphas = alphas
        self.betas = betas
        BurstNetwork.__init__(self, x, y, ks, mus, gammas)

