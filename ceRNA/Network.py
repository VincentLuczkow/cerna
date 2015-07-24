import random
from copy import deepcopy
from pdb import set_trace
import numpy as np

from .Calculations import get_sub_matrix
from .RateTests import CombinedTest, WildTypeTest, GammaTest, LambdaTest, KnockoutTest


class Network:
    def __init__(self, model_name, ks, mus, gammas, x, y):
        self.modelName = model_name
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

        self.knockout_lambda_test = CombinedTest(self.real_vector)
        self.knockout_lambda_test.set_matrix(self.wild_type_test.tests["ode"],
                                             usable_knockout_tests,
                                             self.lambda_test.tests["ode"], "ode")
        self.knockout_lambda_test.post_processing("ode")

        self.knockout_gamma_test = CombinedTest(self.real_vector)
        self.knockout_gamma_test.set_matrix(self.wild_type_test.tests["ode"],
                                            self.knockout_test.tests["ode"],
                                            self.gamma_test.tests["ode"], "ode")

        self.knockout_gamma_test.post_processing("ode")

    def setup(self):
        self.wild_type_test.setup()
        self.gamma_test.setup()
        self.lambda_test.setup()
        self.knockout_test.setup()

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
            ks[i] = random.uniform(10, 50)
            # ks[i] = 6
            mus[i] = np.random.normal(10, 0.5)
            # mus[i] = 5
            gammas[i] = {}
            for j in range(x, n):
                value = np.random.normal(1.25, 0.01)
                # value = 2.6
                gammas[i][j] = value
                if j not in gammas:
                    gammas[j] = {}
                gammas[j][i] = value

        for i in range(x, n):
            ks[i] = random.uniform(10, 50)
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
