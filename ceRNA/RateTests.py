from copy import deepcopy
from getpass import getuser
from itertools import combinations
from multiprocessing import Process
import pickle
import random
from pdb import set_trace

import numpy as np

from .Simulate import base_network_ode_solution, simulate
from .NetworkFiles import create_base_network_file
from .Estimators import Estimator


class RateTest:
    test_type = ""
    path_to_stochpy_folder = "/home/" + getuser() + "/Stochpy/"

    def __init__(self, x: int, y: int,
                 ks: np.ndarray,
                 mus: np.ndarray,
                 gammas: dict,
                 model_type: int,
                 alphas: np.ndarray=None,
                 betas: np.ndarray=None):
        # Number of mRNAs
        self.x = x
        # Number of miRNAs
        self.y = y
        self.n = x + y
        # Creation rates
        self.ks = ks
        # Decay Rates
        self.mus = mus
        # Interaction Rates
        self.gammas = gammas
        # Burst On Rate
        self.alphas = alphas
        # Burst Off Rate
        self.betas = betas
        self.number_of_tests = 0
        self.number_of_changes = 0
        self.tests = {}
        self.row_accuracies = {}
        self.average_row_accuracy = {}
        self.psc_file_names = []
        self.result_file_names = []
        self.change_sets = []

    def setup(self):
        self.tests["ode"] = self.mean_field_solutions()
        self.psc_file_names, self.result_file_names = self.create_psc_files()

    def generate_changes(self, change_type: int=0):
        return None, None

    def create_psc_files(self) -> tuple:
        psc_files = []
        results_files = []
        base_file_name = "BaseNet{0},{1}{2}Test".format(self.x, self.y, self.test_type)

        for change in range(self.number_of_changes):
            change_file_name = base_file_name + str(change)
            new_ks, new_gammas = self.change_sets[change]
            if new_ks is None:
                new_ks = self.ks
            if new_gammas is None:
                new_gammas = self.gammas

            create_base_network_file(self.x, self.y, new_ks, self.mus, new_gammas, change_file_name)
            psc_files.append(change_file_name)

        for test in range(self.number_of_tests):
            results_file_name = RateTest.path_to_stochpy_folder + base_file_name + str(test) + "Results"
            results_files.append(results_file_name)

        return psc_files, results_files

    def compute_matrix(self, matrix) -> np.ndarray:
        return np.empty([self.n, self.n])

    def prepare_rows_for_matrix(self, run_type: str, start_index: int, end_index: int):
        pass

    def calculate_row_accuracies(self, run_type: str):
        row_size = len(self.tests[run_type][0])
        self.row_accuracies[run_type] = np.empty(row_size)
        for row in range(row_size):
            accuracy = row_accuracy(self.x, self.y, self.ks, self.mus, self.tests[run_type][row])
            self.row_accuracies[run_type][row] = accuracy
        self.average_row_accuracy[run_type] = self.row_accuracies[run_type].mean()

    def collate_sim_data(self):
        tests = np.ones([self.number_of_tests, 2 * self.n])
        for test in range(self.number_of_tests):
            results_file_name = self.result_file_names[test]
            row = pickle.load(open(results_file_name, "rb"))
            tests[test] = row
        self.tests["sim"] = tests
        self.calculate_row_accuracies("sim")

    def run_simulations(self, start: int, stop: int) -> int:
        actual_stop = min(stop, self.number_of_tests)
        for test in range(start, actual_stop):
            change = test % self.number_of_changes
            # Will change to generating a separate process, eventually
            psc_file_name = self.psc_file_names[change]
            results_file_name = self.result_file_names[test]
            process = Process(target=simulate, args=(psc_file_name, results_file_name, self.n))
            process.start()
        return 0

    def mean_field_solutions(self) -> np.ndarray:
        tests = np.ones([self.number_of_tests, 2 * self.n])

        for test in range(self.number_of_tests):
            change = test % self.number_of_changes
            new_ks, new_gammas = self.change_sets[change]
            if new_ks is None:
                new_ks = self.ks
            if new_gammas is None:
                new_gammas = self.gammas

            tests[test][self.n:] = base_network_ode_solution(new_ks, self.mus, new_gammas)
        return tests

    def generate_overdetermined_test(self, extra_x: int, extra_y: int):
        new_x = self.x + extra_x
        new_y = self.y + extra_y
        return

    def generate_underdetermined_test(self, lost_x: int, lost_y: int):
        new_x = self.x - lost_x
        new_y = self.y - lost_y
        return


class WildTypeTest(RateTest):

    test_type = "W"

    def __init__(self,
                 x: int,
                 y: int,
                 ks: np.ndarray,
                 mus: np.ndarray,
                 gammas: dict,
                 model_type: int):
        RateTest.__init__(self, x, y, ks, mus, gammas, model_type)
        self.change_sets = self.generate_changes()
        self.number_of_tests = 1
        self.number_of_changes = 1

    def generate_changes(self, change_type: int=0):
        return [(None, None)]

    def prepare_rows_for_matrix(self, run_type: str, start_index: int, end_index: int):
        return self.tests[run_type]


class GammaTest(RateTest, Estimator):

    test_type = "G"

    def __init__(self,
                 x: int,
                 y: int,
                 real_vector: np.ndarray,
                 ks: np.ndarray,
                 mus: np.ndarray,
                 gammas: np.ndarray,
                 model_type: int):
        RateTest.__init__(self, x, y, ks, mus, gammas, model_type)
        self.number_of_tests = 2 * self.n
        self.number_of_changes = self.n
        Estimator.__init__(self, real_vector)
        self.change_sets = self.generate_changes(1)

    def generate_changes(self, change_type: int=0):
        change_set = []
        if change_type == 0:
            change_generator = self.generate_changes_mrna_method
        elif change_type == 1:
            change_generator = self.generate_changes_gamma_scaling_method
        elif change_type == 2:
            change_generator = self.generate_changes_gamma_removal_method
        else:
            change_generator = self.generate_changes_gamma_removal_method

        for change in range(self.number_of_changes):
            new_gammas = change_generator(change)
            change_set.append((None, new_gammas))
        return change_set

    def generate_changes_mrna_method(self, test: int):
        mRNA = test % self.x
        gamma_changes = []
        for miRNA in self.gammas[mRNA]:
            multiplier = np.random.uniform(0.5, 2.0)
            new_value = self.gammas[mRNA][miRNA] * multiplier
            gamma_changes.append((mRNA, miRNA, new_value))
        new_gammas = generate_modified_gammas(deepcopy(self.gammas), gamma_changes)
        return new_gammas

    def generate_changes_gamma_scaling_method(self, test: int):
        change_size = 2
        gamma_changes = []
        for change in range(change_size):
            mRNA = random.choice(list(self.gammas.keys()))
            miRNA = random.choice(list(self.gammas[mRNA].keys()))
            multiplier = np.random.uniform(5, 50.0)
            new_value = self.gammas[mRNA][miRNA] * multiplier
            gamma_changes.append((mRNA, miRNA, new_value))

        new_gammas = generate_modified_gammas(deepcopy(self.gammas), gamma_changes)
        return new_gammas

    def generate_changes_gamma_removal_method(self, test: int):
        change_size = 3
        gamma_changes = []
        for change in range(change_size):
            mRNA = random.choice(list(self.gammas.keys()))
            miRNA = random.choice(list(self.gammas[mRNA].keys()))
            gamma_changes.append((mRNA, miRNA, 0))

        new_gammas = generate_modified_gammas(deepcopy(self.gammas), gamma_changes)
        return new_gammas

    def post_processing(self, run_type: str):
        self.matrices[run_type] = self.compute_matrix(self.tests[run_type])
        Estimator.post_processing(self, run_type)

    def compute_matrix(self, tests) -> np.ndarray:
        matrix = np.empty([self.number_of_tests, self.estimate_size])
        for i in range(self.number_of_tests - 1):
            matrix[i] = tests[i + 1][self.n:] - tests[i][self.n:]
        matrix[-1] = tests[-1][self.n:] - tests[0][self.n:]
        return matrix

    # TODO
    def prepare_rows_for_matrix(self, run_type: str, start_index: int, end_index: int):
        tests = self.tests[run_type]


class LambdaTest(RateTest, Estimator):

    test_type = "L"

    def __init__(self,
                 x: int,
                 y: int,
                 real_vector: np.ndarray,
                 ks: np.ndarray,
                 mus: np.ndarray,
                 gammas: np.ndarray,
                 model_type: int,
                 knockout=-1,
                 creation_rate_changes=None):
        if knockout is -1:
            self.knockout = random.randrange(x + y)
        else:
            self.knockout = knockout
        if not creation_rate_changes:
            self.creation_rate_changes = self.select_creation_rate_changes(x+y, self.knockout, 5, 1, x+y-1)
        else:
            self.creation_rate_changes = creation_rate_changes

        RateTest.__init__(self, x, y, ks, mus, gammas)
        self.number_of_tests = 4 * self.n
        self.number_of_changes = self.n
        self.number_of_rows = int(self.number_of_tests / 2)
        Estimator.__init__(self, real_vector)
        self.wild_tests = {}
        self.knockout_tests = {}
        self.change_sets = self.generate_changes()

    # Predetermines all creation rate changes for this network.
    def generate_changes(self, change_type: int=0) -> list:
        change_sets = []
        change_size = 1
        change_iterator = combinations(list(self.creation_rate_changes), change_size)

        for change in range(self.number_of_changes):
            try:
                next_change = next(change_iterator)
            except StopIteration:
                change_size = (change_size + 1) % 5
                change_iterator = combinations(list(self.creation_rate_changes), change_size)
                next_change = next(change_iterator)
            print("Next change: {0}".format(next_change))

            actual_changes = []
            for change_set in next_change:
                for RNA in self.creation_rate_changes[change_set]:
                    multiplier = np.random.uniform(.1, 10)
                    actual_changes.append((RNA, self.ks[RNA] * multiplier))

            new_ks = generate_modified_ks(deepcopy(self.ks), [], actual_changes)
            new_knockout_ks = generate_modified_ks(deepcopy(new_ks), [self.knockout], [])

            change_sets.append((new_ks, None))
            change_sets.append((new_knockout_ks, None))
        return change_sets

    def post_processing(self, run_type: str):
        self.matrices[run_type] = self.compute_matrix(self.tests[run_type])
        Estimator.post_processing(self, run_type)

    def compute_matrix(self, tests: np.ndarray):
        # For lambda tests, two tests are required to produce a single row of the matrix.
        diff_matrix = np.empty([self.number_of_rows, 2 * self.n])
        # Compute the differences between wild type tests and corresponding knockout tests.+
        for row in range(self.number_of_rows):
            # Wild type tests are in even rows, knockout tests are in odd rows.
            tests[row * 2 + 1][self.knockout] = 0
            diff_matrix[row] = tests[row * 2] - tests[row * 2 + 1]

        matrix = np.empty([self.number_of_rows, self.estimate_size])
        for row in range(self.number_of_rows - 1):
            matrix[row] = diff_matrix[row + 1][self.estimate_size:] - diff_matrix[row][self.estimate_size:]
        matrix[-1] = diff_matrix[-1][self.estimate_size:] - diff_matrix[0][self.estimate_size:]
        return matrix

    @staticmethod
    def select_creation_rate_changes(n: int, knockout: int, number_of_changes: int, lower_bound: int,
                                     upper_bound: int) -> dict:
        changes = {}
        possible_changes = list(range(n))
        del possible_changes[knockout]
        for i in range(number_of_changes):
            new_change = random.sample(possible_changes, random.randrange(lower_bound, upper_bound))
            while new_change in changes.values():
                new_change = random.sample(possible_changes, random.randrange(lower_bound, upper_bound))
            changes[i] = new_change
        return changes

    # TODO
    def prepare_rows_for_matrix(self, run_type: str, start_index: int, end_index: int):
        super().prepare_rows_for_matrix(run_type, start_index, end_index)


class KnockoutTest(RateTest):

    test_type = "K"

    def __init__(self, x, y, ks, mus, gammas):
        RateTest.__init__(self, x, y, ks, mus, gammas)
        self.number_of_tests = self.n
        self.number_of_changes = self.n
        self.change_sets = self.generate_changes()

    def generate_changes(self, change_type: int=0) -> list:
        change_set = []

        for change in range(self.number_of_changes):
            new_ks = generate_modified_ks(deepcopy(self.ks), [change], [])
            change_set.append((new_ks, None))
        return change_set

    # TODO
    def prepare_rows_for_matrix(self, run_type: str, start_index: int, end_index: int):
        super().prepare_rows_for_matrix(run_type, start_index, end_index)


def row_accuracy(x: int, y: int, ks: np.ndarray, mus: np.ndarray, row: np.ndarray):
    n = x + y
    val = 0
    for i in range(x):
        val += ks[i] * row[i] - row[i + n] * mus[i]
    for i in range(x, n):
        val -= ks[i] * row[i] - row[i + n] * mus[i]
    return val


def generate_modified_ks(ks: np.ndarray, knockouts: list, creation_rate_changes: list) -> np.ndarray:
    for knockout in knockouts:
        ks[knockout] = 0
    for creation_rate_change in creation_rate_changes:
        ks[creation_rate_change[0]] = creation_rate_change[1]
    return ks


def generate_modified_gammas(gammas: dict, gamma_changes: list):
    for gamma_change in gamma_changes:
        gammas[gamma_change[0]][gamma_change[1]] = gamma_change[2]
        gammas[gamma_change[1]][gamma_change[0]] = gamma_change[2]
    return gammas
