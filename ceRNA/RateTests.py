from copy import deepcopy, copy
from getpass import getuser
from itertools import combinations
from multiprocessing import Process
import pickle
import random
from pdb import set_trace
from typing import List, Tuple

import numpy as np

from .Simulate import base_network_ode_solution, simulate
from .NetworkFiles import create_base_network_file
from .Estimators import Esti
from .Definitions import *


class RateTest:
    test_type = ""
    path_to_stochpy_folder = "/home/" + getuser() + "/Stochpy/"

    def __init__(self, x: int, y: int,
                 ks: np.ndarray, mus: np.ndarray, gammas: dict,
                 deterministic_solver=None, file_writer=None):
        # Number of mRNAs
        self.x = x
        # Number of miRNAs
        self.y = y
        # Total number of RNAs in the network
        self.n = x + y
        # Birth rates
        self.ks = ks
        # Death rates
        self.mus = mus
        # Interaction rates
        self.gammas = gammas
        # Function for running deterministic tests.
        self.deterministic_solver = deterministic_solver
        # Function for creating simulation files.
        self.file_writer = file_writer
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

    def mean_field_approximation(self):
        pass

    def generate_changes(self, change_type: int=0):
        return None, None

    def create_psc_files(self) -> tuple:
        psc_file_names = []  # type: List[str]
        results_file_names = []  # type: List[str]
        base_file_name = "BaseNet{0},{1}{2}Test".format(self.x, self.y, self.test_type)

        for change in range(self.number_of_changes):
            change_file_name = base_file_name + str(change)
            changed_ks, changed_gammas = self.change_sets[change]
            if changed_ks is None:
                changed_ks = self.ks
            if changed_gammas is None:
                changed_gammas = self.gammas

            create_base_network_file(self.x, self.y, changed_ks, self.mus, changed_gammas, change_file_name)
            psc_file_names.append(change_file_name)

        for test in range(self.number_of_tests):
            results_file_name = RateTest.path_to_stochpy_folder + base_file_name + str(test) + "Results"
            results_file_names.append(results_file_name)

        return psc_file_names, results_file_names

    # Returns a submatrix suitable for use by an Estimator.
    # The submatrix is row start_index to row end_index-1
    def prepare_rows_for_matrix(self, run_type: str, start_index: int, end_index: int) -> np.ndarray:
        return self.tests[run_type][start_index: end_index]

    # Calculates how closely this RateTest matches the prediction from the mean field approximation.
    def calculate_row_accuracies(self, run_type: str):
        row_size = len(self.tests[run_type][0])
        self.row_accuracies[run_type] = np.empty(row_size)
        for row in range(row_size):
            accuracy = row_accuracy(self.x, self.y, self.ks, self.mus, self.tests[run_type][row])
            self.row_accuracies[run_type][row] = accuracy
        self.average_row_accuracy[run_type] = self.row_accuracies[run_type].mean()

    # Retrieves
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

    test_type = "wild"

    def __init__(self, x: int, y: int,
                 ks: np.ndarray, mus: np.ndarray, gammas: dict,
                 deterministic_solver=None, file_writer=None):
        RateTest.__init__(self, x, y, ks, mus, gammas, deterministic_solver, file_writer)
        self.change_sets = self.generate_changes()
        self.number_of_tests = 1
        self.number_of_changes = 1

    def generate_changes(self, change_type: int=0) -> List[Tuple]:
        return [(None, None)]

    def prepare_rows_for_matrix(self, run_type: str, start_index: int, end_index: int):
        return self.tests[run_type]


class GammaTest(RateTest, Esti):

    test_type = "gamma"

    def __init__(self, x: int, y: int,
                 ks: np.ndarray, mus: np.ndarray, gammas: dict, real_vector,
                 deterministic_solver=None, file_writer=None):

        RateTest.__init__(self, x, y, ks, mus, gammas, deterministic_solver, file_writer)
        self.number_of_tests = 2 * self.n
        self.number_of_changes = self.n
        Esti.__init__(self, real_vector)
        self.change_sets = self.generate_changes(1)

    def generate_changes(self, change_type: int=0) -> List[Tuple[np.ndarray, np.ndarray]]:
        change_set = []
        if change_type == GammaChange.MRNA_METHOD:
            change_generator = self.generate_changes_mrna_method
        elif change_type == GammaChange.GAMMA_SCALING_METHOD:
            change_generator = self.generate_changes_gamma_scaling_method
        elif change_type == GammaChange.GAMMA_REMOVAL_METHOD:
            change_generator = self.generate_changes_gamma_removal_method
        else:
            change_generator = self.generate_changes_gamma_removal_method

        for change in range(self.number_of_changes):
            new_gammas = change_generator(change)
            change_set.append((None, new_gammas))
        return change_set

    def generate_changes_mrna_method(self, test: int) -> np.ndarray:
        mRNA = test % self.x
        gamma_changes = []
        for miRNA in self.gammas[mRNA]:
            multiplier = np.random.uniform(0.5, 2.0)
            new_value = self.gammas[mRNA][miRNA] * multiplier
            gamma_changes.append((mRNA, miRNA, new_value))
        new_gammas = generate_modified_gammas(deepcopy(self.gammas), gamma_changes)
        return new_gammas

    def generate_changes_gamma_scaling_method(self, test: int) -> np.ndarray:
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

    def generate_changes_gamma_removal_method(self, test: int) -> np.ndarray:
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
        Esti.post_processing(self, run_type)

    def compute_matrix(self, tests) -> np.ndarray:
        matrix = np.empty([self.number_of_tests, self.estimate_size])
        for i in range(self.number_of_tests - 1):
            matrix[i] = tests[i + 1][self.n:] - tests[i][self.n:]
        matrix[-1] = tests[-1][self.n:] - tests[0][self.n:]
        return matrix

    def prepare_rows_for_matrix(self, run_type: str, start_index: int, end_index: int):
        tests = self.tests[run_type]
        number_of_rows = end_index - start_index
        matrix = np.empty([number_of_rows, 2 * self.n])
        for row in range(start_index, end_index):
            matrix[row - start_index] = tests[(row + 1) % number_of_rows] - tests[row]
        return matrix


class NewGammaTest(RateTest):
    test_type = "gamma"

    def __init__(self, x: int, y: int,
                 ks: np.ndarray, mus: np.ndarray, gammas: dict, real_vector,
                 deterministic_solver=None, file_writer=None):

        RateTest.__init__(self, x, y, ks, mus, gammas, deterministic_solver, file_writer)
        self.number_of_tests = 2 * self.n
        self.number_of_changes = self.n

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

    # Generates changes by choosing a particular mRNA, and modifying all of its interaction rates
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

    def prepare_rows_for_matrix(self, run_type: str, start_index: int, end_index: int) -> np.ndarray:
        tests = self.tests[run_type]
        number_of_rows = end_index - start_index
        matrix = np.empty([number_of_rows, 2 * self.n])
        for row in range(start_index, end_index):
            matrix[row - start_index] = tests[(row + 1) % number_of_rows] - tests[row]
        return matrix


class LambdaTest(RateTest):

    test_type = "lambda"

    def __init__(self, x: int, y: int, ks: np.ndarray, mus: np.ndarray, gammas: np.ndarray,
                 deterministic_solver=None, file_writer=None,
                 knockout=-1, creation_rate_changes=None):
        if knockout is -1:
            self.knockout = random.randrange(x + y)
        else:
            self.knockout = knockout
        if not creation_rate_changes:
            self.creation_rate_changes = self.select_creation_rate_changes(x+y, self.knockout, 15, 1, x+y-1)
        else:
            self.creation_rate_changes = creation_rate_changes

        RateTest.__init__(self, x, y, ks, mus, gammas, deterministic_solver, file_writer)
        self.number_of_tests = 4 * self.n
        self.number_of_changes = self.n
        self.number_of_rows = int(self.number_of_tests / 2)
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
        pass

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

    # TODO: Testing
    def prepare_rows_for_matrix(self, run_type: str, start_index: int, end_index: int):
        tests = self.tests[run_type]
        number_of_rows = end_index - start_index

        difference_matrix = np.empty([self.number_of_rows, 2 * self.n])
        # Compute the differences between wild type tests and corresponding knockout tests.+
        for row in range(self.number_of_rows):
            # Wild type tests are in even rows, knockout tests are in odd rows.
            tests[row * 2 + 1][self.knockout] = 0
            difference_matrix[row] = tests[row * 2] - tests[row * 2 + 1]

        matrix = np.empty([number_of_rows, 2 * self.n])
        for row in range(start_index, end_index):
            matrix[row - start_index] = difference_matrix[(row + 1) % number_of_rows] - difference_matrix[row]
        set_trace()
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




class KnockoutTest(RateTest):

    test_type = "knockout"

    def __init__(self, x: int, y: int,
                 ks: np.ndarray, mus: np.ndarray, gammas: np.ndarray,
                 deterministic_solver=None, file_writer=None):
        RateTest.__init__(self, x, y, ks, mus, gammas, deterministic_solver, file_writer)
        self.number_of_tests = self.n
        self.number_of_changes = self.n
        self.change_sets = self.generate_changes()

    def generate_changes(self, change_type: int=0) -> list:
        change_set = []

        for change in range(self.number_of_changes):
            new_ks = generate_modified_ks(deepcopy(self.ks), [change], [])
            change_set.append((new_ks, None))
        return change_set

    # TODO: Implement
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
