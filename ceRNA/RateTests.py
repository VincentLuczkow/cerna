from copy import deepcopy
from getpass import getuser
from itertools import combinations
from multiprocessing import Process
import pickle
import random
import pdb
import numpy as np

from .Calculations import estimate_parameters, rmse_vector, percent_error_vector, set_sub_matrix
from .Simulate import base_network_ode_solution, simulate
from .NetworkFiles import create_base_network_file


class Estimator:
    def __init__(self, real_vector):
        self.real_vector = real_vector
        self.estimate_size = len(real_vector)
        self.matrices = {}
        self.estimates = {}

        self.rmses = {}
        self.average_rmse = {}

        self.percent_errors = {}
        self.average_percent_error = {}

    def print_results(self, run_type: str):
        print("{0} Results:".format(run_type))
        print("    RMSE Vector: {0}".format(self.rmses[run_type]))
        print("    Average RMSE: {0}".format(self.average_rmse[run_type]))
        print("    PE Vector: {0}".format(self.percent_errors[run_type]))
        print("    Average PE: {0}".format(self.average_percent_error[run_type]))

    def post_processing(self, run_type: str):
        self.estimates[run_type] = estimate_parameters(self.matrices[run_type])

        # Accuracy calculations
        self.rmses[run_type] = rmse_vector(self.real_vector, self.estimates[run_type])
        self.average_rmse[run_type] = self.rmses[run_type].mean()
        self.percent_errors[run_type] = percent_error_vector(self.real_vector, self.estimates[run_type])
        self.average_percent_error[run_type] = self.percent_errors[run_type].mean()


class RateTest:

    path_to_stochpy_folder = "/home/" + getuser() + "/Stochpy/"

    def __init__(self, x, y, ks, mus, gammas):
        self.x = x
        self.y = y
        self.n = x + y
        self.ks = ks
        self.mus = mus
        self.gammas = gammas
        self.number_of_tests = 0
        self.tests = {}
        self.row_accuracies = {}
        self.average_row_accuracy = {}
        self.psc_file_names = []
        self.result_file_names = []

    def setup(self):
        self.tests["ode"] = self.mean_field_solutions()
        self.psc_file_names, self.result_file_names = self.create_psc_files()

    def create_psc_files(self) -> tuple:
        return "", ""

    def compute_matrix(self, matrix) -> np.ndarray:
        return np.empty([self.n, self.n])

    def calculate_row_accuracies(self, run_type: str):
        row_size = len(self.tests[run_type][0])
        self.row_accuracies[run_type] = np.empty(row_size)
        for row in range(row_size):
            accuracy = row_accuracy(self.x, self.y, self.ks, self.mus, self.tests[run_type][row])
            self.row_accuracies[run_type][row] = accuracy
        self.average_row_accuracy[run_type] = self.row_accuracies[run_type].mean()

    def collate_sim_data(self):
        self.calculate_row_accuracies("sim")

    def run_simulations(self, start: int, stop: int) -> int:
        for test in range(start, stop+1):
            # Will change to generating a separate process, eventually
            psc_file_name = self.psc_file_names[test]
            results_file_name = self.result_file_names[test]
            process = Process(target=simulate, args=(psc_file_name, results_file_name, self.n))
            process.start()
        return 0

    def mean_field_solutions(self) -> np.ndarray:
        return np.empty([self.n, self.n])


class WildTypeTest(RateTest):
    def __init__(self, x, y, ks, mus, gammas):
        super().__init__(x, y, ks, mus, gammas)

    def create_psc_files(self) -> tuple:
        test_file_name = "BaseNet{0},{1}WTest".format(self.x, self.y)
        results_file_name = "BaseNet{0},{1}WTestResults".format(self.x, self.y)
        create_base_network_file(self.x, self.y, self.ks, self.mus, self.gammas, test_file_name)
        return test_file_name, results_file_name

    def mean_field_solutions(self) -> np.ndarray:
        tests = np.ones(2 * self.n)
        tests[self.n:] = base_network_ode_solution(self.ks, self.mus, self.gammas)
        return tests


class GammaTest(RateTest, Estimator):
    def __init__(self, x, y, real_vector, ks, mus, gammas):
        RateTest.__init__(self, x, y, ks, mus, gammas)
        self.number_of_tests = 3 * self.n
        Estimator.__init__(self, real_vector)
        self.change_set = self.generate_changes(1)

    def generate_changes(self, change_type: int):
        change_set = []
        if change_type == 0:
            change_generator = self.generate_changes_mRNA_method
        elif change_type == 1:
            change_generator = self.generate_changes_gamma_scaling_method
        else:
            change_generator = self.generate_changes_gamma_removal_method

        for test in range(self.number_of_tests):
            new_gammas = change_generator(test)
            change_set.append(new_gammas)
        return change_set

    def generate_changes_mRNA_method(self, test: int):
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

    def create_psc_files(self) -> int:
        psc_files = []
        results_files = []
        base_test_name = "BaseNet{0},{1}GTest".format(self.x, self.y)

        for test in range(self.n):
            test_file_name = base_test_name + str(test)
            results_file_name = RateTest.path_to_stochpy_folder + test_file_name + "Results"
            new_gammas = self.change_set[test]

            create_base_network_file(self.x, self.y, self.ks, self.mus, new_gammas, test_file_name)
            psc_files.append(test_file_name)
            results_files.append(results_file_name)

        return psc_files, results_files

    def post_processing(self, run_type: str):
        self.matrices[run_type] = self.compute_matrix(self.tests[run_type])
        Estimator.post_processing(self, run_type)

    def mean_field_solutions(self) -> np.ndarray:
        tests = np.ones([self.number_of_tests, 2 * self.n])

        for test in range(self.number_of_tests):
            new_gammas = self.change_set[test]
            tests[test][self.n:] = base_network_ode_solution(self.ks, self.mus, new_gammas)
            accuracy = row_accuracy(self.x, self.y, self.ks, self.mus, tests[test])
            if accuracy > 1e-5:
                print("Bad")
                pdb.set_trace()
        return tests

    def collate_sim_data(self):
        tests = np.ones([self.n, 2 * self.n])
        for test in range(self.number_of_tests):
            results_file_name = self.result_file_names[test]
            row = pickle.load(open(results_file_name, "rb"))
            tests[test] = row
        self.tests["sim"] = tests
        RateTest.collate_sim_data(self)

    def compute_matrix(self, tests) -> np.ndarray:
        matrix = np.empty([self.number_of_tests, self.estimate_size])
        for i in range(self.number_of_tests - 1):
            matrix[i] = tests[i + 1][self.n:] - tests[i][self.n:]
        matrix[self.number_of_tests - 1] = np.zeros(self.n)
        return matrix


class LambdaTest(RateTest, Estimator):
    def __init__(self, x, y, real_vector, ks, mus, gammas, knockout=-1, creation_rate_changes=None):
        RateTest.__init__(self, x, y, ks, mus, gammas)
        self.number_of_tests = 6 * self.n
        self.number_of_rows = int(self.number_of_tests / 2)
        Estimator.__init__(self, real_vector)
        self.wild_tests = {}
        self.knockout_tests = {}
        if knockout is -1:
            self.knockout = random.randrange(self.n)
        else:
            self.knockout = knockout
        if not creation_rate_changes:
            self.creation_rate_changes = self.select_creation_rate_changes(self.n, self.knockout, 5, 1, self.n - 1)
        else:
            self.creation_rate_changes = creation_rate_changes

        self.change_sets = self.generate_change_sets()

    # Predetermines all creation rate changes for this network.
    def generate_change_sets(self) -> list:
        change_sets = []
        change_size = 1
        change_iterator = combinations(list(self.creation_rate_changes), change_size)

        for test in range(self.number_of_tests):
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

            change_sets.append(new_ks)
            change_sets.append(new_knockout_ks)

        return change_sets

    def create_psc_files(self) -> tuple:

        test_files = []
        result_files = []

        wild_type_base_name = "BaseNet{0},{1}LTest".format(self.x, self.y)
        knockout_base_name = "BaseNet{0},{1}LKTest".format(self.x, self.y)

        for test in range(self.number_of_rows):
            wild_type_test_name = wild_type_base_name + str(test)
            wild_type_results_name = RateTest.path_to_stochpy_folder +  wild_type_test_name + "Results"
            knockout_test_name = knockout_base_name + "K" + str(test)
            knockout_results_name = RateTest.path_to_stochpy_folder + knockout_test_name + "Results"

            # Create a new set of creation rates, with the changes implemented.
            new_ks = self.change_sets[2 * test]
            new_knockout_ks = self.change_sets[2 * test + 1]

            create_base_network_file(self.x, self.y, new_ks, self.mus, self.gammas, wild_type_test_name)
            create_base_network_file(self.x, self.y, new_knockout_ks, self.mus, self.gammas, knockout_test_name)

            test_files.append(wild_type_test_name)
            test_files.append(knockout_test_name)
            result_files.append(wild_type_results_name)
            result_files.append(knockout_results_name)

        return test_files, result_files

    def post_processing(self, run_type: str):
        self.matrices[run_type] = self.new_compute_matrix(self.tests[run_type])
        Estimator.post_processing(self, run_type)

    def mean_field_solutions(self) -> np.ndarray:
        wild_tests = np.ones([self.n, 2 * self.n])
        knockout_tests = np.ones([self.n, 2 * self.n])
        tests = np.ones([self.number_of_tests, 2 * self.n])
        for test in range(self.number_of_tests):
            #new_ks, new_knockout_ks = self.change_sets[test]
            new_ks = self.change_sets[test]
            tests[test][self.n:] = base_network_ode_solution(new_ks, self.mus, self.gammas)

            # Data for the wildtype with modified creation rates.
            #wild_tests[test][self.n:] = base_network_ode_solution(new_ks, self.mus, self.gammas)

            # Get data for the knockout with the same modified creation rates.
            #knockout_tests[test][self.n:] = base_network_ode_solution(new_knockout_ks, self.mus, self.gammas)
            # Set the knockout RNA to zero.
            #knockout_tests[test][self.knockout] = 0
        #self.wild_tests["ode"] = wild_tests
        #self.knockout_tests["ode"] = knockout_tests

        return tests

    def compute_matrix(self, tests):
        matrix = np.zeros([self.number_of_tests, self.n])
        # Set up matrix
        for i in range(self.n - 1):
            matrix[i] = tests[i + 1][self.n:] - tests[i][self.n:]
        return matrix

    def new_compute_matrix(self, tests):
        # For lambda tests, two tests are required to produce a single row of the matrix.
        difference_matrix = np.empty([self.number_of_rows, 2 * self.n])
        # Compute the differences between wild type tests and corresponding knockout tests.+
        for row in range(self.number_of_rows):
            # Wild type tests are in even rows, knockout tests are in odd rows.
            tests[row * 2 + 1][self.knockout] = 0
            difference_matrix[row] = tests[row * 2] - tests[row * 2 + 1]
            if row_accuracy(self.x, self.y, self.ks, self.mus, difference_matrix[row]) > 1e-5:
                pdb.set_trace()
        matrix = np.empty([self.number_of_rows, self.estimate_size])
        for row in range(self.number_of_rows - 1):
            matrix[row] = difference_matrix[row + 1][self.estimate_size:] - difference_matrix[row][self.estimate_size:]
        matrix[self.number_of_rows-1] = \
            difference_matrix[row][self.estimate_size:] - difference_matrix[0][self.estimate_size:]
        return matrix

    def collate_sim_data(self):
        wild_tests = np.ones([self.n, 2 * self.n])
        knockout_tests = np.ones([self.n, 2 * self.n])
        for test in range(self.n):
            wild_results_file_name = self.result_file_names[2 * test]
            knockout_results_file_name = self.result_file_names[2 * test + 1]
            wild_row = pickle.load(open(wild_results_file_name, "rb"))
            knockout_row = pickle.load(open(knockout_results_file_name, "rb"))
            wild_tests[test] = wild_row
            knockout_tests[test] = knockout_row
        self.wild_tests["sim"] = wild_tests
        self.knockout_tests["sim"] = knockout_tests
        self.tests["sim"] = wild_tests - knockout_tests
        RateTest.collate_sim_data(self)

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
    def __init__(self, x, y, ks, mus, gammas):
        super().__init__(x, y, ks, mus, gammas)
        self.change_sets = self.generate_changes()

    def generate_changes(self) -> list:
        change_set = []

        for test in range(self.n):
            new_ks = generate_modified_ks(deepcopy(self.ks), [test], [])
            change_set.append(new_ks)
        return change_set

    def create_psc_files(self) -> tuple:
        test_files = []
        result_files = []
        base_test_name = "BaseNet{0},{1}KTest".format(self.x, self.y)

        for test in range(self.n):
            test_file_name = base_test_name + str(test)
            results_file_name = RateTest.path_to_stochpy_folder + test_file_name + "Results"
            new_ks = self.change_sets[test]
            create_base_network_file(self.x, self.y, new_ks, self.mus, self.gammas, test_file_name)

            test_files.append(test_file_name)
            result_files.append(results_file_name)

        return test_files, result_files

    def mean_field_solutions(self) -> np.ndarray:
        tests = np.ones([self.n, 2 * self.n])

        for test in range(self.n):
            new_ks = self.change_sets[test]
            tests[test][self.n:] = base_network_ode_solution(new_ks, self.mus, self.gammas)
            tests[test][test] = 0
        return tests

    def collate_sim_data(self):
        tests = np.ones([self.n, 2 * self.n])
        for test in range(self.n):
            results_file_name = self.result_file_names[2 * test]
            row = pickle.load(open(results_file_name, "rb"))
            tests[test] = row
        self.tests["sim"] = tests
        RateTest.collate_sim_data(self)


class CombinedTest(Estimator):
    def __init__(self, real_vector):
        Estimator.__init__(self, real_vector)

    def set_matrix(self, wild_type: np.ndarray, knockouts: np.ndarray, decay: np.ndarray, run_type: str) -> int:
        new_matrix = np.empty([self.estimate_size, self.estimate_size])
        new_matrix[0] = wild_type

        knockout_row_map = range(1, len(knockouts))
        knockouts_column_map = range(self.estimate_size)
        set_sub_matrix(knockout_row_map, knockouts_column_map, new_matrix, knockouts)

        decay_row_map = range(len(knockouts), self.estimate_size - 1)
        decay_column_map = range(self.estimate_size)
        set_sub_matrix(decay_row_map, decay_column_map, new_matrix, decay)
        new_matrix[self.estimate_size-1] = np.zeros(self.estimate_size)
        self.matrices[run_type] = new_matrix


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
