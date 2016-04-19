import numpy as np
from ceRNA.Calculations import estimate_parameters, rmse_vector, percent_error_vector, set_sub_matrix


class Estimator:
    def __init__(self, real_vector: np.ndarray, tests: np.ndarray):
        self.real_vector = real_vector
        self.estimate_size = len(real_vector)
        self.tests = tests
        self.number_of_tests = len(tests)
        self.matrix = self.create_matrix()
        self.estimate = None

        self.rmses = None
        self.average_rmse = 0

        self.relative_errors = None
        self.average_relative_error = 0

    def create_matrix(self) -> np.ndarray:
        return self.tests

    def calculate_accuracy(self):
        self.estimate = estimate_parameters(self.matrix)[-len(self.real_vector):]

        # Accuracy calculations
        self.rmses = rmse_vector(self.real_vector, self.estimate)
        self.average_rmse = self.rmses.mean()
        self.relative_errors = percent_error_vector(self.real_vector, self.estimate)
        self.average_relative_error = self.relative_errors.mean()
        self.print_results()

    def print_results(self):
        print("Results:")
        print("    RMSE Vector: {0}".format(self.rmses))
        print("    Average RMSE: {0}".format(self.average_rmse))
        print("    PE Vector: {0}".format(self.relative_errors))
        print("    Average PE: {0}".format(self.average_relative_error))


class DecayEstimator(Estimator):
    def __init__(self, real_vector: np.ndarray, tests: np.ndarray):
        Estimator.__init__(self, real_vector, tests)

    # Returns the right half of tests.
    def create_matrix(self) -> np.ndarray:
        matrix = self.tests[:, -self.estimate_size:]
        return matrix


class Esti:
    def __init__(self, real_vector: np.ndarray):
        self.real_vector = real_vector
        self.estimate_size = len(real_vector)
        self.matrices = {}
        self.estimates = {}

        self.rmses = {}
        self.average_rmse = {}

        self.percent_errors = {}
        self.average_relative_error = {}

    def print_results(self, run_type: str):
        print("{0} Results:".format(run_type))
        print("    RMSE Vector: {0}".format(self.rmses[run_type]))
        print("    Average RMSE: {0}".format(self.average_rmse[run_type]))
        print("    PE Vector: {0}".format(self.percent_errors[run_type]))
        print("    Average PE: {0}".format(self.average_relative_error[run_type]))

    def post_processing(self, run_type: str):
        self.estimates[run_type] = estimate_parameters(self.matrices[run_type])

        # Accuracy calculations
        self.rmses[run_type] = rmse_vector(self.real_vector, self.estimates[run_type])
        self.average_rmse[run_type] = self.rmses[run_type].mean()
        self.percent_errors[run_type] = percent_error_vector(self.real_vector, self.estimates[run_type])
        self.average_relative_error[run_type] = self.percent_errors[run_type].mean()

    @staticmethod
    def prepare_gamma_tests(tests: np.ndarray):
        n = int(len(tests[0]) / 2)



class CombinedTest(Esti):
    def __init__(self, real_vector):
        Esti.__init__(self, real_vector)

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