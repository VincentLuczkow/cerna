import numpy as np
from ceRNA.Calculations import estimate_parameters, rmse_vector, percent_error_vector


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

    def print_results(self):
        print("Results:")
        print("    Average RMSE: {0}".format(self.average_rmse))
        print("    Average PE:   {0}".format(self.average_relative_error))


class DecayEstimator(Estimator):
    def __init__(self, real_vector: np.ndarray, tests: np.ndarray):
        Estimator.__init__(self, real_vector, tests)

    # Returns the right half of tests.
    def create_matrix(self) -> np.ndarray:
        matrix = self.tests[:, -self.estimate_size:]
        return matrix