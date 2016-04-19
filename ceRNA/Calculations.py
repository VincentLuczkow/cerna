import numpy as np
from pdb import set_trace


def get_sub_matrix(rows, columns, matrix):
    x = len(columns)
    y = len(rows)
    sub_matrix = np.zeros([y, x])
    for i in rows:
        for j in columns:
            sub_matrix[rows[i]][columns[j]] = matrix[i][j]
    return sub_matrix


def set_sub_matrix(rows, columns, matrix, sub_matrix):
    x = len(columns)
    y = len(rows)
    for i in range(y):
        for j in range(x):
            matrix[rows[i]][columns[j]] = sub_matrix[i][j]
    return matrix


def estimate_parameters(matrix) -> tuple:
    set_trace()
    u, singular_values, v = np.linalg.svd(matrix, full_matrices=True)
    # We are dealing with overdetermined homogeneous systems. The best solution is the column
    # of v that corresponds to the lowest singular value. Because singular values are
    # in descending order, this is always the last column.
    nullspace = v[-1]
    # It doesn't really matter which direction the vector is pointing, but we do want it consistent.
    # So we scale it so that the first value is positive.
    if nullspace[0] < 0:
        nullspace *= -1
    estimate = abs(nullspace)
    return estimate


def rmse_vector(true: np.ndarray, estimate: np.ndarray):
    vector = np.zeros(len(estimate))
    for value in range(len(estimate)):
        vector[value] = root_mean_squared_error(true, estimate, value)
    return vector


def root_mean_squared_error(true: np.ndarray, estimate: np.ndarray, value_to_scale_by: int):
    scaling_factor = true[value_to_scale_by] / estimate[value_to_scale_by]
    scaled_estimate = scaling_factor * estimate
    return np.sqrt((np.square(scaled_estimate - true)).mean())


def percent_error_vector(true: np.ndarray, estimate: np.ndarray):
    vector = np.zeros(len(estimate))
    for value in range(len(estimate)):
        vector[value] = percentage_error(true, estimate, value)
    return vector


def percentage_error(true: np.ndarray, estimate: np.ndarray, value_to_scale_by: int):
    scaling_factor = true[value_to_scale_by] / estimate[value_to_scale_by]
    scaled_estimate = scaling_factor * estimate
    distance_vector = np.sqrt(np.square(scaled_estimate-true))
    percent_error_vector = distance_vector / true
    percent_error = percent_error_vector.mean()
    return percent_error