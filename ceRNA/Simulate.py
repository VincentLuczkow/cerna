import pickle
import numpy as np
import scipy.integrate
import stochpy
from typing import Callable


def base_network_ode_solution(ks, mus, gammas):
    # Returns a function which will calculate the derivatives
    n = len(ks)
    new_gammas = np.zeros([n, n])
    for RNA in gammas:
        for g in gammas[RNA]:
            new_gammas[RNA][g] = gammas[RNA][g]
            new_gammas[g][RNA] = gammas[RNA][g]

    def calculate_derivatives(species_means, sim_time):
        derivatives = np.zeros(n)
        np.add(derivatives, ks, derivatives)
        np.subtract(derivatives, np.multiply(mus, species_means), derivatives)
        np.subtract(derivatives, np.multiply(np.dot(new_gammas, species_means), species_means), derivatives)
        return derivatives

    species = 10 * np.random.rand(n)
    stop_time = 100
    numpoints = 1000
    t = [stop_time * float(i) / (numpoints - 1) for i in range(numpoints)]
    full_solution = scipy.integrate.odeint(calculate_derivatives, species, t, atol=1.0e-8, rtol=1.0e-8)
    solution = full_solution[-1]

    return solution


def get_burst_network_solver(p_on: np.ndarray, alphas: np.ndarray, betas: np.ndarray) -> Callable:
    n = len(alphas)

    def numerical_approximator(ks, mus, gammas):
        new_gammas = np.zeros([n, n])
        for RNA in gammas:
            for g in gammas[RNA]:
                new_gammas[RNA][g] = gammas[RNA][g]
                new_gammas[g][RNA] = gammas[RNA][g]

        # The actual import function.
        def calculate_derivatives(species_means, sim_time):
            derivatives = np.zeros(n)
            # Component from birth processes
            np.add(derivatives, np.multiply(ks, p_on), derivatives)
            # Component from death processes.
            np.subtract(derivatives, np.multiply(mus, species_means), derivatives)
            # Component from mRNA - miRNA interactions.
            np.subtract(derivatives, np.multiply(np.dot(new_gammas, species_means), species_means), derivatives)
            return derivatives

        species = 10 * np.random.rand(n)
        stop_time = 100
        numpoints = 1000
        t = [stop_time * float(i) / (numpoints - 1) for i in range(numpoints)]
        full_solution = scipy.integrate.odeint(calculate_derivatives, species, t, atol=1.0e-8, rtol=1.0e-8)
        solution = full_solution[-1]

        return solution

    return numerical_approximator


def simulate(input_file: str, output_file: str, n: int):
    simulation = stochpy.SSA(File=input_file)
    simulation.DoCompleteStochSim()
    means = simulation.data_stochsim.species_means
    array = np.ones(2 * n)
    array[n:] = convert_stochpy_means_to_array(means)
    # Store modified data
    pickle.dump(array, open(output_file, "wb"))


# StochPy stores mean values in a dictionary.
# This function converts the dictionary to an array.
def convert_stochpy_means_to_array(means: dict) -> np.ndarray:
    n = len(means)
    array = np.zeros(n)
    for RNA in range(n):
        array[RNA] = means["r{0}".format(RNA)]

    return array