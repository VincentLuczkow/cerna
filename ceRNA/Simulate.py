import pickle
import numpy as np
import scipy.integrate
import stochpy
from pdb import set_trace
from typing import Callable


def base_network_ode_solution(ks, mus, gammas):
    # Returns a function which will calculate the derivatives
    n = len(ks)

    def calculate_derivatives(species_means, sim_time):
        derivatives = np.zeros(n)
        np.add(derivatives, ks, derivatives)
        np.subtract(derivatives, np.multiply(mus, species_means), derivatives)
        #set_trace()
        np.subtract(derivatives, np.multiply(np.dot(gammas, species_means), species_means), derivatives)
        return derivatives

    species = 10 * np.random.rand(n)
    stop_time = 100
    numpoints = 1000
    t = stop_time / (numpoints-1) * np.arange(numpoints)
    full_solution = scipy.integrate.odeint(calculate_derivatives, species, t, atol=1.0e-8, rtol=1.0e-8)
    solution = full_solution[-1]

    return solution


def get_burst_network_solver(p_on: np.ndarray, alphas: np.ndarray, betas: np.ndarray) -> Callable:
    n = len(p_on)

    def numerical_approximator(ks, mus, gammas):
        arrival_rates = np.multiply(ks, p_on)

        # Given the current species means, calculates derivatives.
        def calculate_derivatives(species_means, sim_time):
            derivatives = np.zeros(n)
            # Component from birth processes
            np.add(derivatives, arrival_rates, derivatives)
            # Component from death processes.
            np.subtract(derivatives, np.multiply(mus, species_means), derivatives)
            # Component from mRNA - miRNA interactions.
            np.subtract(derivatives, np.multiply(np.dot(gammas, species_means), species_means), derivatives)
            return derivatives

        species = 10 * np.random.rand(n)
        stop_time = 100
        numpoints = 1000
        t = stop_time / (numpoints-1) * np.arange(numpoints)
        full_solution = scipy.integrate.odeint(calculate_derivatives, species, t, atol=1.0e-8, rtol=1.0e-8)
        solution = full_solution[-1]

        return solution

    return numerical_approximator


# Returns a function that solves the symmetric complex network
def get_symmetric_complex_network_solver(x: int, y: int, p_on: np.ndarray, alphas: np.ndarray, betas: np.ndarray,
                                         deltas: np.ndarray, zetas: np.ndarray) -> Callable:
    """Returns a function which, when passed values for arrival, decay, and interaction rates,
    simulates the corresponding system."""
    n = len(p_on)

    def numerical_solution(ks, mus, gammas):
        arrival_rates = np.multiply(ks, p_on)

        # The actual import function.
        def calculate_derivatives(species_means, sim_time):
            derivatives = np.zeros(n + x*y)

            # Useful intermediate values
            # m_(i,j) = gamma_(i,j) * <i> * <j>
            interaction_rates_matrix = gammas * np.outer(species_means, species_means)
            number_of_complexes = np.size(interaction_rates_matrix)

            # Derivatives for RNAs
            # Component from birth processes for RNAs
            np.add(derivatives[:n], arrival_rates, derivatives[:n])
            # Component from death processes for RNAs
            np.subtract(derivatives[:n], np.multiply(mus, species_means[:n]), derivatives[:n])
            # Subtract the mRNA-miRNA component from RNA derivatives
            gamma_contributions_to_RNAs = np.sum(interaction_rates_matrix, 0)
            np.subtract(derivatives[:n], gamma_contributions_to_RNAs, derivatives[:n])
            # Add the complex-RNA component to RNA derivatives
            delta_values = deltas * np.reshape(derivatives[n:], [n, n])
            np.add(derivatives[:n], np.sum(delta_values, 0)), derivatives[:n]

            # Derivatives for complexes
            # Add the gamma component:
            # d/dt <c_(i,j)> += gamma_(i,j) * <i> * <j>
            np.add(derivatives[n:], np.reshape(interaction_rates_matrix, number_of_complexes, derivatives[n:]))
            # Subtract the zeta component:
            # d/dt <c_(i,j)> -= zeta_(i,j) * <c_(i,j)>
            np.subtract(derivatives[n:], zetas * species_means[n:], derivatives[n:])
            # Subtract the delta component:
            # d/dt <c_(i,j)> -= delta_(i,j) * <c_(i,j)>
            np.subtract(derivatives[n:], deltas * species_means[n:], derivatives[n:])

            return derivatives

        species = 10 * np.random.rand(n + x*y)
        stop_time = 100
        numpoints = 1000
        t = stop_time / (numpoints-1) * np.arange(numpoints)
        full_solution = scipy.integrate.odeint(calculate_derivatives, species, t, atol=1.0e-8, rtol=1.0e-8)
        # We only want the results for RNAs
        solution = full_solution[-1][:n]

        return solution

    return numerical_solution


def simulate(input_file: str, output_file: str, n: int):
    simulation = stochpy.SSA()
    simulation.Model(input_file)
    simulation.DoStochSim(trajectories=1000)
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