import pickle
import numpy as np
import scipy.integrate
import stochpy


def base_network_ode_solution(ks, mus, gammas):
    # Returns a function which will calculate the derivatives
    n = len(ks)
    new_gammas = np.zeros([n, n])
    for RNA in gammas:
        for g in gammas[RNA]:
            new_gammas[RNA][g] = gammas[RNA][g]
            new_gammas[g][RNA] = gammas[RNA][g]

    def get_base_derivatives_function(ks: np.ndarray, mus: np.ndarray, gammas: dict):

        def function(species_means, sim_time):
            derivatives = np.zeros(n)
            np.add(derivatives, ks, derivatives)
            np.subtract(derivatives, np.multiply(mus, species_means), derivatives)
            np.subtract(derivatives, np.multiply(np.dot(new_gammas, species_means), species_means), derivatives)
            return derivatives

        return function

    species = [10] * len(ks)
    stop_time = 100
    numpoints = 1000
    t = [stop_time * float(i) / (numpoints - 1) for i in range(numpoints)]
    func = get_base_derivatives_function(ks, mus, gammas)
    full_solution = scipy.integrate.odeint(func, species, t, atol=1.0e-8, rtol=1.0e-8)
    solution_to_return = full_solution[len(full_solution) - 1]

    for i in range(len(solution_to_return)):
        if solution_to_return[i] <= 1e-10:
            solution_to_return[i] = 0
    return solution_to_return


def burst_network_ode_solution(ks, mus, gammas, bursts, x, y, knockouts, gamma_changes, lambda_changes, sim):
    solution_to_return = 0

    def get_burst_derivatives_function(species, time, ks, mus, gammas, alphas, betas):
        derivatives = []

        return 0
    return solution_to_return


def simulate(input_file: str, output_file: str, n: int):
    simulation = stochpy.SSA(File=input_file)
    simulation.DoCompleteStochSim()
    means = simulation.data_stochsim.species_means
    array = np.ones(2 * n)
    array[n:] = convert_stochpy_means_to_array(means)
    # Store modified data
    pickle.dump(array, open(output_file, "wb"))


def convert_stochpy_means_to_array(means: dict) -> np.ndarray:
    n = len(means)
    array = np.zeros(n)
    for RNA in range(n):
        array[RNA] = means["r{0}".format(RNA)]

    return array