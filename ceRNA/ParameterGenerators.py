"""
Provides various functions for generating parameters. Incidentally, after writing this I would happily kill
for some type classes.
"""

import scipy.stats as stats
import numpy as np
import random
from copy import deepcopy
from itertools import product
import ceRNA.Constants as Constants


def generate_iid_species_rates(x: int, y: int,
                               distribution: stats.rv_continuous=stats.truncnorm,
                               parameters: np.ndarray=np.array([5, 15])):
    """
    Generate a value drawn from the given distribution for each species.
    Used to generate arrival, decay, and bursting rates.
    """
    rates = distribution.rvs(*parameters, size=(x+y))
    return rates


def generate_type_differentiated_rates(x: int, y: int,
                                       mRNA_distribution: stats.rv_continuous=stats.truncnorm,
                                       mRNA_parameters: np.ndarray=np.array([5, 15]),
                                       miRNA_distribution: stats.rv_continuous=stats.truncnorm,
                                       miRNA_parameters: np.ndarray=np.array([5, 15])) -> np.ndarray:
    """
    Generates a value for each species. Values for mRNA species are drawn from mRNA distribution, likewise for
    miRNAs species.
    Used to get arrival, decay, and burst rates.
    """
    rates = np.zeros(x+y)
    rates[:x] = mRNA_distribution.rvs(*mRNA_parameters, size=x)
    rates[x:] = miRNA_distribution.rvs(*miRNA_parameters, size=y)
    return rates


def generate_iid_gammas(x: int, y: int,
                        distribution: stats.rv_continuous=stats.truncnorm,
                        parameters: np.ndarray=np.array([5, 15])) -> np.ndarray:
    """
    Generates iid gamma rates between each mRNA and miRNA. Rates are drawn from the given distribution with the given
    parameters.
    """
    rates = np.zeros([x+y, x+y])
    mRNA_to_miRNA = np.zeros([x, y])
    for row in range(x):
        mRNA_to_miRNA[row] = distribution.rvs(*parameters, size=y)

    #  Remove some gamma values
    legal = False
    culled_gammas = cull_gammas(x, y, mRNA_to_miRNA)

    rates[:x, x:] = culled_gammas
    rates[x:, :x] = culled_gammas.T

    return rates


def cull_gammas(x, y, mRNA_to_miRNA: np.ndarray) -> np.ndarray:
    """
    Removes gammas (sets them to 0) in such a way that the network remains connected. Currently very hacky: uses
    DFS to ensure connectivity. Would work better using a min-cut algorithm.
    """
    legal = False
    while not legal:
        temp_rates = deepcopy(mRNA_to_miRNA)
        number_to_cull = int(.1 * x * y)
        gamma_pairs = list(product(range(x), range(y)))
        interactions_to_remove = random.sample(gamma_pairs, number_to_cull)
        for interactions in interactions_to_remove:
            temp_rates[interactions[0]][interactions[1]] = 0.0
        legal = check_network_legality(x, y, temp_rates)
    return temp_rates


def check_network_legality(x: int, y: int, rates: np.ndarray):
    """
    Runs DFS. Returns false if the graph is disconnected.
    """
    not_reached = list(range(x+y))
    shadowed = [0]
    while shadowed:
        for node in shadowed:
            shadowed.remove(node)
            if node >= x:
                neighbors = np.nonzero(rates[:, node-x])
            else:
                neighbors = np.add(np.nonzero(rates[node]), x)
            for neighbor in neighbors[0]:
                if neighbor in not_reached:
                    shadowed.append(neighbor)
                    not_reached.remove(neighbor)
    return len(not_reached) == 0


def generate_iid_dissociation_rates(x: int, y: int,
                                    gammas: np.ndarray,
                                    distribution: stats.rv_continuous=stats.truncnorm,
                                    parameters: np.ndarray=np.array([5, 15])) -> np.ndarray:
    """
    Generates rates for complex dissociation.
    """
    rates = np.zeros([x+y, x+y])

    return rates