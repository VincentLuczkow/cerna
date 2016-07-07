"""
Predefines a number of networks. New network topologies should be added here.
"""
import scipy.stats as stats
import numpy as np

from typing import Dict, Callable

import ceRNA.ParameterGenerators as ParameterGenerators
import ceRNA.Networks as Networks
import ceRNA.Constants as Constants


def simple_network(x: int, y: int, network_type: str):
    """
    Creates a simple network of the given size.
    """

    # Generate ks and mus
    species_distribution = stats.uniform
    parameters = np.array([5,25])
    ks = ParameterGenerators.generate_iid_species_rates(x, y, species_distribution, parameters)
    mus = ParameterGenerators.generate_iid_species_rates(x, y, species_distribution, parameters)

    # Generate gammas
    parameters = np.array([10, 20])
    gammas = ParameterGenerators.generate_iid_gammas(x, y, species_distribution, parameters)

    args = [x, y, ks, mus, gammas]

    if network_type != Constants.NetworkModels.BASE:
        # Generate burst parameters.
        parameters = np.array([5, 10])
        alphas = ParameterGenerators.generate_iid_species_rates(x, y, species_distribution, parameters)
        parameters = np.array([1, 2])
        betas= ParameterGenerators.generate_iid_species_rates(x, y, species_distribution, parameters)
        args.extend([alphas, betas])

        if network_type != Constants.NetworkModels.BURST:
            # Generate complex parameters
            parameters = np.array([10, 20])

            if network_type != Constants.NetworkModels.SCOM:
                # Generate asymmetric dissociation parameters
                pass

    network_class = Networks.network_mapping[network_type]

    # Generate the network
    network = network_class(*args)

    return network


def differentiated_network(x: int, y: int, network_type: str):
    """
    Creates a network with different rates for mRNAs and miRNAs
    """
    pass


def generate_large_gamma_network(x: int, y: int, network_type: str):
    """
    Creates a network with very high gamma rates, compared to any other rates. In general this restricts the network
    to a state where no pair of mutually degrading RNAs exists for a non-negligible amount of time.
    """
    pass


network_topology_mapping = {
    Constants.NetworkTopologies.SIMPLE : simple_network
}  # type: Dict[int, Callable]