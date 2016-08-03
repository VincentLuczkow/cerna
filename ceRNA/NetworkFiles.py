import getpass
from getpass import getuser
from typing import Callable, Tuple
import pdb
import random
from copy import deepcopy
import re

import numpy as np


# Get creation reactions for x mRNAs and y miRNAs, using the base model.
def base_creation_and_decay_reactions(ks, mus):
    string_list = ["#Reactions\n"]
    reaction_count = 0

    for RNA in range(len(ks)):
        string_list.append("R{}:\n".format(reaction_count))
        string_list.append("    $pool > r{}\n".format(RNA))
        string_list.append("    k{}\n\n".format(RNA))
        reaction_count += 1

    for RNA in range(len(mus)):
        string_list.append("R{}:\n".format(reaction_count))
        string_list.append("    r{} > $pool\n".format(RNA))
        string_list.append("    mu{} * r{}\n\n".format(RNA, RNA))
        reaction_count += 1

    string = ''.join(string_list)
    return string


# Get a string setting the creation and decay reactions for every
# RNA and every burst factor in a burst network.
# ks is the array of creation parameters.
# mus is the array of decay parameters,
def burst_creation_and_decay_reactions(n):
    string_list = ["#Reactions\n"]

    for RNA in range(n):
        # The reaction from the off burst state to the on burst state.
        # Reaction number
        string_list.append("R{}:\n".format(3*RNA))
        # Reaction equation
        string_list.append("    OFF{0} > ON{0}\n".format(RNA))
        # Reaction rate
        string_list.append("    alpha{0} * OFF{0}\n".format(RNA))

        # The reaction from the on burst state to the off burst state.
        # Reaction number
        string_list.append("R{}:\n".format(3*RNA+1))
        string_list.append("    ON{0} > OFF{0}\n".format(RNA))
        string_list.append("    beta{0} * ON{0}\n".format(RNA))

        # Creation of new RNA
        string_list.append("R{}:\n".format(3*RNA+2))
        string_list.append("    ON{0} > ON{0} + r{0}\n".format(RNA))
        string_list.append("    k{0} * ON{0}\n".format(RNA))

    # Degradation of RNA
    reaction_count = 3 * n
    for RNA in range(n):
        string_list.append("R{}:\n".format(RNA + reaction_count))
        string_list.append("    r{} > $pool\n".format(RNA))
        string_list.append("    mu{0} * r{0}\n\n".format(RNA))

    string = ''.join(string_list)
    return string


# Get a string setting all of the gamma interactions
# in the network.
# gammas is a dictionary containing the interactions and their strengths
# x is the number of mRNAs in the network
# reaction_count is the number of reactions that have already been added to the file
def get_gamma_reactions_string(x, gammas, reaction_count):
    string_list = []
    # For every mRNA
    for mRNA in gammas:
        # Not all values are guaranteed to be instantiated
        for miRNA, rate in gammas[mRNA].items():
            if rate > 0:
                string_list.append("R{}:\n".format(reaction_count))
                string_list.append("    r{} + r{} > $pool\n".format(mRNA, miRNA))
                string_list.append("    r{0} * r{1} * gammar{0}r{1}\n\n".format(mRNA, miRNA))
                reaction_count += 1
    string = ''.join(string_list)
    return string


# Get a string setting the initial species amounts
# for n RNAs, where n is the number of RNAs in the network.
def get_base_species_amounts(ks):
    string_list = ["#Species\n"]

    for RNA in range(len(ks)):
        # If the species has not been knocked out, set its initial value to 0.
        amount = 10 if ks[RNA] > 0 else 0
        string_list.append("    r{0} = {1}\n".format(RNA, amount))
    string = ''.join(string_list)
    return string


# Get a string setting the initial species amounts for
# n burst variables, where n is the number of RNAs in the network.
def get_burst_species_amounts(ks):
    string_list = []
    for RNA in range(len(ks)):
        # All species start in off state.
        string_list.append("    OFF{} = 1\n".format(RNA))
        string_list.append("    ON{} = 0\n".format(RNA))
    string_list.append("\n")
    string = ''.join(string_list)
    return string


# Get a string setting the creation and decay parameters for every RNA
# in the network.
# ks is an array giving the creation parameters.
# mus is an array giving the decay parameters.
def base_arrival_and_decay_parameters(ks: np.ndarray, mus: np.ndarray) -> str:
    string_list = ["#Parameters\n"]
    # parameter_string = "#Parameters\n"
    # For every RNA in th e network
    for RNA in range(len(ks)):
        string_list.append("    k{} = {}\n".format(RNA, ks[RNA]))
        string_list.append("    mu{} = {}\n".format(RNA, mus[RNA]))
    string = ''.join(string_list)
    return string


# Get a string setting all gamma interaction parameters for the network.
# x is the number of mRNAs in the network.
# gammas is a dictionary containing the gamma interactions.
def gamma_parameters(x, gammas):
    string_list = []
    # parameter_string = ""
    for RNA in gammas:
        for miRNA, rate in enumerate(gammas[RNA]):
            if rate > 0:
                string_list.append("    gammar{0}r{1} = {2}\n".format(RNA, miRNA, rate))
    string = ''.join(string_list)
    return string


# Get parameter strings for a set of burst modes
def burst_creation_and_decay_parameters(alphas, betas):
    string_list = []
    for species in range(len(alphas)):
        alpha = alphas[species]
        beta = betas[species]
        string_list.append("    alpha{0} = {1}".format(species, alpha))
        string_list.append("    beta{0} = {1}\n".format(species, beta))

    string = ''.join(string_list)
    return string


def get_base_network_writer(x: int, y: int):
    def file_writer(ks: np.ndarray, mus: np.ndarray, gammas: np.ndarray,
                    file_name: str):
        out_string_list = ["#Random base network with {0} mRNAs and {1} miRNAs\n"]

        # Remove any knockouts
        new_x, new_ks, new_mus, new_gammas = remove_knockouts(x, ks, mus, gammas)
        n = len(new_ks)

        # Reaction Strings
        out_string_list.append(base_creation_and_decay_reactions(new_ks, new_mus))
        out_string_list.append(get_gamma_reactions_string(new_x, new_gammas, 2 * n))

        # Species Strings
        out_string_list.append(get_base_species_amounts(ks))

        # Parameter Strings
        out_string_list.append(base_arrival_and_decay_parameters(new_ks, new_mus))
        out_string_list.append(gamma_parameters(new_x, new_gammas))

        out_string = ''.join(out_string_list)

        network_file = open("/home/" + getpass.getuser() + "/Stochpy/pscmodels/" + file_name + ".psc", "w")
        network_file.write(out_string)
        network_file.close()

    return file_writer


def get_burst_network_writer(x: int, y: int,
                             alphas: np.ndarray, betas: np.ndarray) -> Callable[..., type(None)]:

    def file_writer(ks: np.ndarray, mus: np.ndarray, gammas: np.ndarray, file_name: str) -> type(None):
        n = x + y
        out_string_list = ["#Random burst network with {0} mRNAs and {1} miRNAs".format(x, y)]
        # out_string = "#Random burst network of size " + str(x) + ", " + str(y) + ".\n"

        # Reaction Strings
        out_string_list.append(burst_creation_and_decay_reactions(n))
        # out_string += burst_creation_and_decay_reactions(n)
        # out_string += get_gamma_reactions_string(x, gammas, 4 * (x + y))
        out_string_list.append(get_gamma_reactions_string(x, gammas, 4 * (x + y)))

        # Species Strings
        # out_string += get_base_species_amounts(ks)
        out_string_list.append(get_base_species_amounts(ks))
        # out_string += get_burst_species_amounts(ks)
        out_string_list.append(get_burst_species_amounts(ks))

        # Parameter Strings
        # out_string += base_arrival_and_decay_parameters(ks, mus)
        out_string_list.append(base_arrival_and_decay_parameters(ks, mus))
        # out_string += gamma_parameters(x, gammas)
        out_string_list.append(gamma_parameters(x, gammas))
        # out_string += burst_creation_and_decay_parameters(alphas, betas)
        out_string_list.append(burst_creation_and_decay_parameters(alphas, betas))

        string = ''.join(out_string_list)
        # assert string == out_string
        network_file = open("/home/" + getpass.getuser() + "/Stochpy/pscmodels/" + file_name + ".psc", "w")
        network_file.write(string)
        network_file.close()

    return file_writer


def get_complex_network_writer(x: int, y: int,
                               alphas: np.ndarray, betas: np.ndarray, deltas: np.ndarray) -> Callable[..., type(None)]:

    def file_writer(ks: np.ndarray, mus: np.ndarray, gammas: np.ndarray, file_name: str) -> type(None):
        n = x + y

        out_string_list = ["#Random burst network with {0} mRNAs and {1} miRNAs".format(x, y)]
        # out_string = "#Random burst network of size " + str(x) + ", " + str(y) + ".\n"

        # Reaction Strings
        out_string_list.append(burst_creation_and_decay_reactions(n))
        # out_string += burst_creation_and_decay_reactions(n)
        # out_string += get_gamma_reactions_string(x, gammas, 4 * (x + y))
        out_string_list.append(get_gamma_reactions_string(x, gammas, 4 * (x + y)))

        # Species Strings
        # out_string += get_base_species_amounts(ks)
        out_string_list.append(get_base_species_amounts(ks))
        # out_string += get_burst_species_amounts(ks)
        out_string_list.append(get_burst_species_amounts(ks))

        # Parameter Strings
        # out_string += base_arrival_and_decay_parameters(ks, mus)
        out_string_list.append(base_arrival_and_decay_parameters(ks, mus))
        # out_string += gamma_parameters(x, gammas)
        out_string_list.append(gamma_parameters(x, gammas))
        # out_string += burst_creation_and_decay_parameters(alphas, betas)
        out_string_list.append(burst_creation_and_decay_parameters(alphas, betas))

        string = ''.join(out_string_list)
        # assert string == out_string
        network_file = open("/home/" + getpass.getuser() + "/Stochpy/pscmodels/" + file_name + ".psc", "w")
        network_file.write(string)
        network_file.close()

    return file_writer


# Knockout species (k_i = 0) must be genuinely removed, or the simulations
# won't run as well.
def remove_knockouts(x: int, ks: np.ndarray, mus: np.ndarray, gammas: dict,
                     alphas: np.ndarray = None, betas: np.ndarray = None, deltas: dict = None) -> tuple:
    new_x = 0
    new_ks = []
    new_mus = []
    new_gammas = {}
    if alphas:
        new_alphas = []
        new_betas = []
        if deltas:
            new_deltas = []
    non_knockouts = np.nonzero(ks)[0]
    for species in non_knockouts:
        # Keep track of how many mRNAs are in the new network.
        if species < x:
            new_x += 1
        new_ks.append(ks[species])
        new_mus.append(mus[species])
        new_gammas[species] = {x: gammas[species][x] for x in non_knockouts}


    new_ks = np.array(new_ks)
    new_mus = np.array(new_mus)
    return new_x, new_ks, new_mus, new_gammas
