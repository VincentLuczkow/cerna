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
    string = "#Reactions\n"
    reaction_count = 0
    for RNA in range(len(ks)):
        string += "R{}:\n".format(reaction_count)
        string += "    $pool > r{}\n".format(RNA)
        string += "    k{}\n\n".format(RNA)
        reaction_count += 1
    for RNA in range(len(mus)):
        string += "R{}:\n".format(reaction_count)
        string += "    r{} > $pool\n".format(RNA)
        string += "    mu{} * r{}\n\n".format(RNA, RNA)
        reaction_count += 1
    return string


# Get a string setting the creation and decay reactions for every
# RNA and every burst factor in a burst network.
# ks is the array of creation parameters.
# mus is the array of decay parameters,
def burst_creation_and_decay_reactions(n):
    string = "#Reactions\n"

    for RNA in range(n):
        # The reaction from the off burst state to the on burst state.
        # Reaction number
        string += "R" + str(3 * RNA) + ":\n"
        # Reaction equation
        string += "\tOFF" + str(RNA) + " > ON" + str(RNA) + "\n"
        # Reaction rate
        string += "\talpha" + str(RNA) + "*OFF" + str(RNA) + "\n"

        # The reaction from the on burst state to the off burst state.
        # Reaction number
        string += "R" + str(3 * RNA + 1) + ":\n"
        string += "\tON" + str(RNA) + " > OFF" + str(RNA) + "\n"
        string += "\tbeta" + str(RNA) + "*ON" + str(RNA) + "\n"

        # Creation of new RNA
        string += "R" + str(3 * RNA + 2) + ":\n"
        string += "\tON" + str(RNA) + " > ON" + str(RNA) + " + r" + str(RNA) + "\n"
        string += "\tk" + str(RNA) + "*ON" + str(RNA) + "\n"

    # Degradation of RNA
    reaction_count = 3 * n
    for RNA in range(n):
        string += "R" + str(RNA + reaction_count) + ":\n"
        string += "\tr" + str(RNA) + " > $pool\n"
        string += "\tmu" + str(RNA) + "*r" + str(RNA) + "\n\n"

    return string


# Get a string setting all of the gamma interactions
# in the network.
# gammas is a dictionary containing the interactions and their strengths
# x is the number of mRNAs in the network
# reaction_count is the number of reactions that have already been added to the file
def get_gamma_reactions_string(x, gammas, reaction_count):
    reaction_string = ""
    # For every mRNA
    for mRNA in range(x):
        # Not all values are guaranteed to be instantiated
        try:
            for miRNA in gammas[mRNA]:
                reaction_string += "R{0}:\n".format(reaction_count)
                reaction_string += "    r{0} + r{1} > $pool\n".format(mRNA, miRNA)
                reaction_string += "    r{0} * r{1} * gammar{0}r{1}\n\n".format(mRNA, miRNA)
                reaction_count += 1
        except KeyError:
            pass
    return reaction_string


# Get a string setting the initial species amounts
# for n RNAs, where n is the number of RNAs in the network.
def get_base_species_amount(ks):
    species_string = "#Species\n"
    for RNA in range(len(ks)):
        species_string += "    r{0} = 10\n".format(RNA)
    return species_string


# Get a string setting the initial species amounts for
# n burst variables, where n is the number of RNAs in the network.
def get_burst_species_amounts(ks):
    species_string = ""
    for RNA in range(len(ks)):
        # All species start in off state.
        species_string += "    OFF{} = 1\n".format(RNA)
        species_string += "    ON{} = 0\n".format(RNA)
    species_string += "\n"
    return species_string


# Get a string setting the creation and decay parameters for every RNA
# in the network.
# ks is an array giving the creation parameters.
# mus is an array giving the decay parameters.
def base_arrival_and_decay_parameters(ks: np.ndarray, mus: np.ndarray) -> str:
    parameter_string = "#Parameters\n"
    # For every RNA in the network
    for RNA in range(len(ks)):
        parameter_string += "    k{} = {}\n".format(RNA, ks[RNA])
        parameter_string += "    mu{} = {}\n".format(RNA, mus[RNA])
    return parameter_string


# Get a string setting all gamma interaction parameters for the network.
# x is the number of mRNAs in the network.
# gammas is a dictionary containing the gamma interactions.
def gamma_parameters(x, gammas):
    parameter_string = ""
    for RNA in range(x):
        try:
            for g in gammas[RNA]:
                parameter_string += "\tgammar" + str(RNA) + "r" + str(g) + " = " + str(gammas[RNA][g]) + "\n"
        except KeyError:
            pass
    return parameter_string


# Get parameter strings for a set of burst modes
def burst_creation_and_decay_parameters(alphas, betas):
    n = len(alphas)
    parameter_string = ""
    for burst_factor in range(n):
        alpha = alphas[burst_factor]
        beta = betas[burst_factor]
        parameter_string += "\talpha{0} = {1}\n".format(burst_factor, alpha)
        parameter_string += "\tbeta{0} = {1}\n".format(burst_factor, beta)
    return parameter_string


def get_constant_alpha(alpha):
    return lambda x: alpha


def get_constant_beta(beta):
    return lambda x: beta


# Generate complete network for x mRNAs and y sRNAs.
def create_base_network_file(x: int, y: int,
                             ks: np.ndarray, mus: np.ndarray, gammas: np.ndarray,
                             file_name: str):
    out_string = "#Random base network of size " + str(x) + ", " + str(y) + ".\n"

    # Remove any knockouts
    if not gammas:
        pdb.set_trace()
    new_x, new_ks, new_mus, new_gammas = remove_knockouts(x, ks, mus, gammas)
    n = len(new_ks)

    # Reaction Strings
    out_string += base_creation_and_decay_reactions(new_ks, new_mus)
    out_string += get_gamma_reactions_string(new_x, new_gammas, 2 * n)

    # Species Strings
    out_string += get_base_species_amount(new_ks)

    # Parameter Strings
    out_string += base_arrival_and_decay_parameters(new_ks, new_mus)
    out_string += gamma_parameters(new_x, new_gammas)

    network_file = open("/home/" + getpass.getuser() + "/Stochpy/pscmodels/" + file_name + ".psc", "w")
    network_file.write(out_string)
    network_file.close()


def get_base_network_writer(x: int, y: int):
    def file_writer(ks: np.ndarray, mus: np.ndarray, gammas: np.ndarray,
                    file_name: str):
        out_string = "#Random base network of size " + str(x) + ", " + str(y) + ".\n"

        # Remove any knockouts
        new_x, new_ks, new_mus, new_gammas = remove_knockouts(x, ks, mus, gammas)
        n = len(new_ks)

        # Reaction Strings
        out_string += base_creation_and_decay_reactions(new_ks, new_mus)
        out_string += get_gamma_reactions_string(new_x, new_gammas, 2 * n)

        # Species Strings
        out_string += get_base_species_amount(new_ks)

        # Parameter Strings
        out_string += base_arrival_and_decay_parameters(new_ks, new_mus)
        out_string += gamma_parameters(new_x, new_gammas)

        network_file = open("/home/" + getpass.getuser() + "/Stochpy/pscmodels/" + file_name + ".psc", "w")
        network_file.write(out_string)
        network_file.close()

    return file_writer


def get_burst_network_writer(x: int, y: int,
                             alphas: np.ndarray, betas: np.ndarray) -> Callable[..., type(None)]:

    def file_writer(ks: np.ndarray, mus: np.ndarray, gammas: np.ndarray, file_name: str) -> type(None):
        n = x + y
        out_string = "#Random burst network of size " + str(x) + ", " + str(y) + ".\n"

        # Reaction Strings
        out_string += burst_creation_and_decay_reactions(n)
        out_string += get_gamma_reactions_string(x, gammas, 4 * (x + y))

        # Species Strings
        out_string += get_base_species_amount(ks)
        out_string += get_burst_species_amounts(ks)

        # Parameter Strings
        out_string += base_arrival_and_decay_parameters(ks, mus)
        out_string += gamma_parameters(x, gammas)
        out_string += burst_creation_and_decay_parameters(alphas, betas)

        network_file = open(file_name, "w")
        network_file.write(out_string)
        network_file.close()

    return file_writer


def get_complex_network_writer(x: int, y: int,
                               alphas: np.ndarray, betas: np.ndarray, deltas: np.ndarray) -> Callable[..., type(None)]:

    def file_writer(ks: np.ndarray, mus: np.ndarray, gammas: np.ndarray, file_name: str) -> type(None):
        n = x + y
        out_string = "#Random complete burst network of size " + str(x) + ", " + str(y) + ".\n"

        # Reaction Strings
        out_string += burst_creation_and_decay_reactions(n)
        out_string += get_gamma_reactions_string(x, gammas, 4 * (x + y))

        # Species Strings
        out_string += get_base_species_amount(ks)
        out_string += get_burst_species_amounts(ks)

        # Parameter Strings
        out_string += base_arrival_and_decay_parameters(ks, mus)
        out_string += gamma_parameters(x, gammas)
        out_string += burst_creation_and_decay_parameters(ks, betas)

        network_file = open(file_name, "w")
        network_file.write(out_string)
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
    for species in [x for y in ks if y != 0]:
        # Keep track of how many mRNAs are in the new network.
        if species < x:
            new_x += 1
        new_ks.append(ks[species])
        new_mus.append(mus[species])
        new_gammas[species] = gammas[species]

    new_ks = np.array(new_ks)
    new_mus = np.array(new_mus)
    return new_x, new_ks, new_mus, new_gammas


def read_base_network_file(model_name):
    with open("/home/" + getuser() + "/Stochpy/pscmodels/" + model_name + ".psc", "r") as model_file:
        lines = model_file.readlines()

    # Skip through most of the file to the Parameters section
    for i in range(len(lines)):
        if re.match(r'#Parameters', lines[i].strip()):
            lines = lines[i:]
            break

    x = 0
    y = 0
    creation_rates_dict = {}
    degradation_rates_dict = {}
    gammas = {}
    for line in lines:
        strings = line.rstrip().split()
        parameter = re.match(r'(\D+)(\d+)\D*(\d*)', strings[0])

        # If we have actually seen a match.
        if parameter:
            parameter_type = parameter.group(1)
            parameter_number = int(parameter.group(2))

            if parameter_type == "k":
                creation_rates_dict[parameter_number] = float(strings[2])
            elif parameter_type == "mu":
                degradation_rates_dict[parameter_number] = float(strings[2])
            elif parameter_type == "gammar":
                # If this is the first time this mRNA has appeared in a gamma.
                if parameter_number not in gammas:
                    gammas[parameter_number] = {}
                    x += 1
                # If this is the first time this miRNA has appeared in a gamma.
                if int(parameter.group(3)) not in gammas:
                    gammas[int(parameter.group(3))] = {}
                    y += 1
                # Add the mRNA and the miRNA to the adjency list of the other.
                gammas[parameter_number][int(parameter.group(3))] = float(strings[2])
                gammas[int(parameter.group(3))][parameter_number] = float(strings[2])

    # We want creation and degradation rates as arrays, so we convert them here.
    # We can't do this initially because x and y are unknown before we have
    # finished reading the file.
    ks = np.empty(x + y)
    mus = np.empty(x + y)
    for i in range(x + y):
        ks[i] = creation_rates_dict[i]
        mus[i] = degradation_rates_dict[i]
    return ks, mus, gammas