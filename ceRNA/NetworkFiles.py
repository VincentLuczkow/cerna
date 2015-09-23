import getpass
from getpass import getuser
import random
from copy import deepcopy
import re

import numpy as np


# Get creation reactions for x mRNAs and y miRNAs, using the base model.
def get_base_creation_and_decay_reactions_string(ks, mus):
    string = "#Reactions\n"
    for RNA in range(len(ks)):
        string += "R" + str(RNA) + ":\n"
        string += "\t$pool > r" + str(RNA) + "\n"
        string += "\tk" + str(RNA) + "\n\n"
    reaction_count = len(ks)
    for RNA in range(len(mus)):
        string += "R" + str(RNA + reaction_count) + ":\n"
        string += "\tr" + str(RNA) + " > $pool\n"
        string += "\tmu" + str(RNA) + "*r" + str(RNA) + "\n\n"
    return string


# Get a string setting the creation and decay reactions for every
# RNA and every burst factor in a burst network.
# ks is the array of creation parameters.
# mus is the array of decay parameters,
def get_burst_creation_and_decay_reactions_string(ks, mus):
    string = "#Reactions\n"

    for RNA in range(len(ks)):
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
    reaction_count = 3 * len(ks)
    for RNA in range(len(mus)):
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
def get_base_species_initialization_string(ks):
    species_string = "#Species\n"
    for RNA in range(len(ks)):
        species_string += "    r{0} = 10\n".format(RNA)
    return species_string


# Get a string setting the initial species amounts for
# n burst variables, where n is the number of RNAs in the network.
def get_burst_species_initialization_string(ks):
    species_string = ""
    for RNA in range(len(ks)):
        # All species start in off state.
        species_string += "\tOFF" + str(RNA) + " = 1\n"
        species_string += "\tON" + str(RNA) + " = 0\n"
    species_string += "\n"
    return species_string


# Get a string setting the creation and decay parameters for every RNA
# in the network.
# ks is an array giving the creation parameters.
# mus is an array giving the decay parameters.
def get_base_creation_and_decay_parameters_string(ks, mus):
    parameter_string = "#Parameters\n"
    # For every RNA in the network
    for RNA in range(len(ks)):
        # Add its creation parameter
        parameter_string += "\tk" + str(RNA) + " = " + str(ks[RNA]) + "\n"
        # Add its decay parameter
        parameter_string += "\tmu" + str(RNA) + " = " + str(mus[RNA]) + "\n"
    return parameter_string


# Get a string setting all gamma interaction parameters for the network.
# x is the number of mRNAs in the network.
# gammas is a dictionary containing the gamma interactions.
def get_gamma_parameters_string(x, gammas):
    parameter_string = ""
    for RNA in range(x):
        try:
            for g in gammas[RNA]:
                parameter_string += "\tgammar" + str(RNA) + "r" + str(g) + " = " + str(gammas[RNA][g]) + "\n"
        except KeyError:
            pass
    return parameter_string


# Get parameter strings for a set of burst modes
def get_burst_parameters_string(ks, get_alpha, get_beta):
    parameter_string = ""
    for RNA in range(len(ks)):
        alpha = get_alpha(RNA)
        beta = get_beta(RNA)
        parameter_string += "\talpha" + str(RNA) + " = " + str(alpha) + "\n"
        parameter_string += "\tbeta" + str(RNA) + " = " + str(beta) + "\n"
    return parameter_string


def get_constant_alpha(alpha):
    return lambda x: alpha


def get_constant_beta(beta):
    return lambda x: beta


# Generate complete network for x mRNAs and y sRNAs.
def create_base_network_file(x, y, ks, mus, gammas, file_name):
    out_string = "#Random complete network of size " + str(x) + ", " + str(y) + ".\n"

    # Remove any knockouts
    new_x, new_ks, new_mus, new_gammas = remove_knockouts(x, ks, mus, gammas)
    n = len(new_ks)

    # Reaction Strings
    out_string += get_base_creation_and_decay_reactions_string(new_ks, new_mus)
    out_string += get_gamma_reactions_string(new_x, new_gammas, 2 * n)

    # Species Strings
    out_string += get_base_species_initialization_string(new_ks)

    # Parameter Strings
    out_string += get_base_creation_and_decay_parameters_string(new_ks, new_mus)
    out_string += get_gamma_parameters_string(new_x, new_gammas)

    network_file = open("/home/" + getpass.getuser() + "/Stochpy/pscmodels/" + file_name + ".psc", "w")
    network_file.write(out_string)
    network_file.close()


# Knockout species (k_i = 0) must be genuinely removed, or the simulations
# won't run as well.
def remove_knockouts(x: int, ks: np.ndarray, mus: np.ndarray, gammas: dict) -> tuple:
    new_x = 0
    new_ks = []
    new_mus = []
    new_gammas = {}
    for species in range(len(ks)):
        # If the species has not been knockout out, add it to the actual network
        if ks[species] != 0:
            # Keep track of how many mRNAs are in the new network.
            if species < x:
                new_x += 1
            new_ks.append(ks[species])
            new_mus.append(mus[species])
            new_gammas[species] = gammas[species]
    new_ks = np.array(new_ks)
    new_mus = np.array(new_mus)
    return new_x, new_ks, new_mus, new_gammas


def create_burst_network_file(x, y, ks, mus, gammas, alpha=.5, beta=.5):
    out_string = "#Random complete burst network of size " + str(x) + ", " + str(y) + ".\n"

    # Reaction Strings
    out_string += get_burst_creation_and_decay_reactions_string(ks, mus)
    out_string += get_gamma_reactions_string(x, gammas, 4 * (x + y))

    # Species Strings
    out_string += get_base_species_initialization_string(ks)
    out_string += get_burst_species_initialization_string(ks)

    # Parameter Strings
    out_string += get_base_creation_and_decay_parameters_string(ks, mus)
    out_string += get_gamma_parameters_string(x, gammas)
    out_string += get_burst_parameters_string(ks, get_constant_alpha(alpha), get_constant_beta(beta))

    network_file = open(
        "/home/" + getpass.getuser() + "/Stochpy/pscmodels/RNABurstNetwork" + str(x) + "," + str(y) + ".psc", "w")
    network_file.write(out_string)
    network_file.close()


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