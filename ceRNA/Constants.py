import enum


class GammaChange(enum.Enum):
    MRNA_METHOD = 0
    GAMMA_SCALING_METHOD = 1
    GAMMA_REMOVAL_METHOD = 2


class RateTests(enum.Enum):
    WILD = "wild"
    GAMMA = "gamma"
    LAMBDA = "lambda"
    KNOCKOUT = "knockout"


class NetworkModels(enum.Enum):
    BASE = 0
    BURST = 1
    SCOM = 2
    CCOM = 3


class NetworkTopologies(enum.Enum):
    SIMPLE = 0
    DIFFERENTIATED = 1
    LARGE_GAMMAS = 2