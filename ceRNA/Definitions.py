import enum


class GammaChange(enum.Enum):
    MRNA_METHOD = 0
    GAMMA_SCALING_METHOD = 1
    GAMMA_REMOVAL_METHOD = 2


class RateTest(enum.Enum):
    WILD = 0
    GAMMA = 1
    LAMBDA = 2
    KNOCKOUT = 3