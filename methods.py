from enum import Enum

class WindowMethod(Enum):
    CUMULATIVE = 1
    SHIFT = 2

class ProbabilityMethods(Enum):
    RANGE = 1
    VALUES = 2

class SelectionMethod(Enum):
    TOP = 1
    WEIGHTED = 2
    HIGHEST = 3
    RANDOM = 4
    PROB = 5

