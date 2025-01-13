from enum import Enum

"""
Contains all the different ways of selecting a subset of neighbours from all possible neighbours.
"""
class NeighbourSelectionMechanism(Enum):
    ALL = 0, 
    NEAREST = 1,
    FARTHEST = 2,
    LEAST_ORIENTATION_DIFFERENCE = 3,
    HIGHEST_ORIENTATION_DIFFERENCE = 4,
    RANDOM = 5