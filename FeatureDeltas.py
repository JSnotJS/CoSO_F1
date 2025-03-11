from Feature import Feature
import numpy as np

class FeatureDeltas(Feature):

    primitives = { 
    "U": np.array([1,0,0]),
    "D": np.array([-1,0,0]),
    "R": np.array([0,1,0]),
    "L": np.array([0,-1,0]),
    "F": np.array([0,0,1]),
    "B": np.array([0,0,-1])
    }

    def __init__(self):
        ...

    def get_feature_value(self, seq):
        delta = np.array([0,0,0])
        for ch in seq:
            delta += self.primitives[ch]
        return delta
    