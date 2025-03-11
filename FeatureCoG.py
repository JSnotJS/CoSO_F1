from Feature import Feature
import numpy as np

class FeatureCoG(Feature):

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
        summed_positions = np.array([0,0,0])
        current_position = np.array([0,0,0])

        for letter in seq:
            current_position += self.primitives[letter]
            summed_positions += current_position

        cog = summed_positions / (len(seq) + 1)

        return cog
    
    