from features.Feature import Feature
import numpy as np

class FeatureParenthesis(Feature):

    def __init__(self):
        ...

    def get_feature_value(self, seq):
        parenthesis = 0
        seen_parenthesis = False
        commas_before = 0
        for ch in seq:
            if not seen_parenthesis:
                if ch == ",":
                    commas_before = 1
            if ch == "(":
                seen_parenthesis = 1
                parenthesis += 1
            elif ch == ")":
                seen_parenthesis = 1
                parenthesis -= 1
        
        lastnonmod = 0
        if seq[-1] in ["X", "(", ")", ","]:
            lastnonmod = 1
        return np.array([parenthesis, commas_before, lastnonmod])
    