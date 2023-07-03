from typing import List

import numpy as np


class RewardMetric(object):
    less_is_better = False
    def __init__(self):
        self.name == "undefined"
    
    def get_name(self) -> str:
        return self.name

    #returns an array of rewards for each position in the recommendations
    def __call__(self, recommendations, actual) -> np.ndarray:
        raise NotImplementedError

