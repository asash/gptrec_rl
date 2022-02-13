from random import Random
from scipy.sparse import csr_matrix

import numpy as np
from aprec.recommenders.dnn_sequential_recommender.target_builders.target_builders import TargetBuilder

class ItemsMaskingTargetsBuilder(TargetBuilder):
    def __init__(self, random_seed=31337, ignore_value=-100): #-100 is used by default in hugginface's BERT implementation
        self.random = Random()
        self.random.seed(random_seed) 
        self.targets = []
        self.ignore_value = ignore_value

    def build(self, user_targets):
        targets = []
        for user in user_targets:
            user_target = [self.ignore_value] * self.sequence_len
            for pos in user:
                user_target[pos[0]] = pos[1][1]
            targets.append(user_target)
        self.targets = np.array(targets)



    def get_targets(self, start, end):
        return [self.targets[start:end]], self.targets[start:end]
