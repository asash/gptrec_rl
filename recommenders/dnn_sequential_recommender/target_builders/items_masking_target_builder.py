from random import Random, sample

import numpy as np
from aprec.recommenders.dnn_sequential_recommender.target_builders.target_builders import TargetBuilder

class ItemsMaskingTargetsBuilder(TargetBuilder):
    def __init__(self, random_seed=31337, 
                       relative_positions_encoding = True, 
                       ignore_value=-100, 
                       negatives_sampler=None
                       ): #-100 is used by default in hugginface's BERT implementation
        self.random = Random()
        self.random.seed(random_seed) 
        self.targets = []
        self.ignore_value = ignore_value
        self.relative_positions_encoding = relative_positions_encoding
        self.negatives_sampler = None
        self.positions = []
        self.negatives_sampler = negatives_sampler

    def set_n_items(self, n):
        super().set_n_items(n)
        if self.negatives_sampler is not None:
            self.negatives_sampler.set_n_items(n)

    def build(self, user_targets):
        targets = []
        positions = []
        negatives = []
        for seq_len, user in user_targets:
            user_positions = []
            if self.negatives_sampler is not None:
                user_negatives = [self.negatives_sampler.default_vector()] * self.sequence_len

            user_target = [self.ignore_value] * self.sequence_len
            if self.relative_positions_encoding:
                split_pos = self.random.randint(self.sequence_len - seq_len, self.sequence_len - 1)
            else:
                split_pos = self.sequence_len - 1

            for i in range(self.sequence_len):
                user_positions.append(self.sequence_len - split_pos  + i) 

            positions.append(user_positions)
            for pos in user:
                user_target[pos[0]] = pos[1][1]
                if self.negatives_sampler is not None:
                    user_negatives[pos[0]] = self.negatives_sampler.sample_negatives(pos[1][1])
            targets.append(user_target)
            if self.negatives_sampler is not None:
                negatives.append(user_negatives)
        self.positions = np.array(positions)
        self.targets = np.array(targets)
        self.negatives = np.array(negatives)

    def get_targets(self, start, end):
        if self.negatives_sampler is None:
            return [self.targets[start:end], self.positions[start:end]], self.targets[start:end]
        else:
            return [self.targets[start:end], self.positions[start:end], self.negatives[start:end]], self.targets[start:end]


class RandomNegativesSampler(object):
    def __init__(self, sample_size, seed=31337, default_value=-100):
        self.sample_size = sample_size
        self.random = Random()
        self.random.seed(seed) 
        self.def_vector = [default_value] * sample_size 
    
    def default_vector(self):
        return self.def_vector

    def set_n_items(self, n):
        self.n_items = n

    def sample_negatives(self, positive):
        result = []
        while len(result) < self.sample_size:
            sample = self.random.randint(0, self.n_items - 1)
            if sample != positive:
                result.append(sample)
        return result

    