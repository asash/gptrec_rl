from collections import Counter
import logging

import numpy as np
from aprec.recommenders.sequential.target_builders.negative_samplers.negatives_sampler import NegativesSampler


class PopularityBasedSampler(NegativesSampler):
    def __init__(self, sample_size=400, seed=31337, pool_size = 10000000):
        self.sample_size = sample_size
        self.random = np.random.default_rng(seed)
        self.pool_size = pool_size

    def set_train_sequences(self, train_sequences):
        logging.warning("building popularity-based negatves sampler...")
        items_counter = Counter()
        all_cnt = 0
        for i in range(len(train_sequences)):
            for item in train_sequences[i]:
                items_counter[item[1]] += 1
                all_cnt += 1
        items, probs = [], []
        for i, cnt in items_counter.most_common():
            items.append(i)
            probs.append(cnt/all_cnt)
        self.pool = self.random.choice(items, self.pool_size, p=probs, replace=True)
        self.pos = 0
        
    def sample_negatives(self, positive):
        if self.pos + self.sample_size >= self.pool_size:
            self.random.shuffle(self.pool)
            self.pos = 0 
        result = self.pool[self.pos: self.pos + self.sample_size]
        values = [0] * len(result)
        self.pos += self.sample_size
        return result, values