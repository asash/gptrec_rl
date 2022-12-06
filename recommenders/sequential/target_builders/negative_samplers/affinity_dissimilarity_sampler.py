import logging
import numpy as np
import tqdm
from .negatives_sampler import NegativesSampler 
class AffinityDissimilaritySampler(NegativesSampler):
    def __init__(self, sample_size=400, seed=31337, smoothing=5):
        self.sample_size = sample_size
        self.random = np.random.default_rng(seed)
        self.smoothing = smoothing
        self.values = [0] * self.sample_size

    def set_train_sequences(self, train_sequences):
        EPS = self.smoothing 
        items_counter = np.zeros(self.n_items)
        item_pairs_counter = np.full((self.n_items, self.n_items), EPS) 
        logging.info("building dissimilarity matrix...")
        for i in tqdm.tqdm(range(len(train_sequences)), ascii=True):
            items = {item[1] for item in train_sequences[i]}
            for item1 in items:
                items_counter[item1] += 1
                for item2 in items:
                    item_pairs_counter[item1, item2] += 1 
                    if item1 != item2:
                        item_pairs_counter[item2, item1] += 1 
        
        dissim = items_counter / item_pairs_counter
        for i in range(self.n_items):
            dissim[i][i] = 0
        norm = np.expand_dims(np.sum(dissim, axis=1), 1)
        dissim = dissim / norm
        self.dissim = dissim
        self.all_items = np.arange(self.n_items)

    def sample_negatives(self, positive):
        result = np.random.choice(self.all_items, self.sample_size, p=self.dissim[positive], replace=True)
        return result, self.values
