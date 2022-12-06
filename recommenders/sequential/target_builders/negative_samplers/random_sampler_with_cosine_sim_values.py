import logging
import numpy as np
import tqdm
from .negatives_sampler import NegativesSampler 
class RandomNegativesWithCosSimValues(NegativesSampler):
    def __init__(self, sample_size=400, seed=31337, lookback=50, use_self=True):
        self.sample_size = sample_size
        self.random = np.random.default_rng(seed)
        self.lookback = lookback
        self.use_self = use_self


    def set_train_sequences(self, train_sequences):
        eps = 1E-6
        smoothing = 1E-3
        items_counter = np.full(self.n_items, eps)
        item_pairs_counter = np.zeros((self.n_items, self.n_items)) 
        logging.info("building cosine similarity matrix...")
        for i in tqdm.tqdm(range(len(train_sequences)), ascii=True):
            seq = train_sequences[i]
            for j in range(0, len(seq)):
                items_counter[seq[j][1]] += 1
            for j in range(0, len(seq)):
                k_min = max(j - self.lookback, 0)
                k_max = j + 1 if self.use_self else j 
                for k in range(k_min, k_max):
                    item_pairs_counter[seq[j][1], seq[k][1]] += 1
        items_counter = np.expand_dims(items_counter, 0)
        squared = item_pairs_counter * item_pairs_counter
        self.sims = (squared / (items_counter) / np.transpose(items_counter))
        self.probs = (1 - self.sims) + smoothing
        norm = np.expand_dims(np.sum(self.probs, axis=1), 1)
        self.probs = self.probs / norm
        self.all_items = np.arange(self.n_items)

    def sample_negatives(self, positive):
        probs = self.probs[positive]
        selected = np.random.choice(self.all_items, self.sample_size, p=probs, replace=True)
        values = self.sims[positive][selected]
        return selected, values

    def top_sims(self, positive):
        best = np.argsort(self.sims[positive])[::-1]
        scores = self.sims[positive][best]
        return best, scores
        