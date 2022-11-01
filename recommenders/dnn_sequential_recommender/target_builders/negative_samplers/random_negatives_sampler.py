from random import Random
from aprec.recommenders.dnn_sequential_recommender.target_builders.negative_samplers.negatives_sampler import NegativesSampler

class RandomNegativesSampler(NegativesSampler):
    def __init__(self, sample_size, seed=31337):
        self.sample_size = sample_size
        self.random = Random()
        self.random.seed(seed) 
        self.values = [0] * self.sample_size 
    
    def sample_negatives(self, positive):
        result = []
        while len(result) < self.sample_size:
            sample = self.random.randint(0, self.n_items - 1)
            if sample != positive:
                result.append(sample)
        return result, self.values
