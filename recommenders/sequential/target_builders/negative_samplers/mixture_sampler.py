from typing import List
from .negatives_sampler import NegativesSampler

class MixtureSampler(NegativesSampler):
    def __init__(self, samplers: List[NegativesSampler]):
        self.samplers = samplers
        self.sample_size = 0
        for sampler in self.samplers:
            self.sample_size += sampler.get_sample_size()
            
    def set_n_items(self, n):
        super().set_n_items(n)
        for sampler in self.samplers:
            sampler.set_n_items(n)

    def set_train_sequences(self, train_sequences):
        for sampler in self.samplers:
            sampler.set_train_sequences(train_sequences) 
    
    def sample_negatives(self, positive):
        result = []
        scores = []
        for sampler in self.samplers:
             negatives, values =  sampler.sample_negatives(positive)
             result += list(negatives)
             scores += list(values)
        return result, scores