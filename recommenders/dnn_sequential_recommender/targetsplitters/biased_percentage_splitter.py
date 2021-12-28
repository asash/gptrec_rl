import numpy as np
from numpy.core.numeric import indices
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.targetsplitter import TargetSplitter


class BiasedPercentageSplitter(TargetSplitter):
    def __init__(self, max_pct, bias=0.5, max_possible_len=10000) -> None:
        super().__init__()
        self.max_pct = max_pct
        self.bias = bias
        current_bias = 1.0
        biases = []
        for i in range(max_possible_len):
            current_bias *= bias 
            biases.append(current_bias)
        self.biases = np.array(list(reversed(biases))) 

    
    def split(self, sequence):
        probs = self.biases[-len(sequence):]/np.sum(self.biases[-len(sequence):])
        indices = range(len(sequence))
        n_target_actions = max(1, int(len(sequence)*self.max_pct))
        target_indices = set(np.random.choice(indices, size=n_target_actions,p=probs, replace=True))
        train = []
        target = []
        for i, elem in enumerate(sequence):
            if i in target_indices:
                target.append(elem)
            else:
                train.append(elem)
        return train, target


