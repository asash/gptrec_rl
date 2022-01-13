from aprec.recommenders.dnn_sequential_recommender.targetsplitters.targetsplitter import TargetSplitter
from random import Random

def exponential_importance(p):
    return lambda n, k: p**(n - k)

class RecencySequenceSampling(TargetSplitter):
    #recency importance is a function that defines the chances of k-th element 
    #to be sampled as a positive in the sequence of the length n

    def __init__(self, max_pct, recency_importance=lambda n, k: 0.8 ** (n - k), seed=31337) -> None:
        super().__init__()
        self.max_pct = max_pct
        self.recency_iportnace = recency_importance
        self.random = Random()
        self.random.seed(seed)

    
    def split(self, sequence):
        sampled_idx = set()
        target = set() 
        cnt = max(1, int(len(sequence)*self.max_pct))
        sampled_f_sum = 0.0
        f = lambda j: self.recency_iportnace(len(sequence), j)
        f_sum = sum(f(i) for i in range(len(sequence)))
        for i in range(cnt):
            val = self.random.random() * (f_sum)
            running_sum = 0.0

            #we iterate backwards because chances are
            #higher of finding the sampled
            # element close to the end of the sequence 

            for j in range(len(sequence) - 1, -1, -1):
                if running_sum + f(j) > val:
                    sampled_idx.add(j)
                    target.add(sequence[j])
                    sampled_f_sum += f(j)
                    break
                running_sum += f(j)
        input = list() 
        for i in range(len(sequence)):
            if i not in sampled_idx:
                input.append(sequence[i])
        return input, list(target)


