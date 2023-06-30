import math

from aprec.evaluation.metrics.ndcg import NDCG
from .metric import Metric

# This is a hack to make the NDCG metric return values in the range [-1, 1] instead of [0, 1]
class NDCGNorm(Metric):
    def __init__(self, k):
        self.name = "ndcg@{}".format(k)
        self.regular_ndcg = NDCG(k)
        self.k = k
        
    def __call__(self, recommendations, actual_actions):
        return (self.regular_ndcg(recommendations, actual_actions) - 0.5) * 2  


