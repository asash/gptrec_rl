import math

from aprec.evaluation.metrics.ndcg import NDCG
from .metric import Metric

# This is a hack to make the NDCG metric return values in the range [-1, 1] instead of [0, 1]
class NDCGNorm(Metric):
    def __init__(self, k):
        self.name = "ndcg@{}".format(k)
        self.regular_ndcg = NDCG(k)
        self.k = k
        self.eps = 0.01
        
    def __call__(self, recommendations, actual_actions):
        result = self.regular_ndcg(recommendations, actual_actions)
        if result == 0:
            result = -self.eps # Negative reward for PPO
        return result

