import numpy as np
from aprec.evaluation.metrics.metric import Metric
from aprec.recommenders.sequential.models.generative.reward_metrics.pcount_reward import PCountReward


class PCOUNT(Metric):
    def __init__(self, k, actions):
        self.name = f"PCOUNT@{k}"
        self.less_is_better = True
        self.pcount_reward_metric = PCountReward(k, actions)

    def __call__(self, recommendations, actual):
        reward = self.pcount_reward_metric(recommendations, actual)
        return -float(np.sum(reward))

        