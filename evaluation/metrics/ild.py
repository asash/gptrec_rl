import numpy as np
from aprec.evaluation.metrics.metric import Metric
from aprec.recommenders.sequential.models.generative.reward_metrics.ild_reward import ILDReward


class ILD(Metric):
    def __init__(self, categories_dict):
        self.name = "ILD"
        self.less_is_better = False
        self.ild_reward_metric = ILDReward(categories_dict)

    def __call__(self, recommendations, actual):
        ild_reward = self.ild_reward_metric(recommendations, actual)
        return float(np.sum(ild_reward))

        