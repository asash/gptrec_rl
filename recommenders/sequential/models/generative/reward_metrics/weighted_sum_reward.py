from typing import List

import numpy as np

from aprec.recommenders.sequential.models.generative.reward_metrics.reward_metric import RewardMetric


class WeightedSumReward(RewardMetric):
    def __init__(self, metrics: List[RewardMetric], weights: List[float]):
        self.weights = weights
        self.metrics = metrics
        self.name = self.get_name()

    def __call__(self, recommendations, actual) -> np.ndarray:
        reward_sum = np.zeros(len(recommendations), dtype=np.float32)
        for i in range(len(self.metrics)):
            reward_sum += self.metrics[i](recommendations, actual) * self.weights[i]
        return reward_sum
        
    def get_name(self) -> str:
        return "WeightedSumReward(" + "_".join([metric.name + ": " + str(weight) for metric, weight in zip(self.metrics, self.weights)]) + ")"