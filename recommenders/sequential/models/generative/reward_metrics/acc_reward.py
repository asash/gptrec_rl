import math

import numpy as np
from .reward_metric import RewardMetric

class ACCReward(RewardMetric):
    def __init__(self, k):
        self.name = "ndcg_reward@{}".format(k)
        self.k = k
        
    def __call__(self, recommendations, actual_actions):
        if(len(recommendations) == 0):
            return np.zeros(len(recommendations), dtype=np.float32)
        actual_set = set([action.item_id for action in actual_actions])
        try:
            recommended = [recommendation[0] for recommendation in recommendations[:self.k]]
        except IndexError:
            pass
        target_found = False
        result = []
        for i in range(len(recommended)):
            if recommended[i] in actual_set:
                target_found = True
                result.append(5.0)
            else:
                if not target_found:
                    result.append(-1.0)
                else:
                    result.append(0.0)
        return np.array(result, dtype=np.float32)
             
         




