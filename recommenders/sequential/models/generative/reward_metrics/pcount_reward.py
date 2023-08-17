from collections import Counter

import numpy as np
from .reward_metric import RewardMetric

class PCountReward(RewardMetric):
    def __init__(self, k, actions):
        self.name = f"pcount@{k}"
        self.k = k
        actions_counter = Counter() 
        for action in actions:
            actions_counter[action.item_id] += 1
        self.probs = {}
        self.k = k

        for item_id in actions_counter:
            self.probs[item_id] = actions_counter[item_id]/len(actions)
        pass
        
    def __call__(self, recommendations, actual_actions):
        result = []
        for item, score in recommendations[:self.k]:
            result.append(-self.probs[item])
        return result + [0.0] * (len(recommendations) - self.k)
    
         
