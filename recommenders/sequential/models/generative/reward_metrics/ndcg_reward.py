import math

import numpy as np
from .reward_metric import RewardMetric

class NDCGReward(RewardMetric):
    def __init__(self, k):
        self.name = "ndcg_reward@{}".format(k)
        self.k = k
        
    def __call__(self, recommendations, actual_actions):
        if(len(recommendations) == 0):
            return np.zeros(len(recommendations), dtype=np.float32)
        actual_set = set([action.item_id for action in actual_actions])
        recommended = [recommendation[0] for recommendation in recommendations[:self.k]]
        cool = set(recommended).intersection(actual_set)
        if len(cool) == 0:
            return np.zeros(len(recommendations), dtype=np.float32)
        ideal_rec = sorted(recommended, key = lambda x: not(x in actual_set))
        idcg = np.sum(NDCGReward.dcg(ideal_rec, actual_set))
        dcg_array = NDCGReward.dcg(recommended, actual_set)
        result = dcg_array/idcg
        return result

    
         

    @staticmethod
    def dcg(id_list, relevant_id_set):
        result = [] 
        for idx in range(len(id_list)):
            i = idx + 1
            if (id_list[idx]) in relevant_id_set:
                result.append(1 / math.log2(i+1))
            else:
                result.append(0.0)
        return np.array(result, dtype=np.float32)





