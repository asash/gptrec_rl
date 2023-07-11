from collections import defaultdict

import numpy as np
from aprec.recommenders.sequential.models.generative.reward_metrics.reward_metric import RewardMetric

#computes cosine similarity between items in the recommendation list based on their categories.
class ILDReward(RewardMetric):
    #categories dict has a set of categories for each item
    def __init__(self, categories_dict):
        self.name = "ILD"
        categoriy_ids = defaultdict(lambda: len(categoriy_ids))
        self.categories = {}
        
        for item in categories_dict:
            for category in categories_dict[item]:
                categoriy_ids[category]

        for item in categories_dict:
            item_categories = np.zeros(len(categoriy_ids))
            for category in categories_dict[item]:
                item_categories[categoriy_ids[category]] = 1
            self.categories[item] = item_categories
        
    
    #for each item in recommendations, compute average cosine similarity between the item and other items in the recommendation list
    def __call__(self, recommendations, actual) -> np.ndarray:
        rewards = []
        for i in range(len(recommendations)):
            reward = 0
            item1 = recommendations[i][0]
            vec1 = self.categories[item1]
            for j in range(0, i):
                if i == j:
                    continue
                item2 = recommendations[j][0]
                vec2 = self.categories[item2]
                cos_sim = 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                reward += cos_sim
            reward /= ((len(recommendations) - 1)*len(recommendations))
            rewards.append(reward)
        result = np.array(rewards, dtype=np.float32)
        return result