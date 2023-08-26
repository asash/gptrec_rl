from collections import defaultdict
from functools import lru_cache
from aprec.api.action import Action
from aprec.recommenders.recommender import Recommender
import gzip
import numpy as np
import dill 


class MMRRecommender(Recommender):
    def __init__(self, checkpoint, categories_dict, mmr_cutoff=1000, mmr_lambda=0.5):
        self.recommender = dill.load(gzip.open(checkpoint, 'rb'))
        self.mmr_cutoff = mmr_cutoff
        self.mmr_lambda = mmr_lambda
        self.out_dir = None
        categoriy_ids = defaultdict(lambda: len(categoriy_ids))
        self.categories = {}
        self.user_seen = defaultdict(set)
        
        for item in categories_dict:
            for category in categories_dict[item]:
                categoriy_ids[category]

        for item in categories_dict:
            item_categories = np.zeros(len(categoriy_ids))
            for category in categories_dict[item]:
                item_categories[categoriy_ids[category]] = 1
            self.categories[item] = item_categories
        self.items_set = set()

    
    @lru_cache(maxsize=20000000)
    def similarity(self, item1, item2):
        return np.dot(self.categories[item1], self.categories[item2]) / (np.linalg.norm(self.categories[item1]) * np.linalg.norm(self.categories[item2]))

    def max_similarity(self, item, items):
        max_sim = -1
        for item2 in items:
            max_sim = max(max_sim, self.similarity(item, item2))
        return max_sim
        
    
    def add_action(self, action: Action):
        self.items_set.add(action.item_id)
        self.user_seen[action.user_id].add(action.item_id)
    
    def rebuild_model(self):
        pass #we load pre-trained recommender
    
    def recommend(self, user_id, limit: int, features=None):
        max_items_to_recommend = self.mmr_cutoff
        if 'FilterSeenRecommender' in str(type(self.recommender)): 
            max_items_to_recommend = min(self.mmr_cutoff, len(self.items_set)-len(self.user_seen[user_id]))

        recommendations = self.recommender.recommend(user_id, max_items_to_recommend)
        if len(recommendations) < limit:
            raise Exception("Not enough recommendations for user " + str(user_id))

        result = [(recommendations[0][0], 1.0)]
        already_recommended = set([recommendations[0][0]])

        for i in range(1, limit):
            max_mmr = float("-inf")
            max_item = None
            for j in range(1, len(recommendations)):
                item, score = recommendations[j]
                if item in already_recommended:
                    continue
                max_sim = self.max_similarity(item, already_recommended)
                mmr = self.mmr_lambda * score - (1 - self.mmr_lambda) * max_sim
                if mmr > max_mmr:
                    max_mmr = mmr
                    max_item = item
            if max_item is None:
                raise Exception("Not enough recommendations for user " + str(user_id))
            already_recommended.add(max_item)
            result.append((max_item, 1.0/(i+1)))
        return result