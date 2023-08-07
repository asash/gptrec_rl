import numpy as np
from aprec.recommenders.recommender import Recommender
from collections import Counter

from aprec.utils.item_id import ItemId

#Recommends from most popular items, but stochatsically according to their popularity
class StochasticTopRecommender(Recommender):
    def __init__(self, recency=1.0): #recency parameter controls how many actions are considered out of all actions
        super().__init__()
        self.items_counter=Counter()
        self.item_scores = {}
        self.actions = []
        self.recency = recency
        self.items = ItemId()

    def add_action(self, action):
        self.actions.append(action)

    def rebuild_model(self):
        self.actions.sort(key=lambda x: x.timestamp)
        n_actions = int(len(self.actions) * self.recency)
        action_counter = 0
        for action in self.actions[-n_actions:]:
            self.items_counter[action.item_id] += 1
            action_counter += 1
            self.items.get_id(action.item_id)
        self.actions = []
        self.item_probs = np.zeros(self.items.size())
        for item, score in self.items_counter.items():
            self.item_probs[self.items.get_id(item)] = score / action_counter
        pass
       
    def recommend(self, user_id, limit, features=None):
        internal_item_ids = np.random.choice(self.items.size(), limit, p=self.item_probs, replace=False)
        recs = []
        for internal_id in internal_item_ids:
            external_id = self.items.reverse_id(internal_id)
            recs.append((external_id, self.item_probs[internal_id]))
        recs = sorted(recs, key=lambda x: -x[1])
        return recs 

    def get_metadata(self):
        return {"top 20 items":  self.most_common[:20]}


    def get_similar_items(self, item_id, limit):
        return self.most_common[:limit]
    
    def recommend_by_items(self, items_list, limit: int):
        return self.recommend(None, limit)

    def name(self):
        return "TopItemsRecommender"

    def get_item_rankings(self):
        result = {}
        for request in self.items_ranking_requests:
            request_result = []
            for item_id in request.item_ids:
                score = self.item_scores.get(item_id, 0)
                request_result.append((item_id, score))
            request_result.sort(key=lambda x: -x[1])
            result[request.user_id] = request_result
        return result



