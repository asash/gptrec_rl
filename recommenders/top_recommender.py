from aprec.recommenders.recommender import Recommender
from collections import Counter

class TopRecommender(Recommender):
    def __init__(self):
       self.items_counter=Counter()         

    def add_action(self, action):
        self.items_counter[action.item_id] += 1

    def rebuild_model(self):
        pass

    def get_next_items(self, user_id, limit):
        return self.items_counter.most_common(limit)

    def get_similar_items(self, item_id, limit):
        return self.items_counter.most_common(limit)

