from aprec.recommenders.recommender import Recommender
from collections import Counter

class TopRecommender(Recommender):
    def __init__(self):
       self.items_counter=Counter()         

    def add_action(self, action):
        self.items_counter[action.item_id] += 1

    def rebuild_model(self):
        self.most_common = self.items_counter.most_common()

    def get_next_items(self, user_id, limit, features=None):
        return self.most_common[:limit]

    def get_similar_items(self, item_id, limit):
        return self.most_common[:limit]

    def name(self):
        return "TopItemsRecommender"
