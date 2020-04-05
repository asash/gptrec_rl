from .recommender import Recommender
from collections import defaultdict

class FilterSeenRecommender(Recommender):
    def __init__(self, recommender):
        self.recommender = recommender
        self.user_seen = defaultdict(lambda: set()) 

    def name(self):
        return self.recommender.name() + "FilterSeen"

    def add_action(self, action):
        self.user_seen[action.user_id].add(action.item_id) 
        self.recommender.add_action(action)

    def rebuild_model(self):
        self.recommender.rebuild_model()

    def get_next_items(self, user_id, limit):
        user_seen_cnt = len(self.user_seen[user_id])
        raw = self.recommender.get_next_items(user_id, limit + user_seen_cnt)
        filtered = filter(lambda item_score: item_score[0] not in self.user_seen[user_id], raw)
        return list(filtered)[:limit]

    def get_similar_items(self, item_id, limit):
        raise(NotImplementedError)

    def recommend_by_items(self, items_list, limit):
        raw = self.recommender.recommend_by_items(items_list, limit + len(items_list))
        filtered = filter(lambda item_score: item_score[0] not in items_list, raw)
        return list(filtered)[:limit]

    def to_str(self):
        raise(NotImplementedError)

    def from_str(self):
        raise(NotImplementedError)
