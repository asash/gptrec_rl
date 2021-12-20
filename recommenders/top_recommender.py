from aprec.recommenders.recommender import Recommender
from collections import Counter

class TopRecommender(Recommender):
    def __init__(self):
        super().__init__()
        self.items_counter=Counter()
        self.item_scores = {}

    def add_action(self, action):
        self.items_counter[action.item_id] += 1

    def rebuild_model(self):
        self.most_common = self.items_counter.most_common()
        for item, score in self.most_common:
            self.item_scores[item] = score

    def recommend(self, user_id, limit, features=None):
        return self.most_common[:limit]


    def get_similar_items(self, item_id, limit):
        return self.most_common[:limit]

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



