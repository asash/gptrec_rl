from collections import Counter
from aprec.api.user import User
from aprec.recommenders.recommender import Recommender


class ConditionalTopRecommender(Recommender):
    """
    This recommender calculates top items based on some condition. For example, we want to recommend
    the most popular hotel in the city, not globally (for global top we can use @TopRecommender).
    """
    def __init__(self, conditional_field: str):
        self.conditional_field: str = conditional_field
        self.items_counts: dict = dict()
        self.precalculated_top_items: dict = dict()
        self.user_field_values: dict = dict()
    
    def add_user(self, user: User):
        if self.conditional_field in user.cat_features:
            self.user_field_values[user.user_id] = user.cat_features[self.conditional_field]
        

    def add_action(self, action):
        if action.user_id not in self.user_field_values and self.conditional_field not in action.data:
            raise Exception(f"this actions does not have required field: {self.conditional_field}")

        field_value: str = action.data[self.conditional_field]
        if field_value not in self.items_counts:
            self.items_counts[field_value] = Counter()
        self.user_field_values[action.user_id] = field_value

        if action.item_id is not None:
            self.items_counts[field_value][action.item_id] += 1

    def rebuild_model(self):
        self.precalculated_top_items = {
            field_value: counter.most_common() for field_value, counter in self.items_counts.items()
        }

    def recommend(self, user_id, limit, features=None):
        if user_id not in self.user_field_values:
            raise Exception("New user without field value")
        return self.precalculated_top_items[self.user_field_values[user_id]][:limit]

    def get_similar_items(self, item_id, limit):
        raise NotImplementedError

    def name(self):
        return "ConditionalTopItemsRecommender"
