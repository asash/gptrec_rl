from tqdm import tqdm


class Recommender():
    def name(self):
        raise NotImplementedError

    def add_action(self, action):
        raise(NotImplementedError)

    def rebuild_model(self):
        raise(NotImplementedError)

    def get_next_items(self, user_id, limit, features=None):
        raise(NotImplementedError)

    #recommendation request = tuple(user_id, features)
    def recommend_batch(self, recommendation_requests, limit):
        results = []
        for user_id, features in tqdm(recommendation_requests, ascii=True):
            results.append(self.get_next_items(user_id, limit, features))
        return results

    def recommend_by_items(self, items_list, limit):
        raise(NotImplementedError)

    def get_similar_items(self, item_id, limit):
        raise(NotImplementedError)

    def to_str(self):
        raise(NotImplementedError)

    def from_str(self):
        raise(NotImplementedError)

    def get_metadata(self):
        return {}

    def set_val_users(self, val_users):
        self.val_users = val_users
