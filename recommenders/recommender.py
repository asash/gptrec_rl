class Recommender():
    def name(self):
        raise NotImplementedError

    def add_action(self, action):
        raise(NotImplementedError)

    def rebuild_model(self):
        raise(NotImplementedError)

    def get_next_items(self, user_id, limit):
        raise(NotImplementedError)

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
