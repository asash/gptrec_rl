class Recommender():
    def add_action(self, action):
        raise(NotImplementedError)

    def rebuild_model(self):
        raise(NotImplementedError)

    def get_next_items(user_id, limit):
        raise(NotImplementedError)

    def get_similar_items(item_id, limit):
        raise(NotImplementedError)



