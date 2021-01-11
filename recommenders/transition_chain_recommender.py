import numpy as np

from aprec.recommenders.recommender import Recommender


class TransitionsChainRecommender(Recommender):
    """
    This recommender was written for situation when we have sequence of events for each user and we want
    to predict the last element of this sequence.
    During the training phase we calculate counts between events in a sequence and last event (target).
    During the inference phase we take all events of user and sum up all counts to predict next event.
    """
    def __init__(self):
        self.item_id_to_index: dict = dict()
        self.index_to_item_id: dict = dict()
        self.items_count: int = 0
        self.transition_matrix: np.array = np.array([])

        self.user_to_items: dict = dict()

    def add_action(self, action):
        if action.item_id not in self.item_id_to_index:
            self.item_id_to_index[action.item_id] = self.items_count
            self.index_to_item_id[self.items_count] = action.item_id
            self.items_count += 1

        if action.user_id not in self.user_to_items:
            self.user_to_items[action.user_id] = [action.item_id]
        else:
            self.user_to_items[action.user_id].append(action.item_id)

    def rebuild_model(self):
        self.transition_matrix = np.zeros(shape=(self.items_count, self.items_count))
        for _, items in self.user_to_items.items():
            target_item = items[-1]
            for item_id in items[:-1]:
                self.transition_matrix[self.item_id_to_index[item_id]][self.item_id_to_index[target_item]] += 1

    def get_next_items(self, user_id, limit):
        if user_id not in self.user_to_items:
            raise Exception("New user without history")
        last_items_indexes = [self.item_id_to_index[idx] for idx in self.user_to_items[user_id]]
        total_predictions = [
            (index, prob) for index, prob in zip(
                range(self.items_count), np.sum(self.transition_matrix[last_items_indexes], axis=0)
            )
        ]
        total_predictions.sort(key=lambda x: -x[1])
        return [(self.index_to_item_id[idx], prob) for idx, prob in total_predictions[:limit]]

    def get_similar_items(self, item_id, limit):
        raise NotImplementedError

    def name(self):
        return "TransitionsChainRecommender"
