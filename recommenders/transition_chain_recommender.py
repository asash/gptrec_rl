import numpy as np

from aprec.recommenders.recommender import Recommender


class TransitionsChainRecommender(Recommender):
    """
    TODO
    """
    def __init__(self, chain_indicator_field: str):
        self.item_id_to_index: dict = dict()
        self.index_to_item_id: dict = dict()
        self.items_count: int = 0
        self.transition_matrix: np.array = np.array([])

        self.user_field_values: dict = dict()
        self.chain_to_items: dict = dict()
        self.chain_indicator_field: str = chain_indicator_field

    def add_action(self, action):
        if self.chain_indicator_field not in action.data:
            raise Exception(f"this actions does not have required field: {self.chain_indicator_field}. Contains: {action.data}")

        if action.item_id not in self.item_id_to_index:
            self.item_id_to_index[action.item_id] = self.items_count
            self.index_to_item_id[self.items_count] = action.item_id
            self.items_count += 1

        if action.data[self.chain_indicator_field] not in self.chain_to_items:
            self.chain_to_items[action.data[self.chain_indicator_field]] = [action.item_id]
        else:
            self.chain_to_items[action.data[self.chain_indicator_field]].append(action.item_id)

        if action.user_id not in self.user_field_values:
            self.user_field_values[action.user_id] = [action.data[self.chain_indicator_field]]
        else:
            if action.data[self.chain_indicator_field] not in self.user_field_values[action.user_id]:
                self.user_field_values[action.user_id].append(action.data[self.chain_indicator_field])

    def rebuild_model(self):
        self.transition_matrix = np.zeros(shape=(self.items_count, self.items_count))
        # считать попарно, а не только в таргет?
        for _, items in self.chain_to_items.items():
            target_item = items[-1]
            for item_id in items[:-1]:
                self.transition_matrix[self.item_id_to_index[item_id]][self.item_id_to_index[target_item]] += 1

    def get_next_items(self, user_id, limit):
        if user_id not in self.user_field_values:
            raise Exception("New user without field value")
        last_chain_id = self.user_field_values[user_id][-1]
        last_chain_items_indexes = [self.index_to_item_id[idx] for idx in self.chain_to_items[last_chain_id]]
        total_predictions = [
            (index, prob) for index, prob in zip(
                range(self.items_count), np.sum(self.transition_matrix[last_chain_items_indexes], axis=0)
            )
        ]
        total_predictions.sort(key=lambda x: -x[1])
        return [(self.index_to_item_id[idx], prob) for idx, prob in total_predictions[:limit]]

    def get_similar_items(self, item_id, limit):
        raise NotImplementedError

    def name(self):
        return "TransitionsChainRecommender"
