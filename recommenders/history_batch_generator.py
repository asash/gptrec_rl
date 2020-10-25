import math
import random

import numpy as np
from scipy.sparse import csr_matrix
from tensorflow.python.keras.utils.data_utils import Sequence


class HistoryBatchGenerator(Sequence):
    def __init__(self, user_actions, history_size, n_items, batch_size=1000, item_bias={}):
        history, target = HistoryBatchGenerator.split_actions(user_actions)
        self.item_bias = item_bias
        self.features_matrix = self.matrix_for_embedding(history, history_size, n_items)
        self.target_matrix = self.one_hot_encoded_matrix(target, n_items)
        self.batch_size = batch_size
        self.current_position = 0
        self.max = self.__len__()

    @staticmethod
    def matrix_for_embedding(user_actions, history_size, n_items):
        result = []
        for actions in user_actions:
            result.append(actions_to_vector(actions, history_size, n_items))
        return np.array(result)

    def one_hot_encoded_matrix(self, user_actions, n_items):
        rows = []
        cols = []
        vals = []
        for i in range(len(user_actions)):
            for action_num in range(len(user_actions[i])):
                action = user_actions[i][action_num]
                rows.append(i)
                cols.append(action[1])
                #vals.append(1.0)
                vals.append(1.0 - self.item_bias.get(action[1], 0.0))
        return csr_matrix((vals, (rows, cols)), shape=(len(user_actions), n_items))

    @staticmethod
    def split_actions(user_actions):
        history = []
        target = []
        for user in user_actions:
            user_history, user_target = HistoryBatchGenerator.split_user(user)
            history.append(user_history)
            target.append(user_target)
        return history, target

    @staticmethod
    def split_user(user):
        history_fraction = random.random()
        n_history_actions = int(len(user) * history_fraction)
        target_actions = user[n_history_actions:]
        return user[:n_history_actions], target_actions

    def __len__(self):
        return math.ceil(self.features_matrix.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        history = self.features_matrix[idx * self.batch_size:(idx + 1) * self.batch_size]
        target = self.target_matrix[idx * self.batch_size:(idx + 1) * self.batch_size].todense()
        return history, target

    def __next__(self):
        if self.current_position >= self.max:
            raise StopIteration()
        result = self.__getitem__(self.current_position)
        self.current_position += 1
        return result


def actions_to_vector(user_actions, vector_size, special_value):
    if len(user_actions) >= vector_size:
        return np.array([action[1] for action in user_actions[-vector_size:]])
    else:
        n_special = vector_size - len(user_actions)
        result_list = [special_value] * n_special + [action[1] for action in user_actions]
        return np.array(result_list)
