import math
import random

import numpy as np
from scipy.sparse import csr_matrix
from tensorflow.python.keras.utils.data_utils import Sequence


class HistoryBatchGenerator(Sequence):
    def __init__(self, user_actions, history_size, n_items, batch_size=1000, n_target_actions=5):
        self.user_actions = user_actions
        self.history_size= history_size
        self.n_items = n_items
        self.n_target_actions = n_target_actions
        self.batch_size = batch_size
        self.features_matrix = None
        self.target_matrix = None
        self.reset()


    def reset(self):
        history, target = self.split_actions(self.user_actions)
        self.features_matrix = self.matrix_for_embedding(history, self.history_size, self.n_items)
        self.target_matrix = self.one_hot_encoded_matrix(target, self.n_items)
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
                vals.append(1.0)
        result =  csr_matrix((vals, (rows, cols)), shape=(len(user_actions), n_items))
        return result


    def split_actions(self, user_actions):
        history = []
        target = []
        for user in user_actions:
            user_history, user_target = self.split_user(user)
            history.append(user_history)
            target.append(user_target)
        return history, target

    def split_user(self, user):
        history_fraction = random.random()
        n_history_actions = int(len(user) * history_fraction)
        target_actions = user[n_history_actions:n_history_actions + self.n_target_actions]
        return user[:n_history_actions], target_actions

    def __len__(self):
        return math.floor(self.features_matrix.shape[0] / self.batch_size)

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
