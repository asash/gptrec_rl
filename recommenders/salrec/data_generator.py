import math
import random
import sys

import numpy as np
from scipy.sparse import csr_matrix
from tensorflow.python.keras.utils.data_utils import Sequence


class DataGenerator(Sequence):
    def __init__(self, user_actions, history_size, n_items, positional=True, batch_size=1000, validation=False, target_decay=0.8,
                min_target_val=0.1, num_actions_to_predict=5):
        self.user_actions = user_actions
        self.history_size= history_size
        self.n_items = n_items
        self.batch_size = batch_size
        self.features_matrix = None
        self.target_matrix = None
        self.validation = validation
        self.target_decay = target_decay
        self.min_target_val = min_target_val
        self.num_actions_to_predict = num_actions_to_predict
        self.positional = positional
        self.reset()


    def reset(self):
        random.shuffle(self.user_actions)
        history, history_positions, target, target_positions = self.split_actions(self.user_actions)
        self.features_matrix = self.matrix_for_embedding(history, self.history_size, self.n_items)
        self.history_positions = np.array(history_positions)
        self.target_positions = np.array(target_positions)
        self.target_matrix = self.get_target_matrix(target, self.n_items)
        self.current_position = 0
        self.max = self.__len__()


    @staticmethod
    def matrix_for_embedding(user_actions, history_size, n_items):
        result = []
        for actions in user_actions:
            result.append(actions_to_vector(actions, history_size, n_items))
        return np.array(result)

    def get_target_matrix(self, user_actions, n_items):
        rows = []
        cols = []
        vals = []
        for i in range(len(user_actions)):
            cur_val = 0.99
            for action_num in range(len(user_actions[i])):
                action = user_actions[i][action_num]
                rows.append(i)
                cols.append(action[1])
                vals.append(cur_val)
                cur_val *= self.target_decay
                if cur_val < self.min_target_val:
                    cur_val = self.min_target_val
        result =  csr_matrix((vals, (rows, cols)), shape=(len(user_actions), n_items))
        return result


    def split_actions(self, user_actions):
        history = []
        history_positions = []
        target = []
        target_positions = []
        for user in user_actions:
            user_history, user_history_positions, user_target, user_target_positions = self.split_user(user)
            history.append(user_history)
            history_positions.append(user_history_positions)
            target.append(user_target)
            target_positions.append(user_target_positions)
        return history, history_positions, target, target_positions

    def split_user(self, user_raw):
        user = user_raw[-self.history_size:]
        if not self.validation:
            history_fraction = random.random()
            n_history_actions = int(len(user) * history_fraction)
            actions_to_predict = np.random.choice(range(len(user)), self.num_actions_to_predict, replace=True)
        else:
            n_history_actions = len(user) - 1
            actions_to_predict = [len(user) - 1] * self.num_actions_to_predict

        history_actions = []
        history_positions = []


        for i in range(len(user)):
            if i in actions_to_predict:
                continue
            history_actions.append(user[i])
            history_positions.append(self.get_position_id(n_history_actions, i))

        target_actions = []
        target_positions = []
        for i in range(len(actions_to_predict)):
            target_actions.append(user[actions_to_predict[i]])
            target_positions.append(self.get_position_id(n_history_actions, actions_to_predict[i]))

        if len(history_positions) < self.history_size:
            history_positions = [0] * (self.history_size - len(history_positions)) + history_positions

        history_actions = history_actions[-self.history_size:]
        history_positions = history_positions[-self.history_size:]

        return history_actions, history_positions, target_actions, target_positions

    def get_position_id(self,n_history_actions, i):
        if i < n_history_actions:
            result = n_history_actions - i

        else:
            result = self.history_size + (i - n_history_actions + 1)
        return result

    def positions(self, sessions, history_size):
        result_reverse =[]
        for session in sessions:
            result_reverse.append(reverse_positions(len(session), history_size))
        return  np.array(result_reverse)

    def __len__(self):
        return self.features_matrix.shape[0] // self.batch_size

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        history = self.features_matrix[start:end]
        model_inputs = [history]
        target = self.target_matrix[start:end].todense()

        if self.positional:
            positions = self.history_positions[start:end]
            target_positions = self.target_positions[start:end]
            model_inputs.append(positions)
            model_inputs.append(target_positions)
        return model_inputs, target

    def __next__(self):
        if self.current_position >= self.max:
            raise StopIteration()
        result = self.__getitem__(self.current_position)
        self.current_position += 1
        return result

def reverse_positions(session_len, history_size):
    if session_len >= history_size:
        return list(range(history_size, 0, -1))
    else:
        return [0] * (history_size - session_len) + list(range(session_len, 0, -1))


def actions_to_vector(user_actions, vector_size, special_value):
    if len(user_actions) >= vector_size:
        return np.array([action[1] for action in user_actions[-vector_size:]])
    else:
        n_special = vector_size - len(user_actions)
        result_list = [special_value] * n_special + [action[1] for action in user_actions]
        return np.array(result_list)
