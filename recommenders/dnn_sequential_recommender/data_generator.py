import random

import numpy as np
from scipy.sparse import csr_matrix
from tensorflow.python.keras.utils.data_utils import Sequence


class DataGenerator(Sequence):
    def __init__(self, user_actions, user_ids,  user_features, history_size,
                 n_items, batch_size=1000, last_item_only=False, target_decay=0.8,
                 min_target_val=0.1, return_drect_positions=False, return_reverse_positions=False,
                 user_id_required=False,
                 max_user_features=0,
                 user_features_required=False
                 ):
        self.user_ids = [[id] for id in user_ids]
        self.user_actions = user_actions
        self.history_size= history_size
        self.n_items = n_items
        self.batch_size = batch_size
        self.sequences_matrix = None
        self.target_matrix = None
        self.last_item_only = last_item_only
        self.target_decay = target_decay
        self.min_target_val = min_target_val
        self.return_direct_positions = return_drect_positions
        self.return_reverse_positions = return_reverse_positions
        self.user_id_required = user_id_required
        self.user_features = user_features
        self.max_user_features = max_user_features
        self.user_features_required = user_features_required
        self.reset()


    def reset(self):
        self.shuffle_data()
        history, target = self.split_actions(self.user_actions)
        self.sequences_matrix = self.matrix_for_embedding(history, self.history_size, self.n_items)
        if self.return_direct_positions or self.return_reverse_positions:
            self.direct_position, self.reverse_position = self.positions(history, self.history_size)

        if self.user_features_required:
            self.user_features_matrix = self.get_features_matrix(self.user_features, self.max_user_features)

        self.target_matrix = self.get_target_matrix(target, self.n_items)
        self.current_position = 0
        self.max = self.__len__()

    def shuffle_data(self):
        actions_with_ids_and_features = list(zip(self.user_actions, self.user_ids, self.user_features))
        random.shuffle(actions_with_ids_and_features)
        user_actions, user_ids, user_features = zip(*actions_with_ids_and_features)

        self.user_actions = user_actions
        self.user_ids = user_ids
        self.user_features = user_features

    @staticmethod
    def get_features_matrix(user_features, max_user_features):
        result = []
        for features in user_features:
            result.append([0] * (max_user_features - len(features)) + features)
        return np.array(result)


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
        result =  csr_matrix((vals, (rows, cols)), shape=(len(user_actions), n_items), dtype='float32')
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
        if not self.last_item_only:
            history_fraction = random.random()
            n_history_actions = int(len(user) * history_fraction)
        else:
            n_history_actions = len(user) - 1
        n_target_actions = len(user) - n_history_actions
        history_actions =user[:n_history_actions]
        target_actions = user[-n_target_actions:]
        return history_actions, target_actions

    def positions(self, sessions, history_size):
        result_direct, result_reverse = [], []
        for session in sessions:
            result_direct.append(direct_positions(len(session), history_size))
            result_reverse.append(reverse_positions(len(session), history_size))
        return np.array(result_direct), np.array(result_reverse)

    def __len__(self):
        return self.sequences_matrix.shape[0] // self.batch_size

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        history = self.sequences_matrix[start:end]
        model_inputs = [history]
        target = np.asarray(self.target_matrix[start:end].todense())
        if self.return_direct_positions:
            direct_pos = self.direct_position[start:end]
            model_inputs.append(direct_pos)
        if self.return_reverse_positions:
            reverse_pos = self.reverse_position[start:end]
            model_inputs.append(reverse_pos)

        if self.user_id_required:
            user_ids = np.array(self.user_ids[start:end])
            model_inputs.append(user_ids)

        if self.user_features_required:
            features = self.user_features_matrix[start:end]
            model_inputs.append(features)

        return model_inputs, target

    def __next__(self):
        if self.current_position >= self.max:
            raise StopIteration()
        result = self.__getitem__(self.current_position)
        self.current_position += 1
        return result

def direct_positions(session_len, history_size):
    if session_len >= history_size:
        return list(range(1, history_size + 1))
    else:
        return [0] * (history_size - session_len) + list(range(1, session_len + 1))


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
