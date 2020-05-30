import random
from collections import defaultdict

from aprec.utils.item_id import ItemId
from aprec.recommenders.recommender import Recommender
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
from tensorflow.keras.utils import Sequence
import numpy as np
import math


class GreedyMLPHistoricalEmbedding(Recommender):
    def __init__(self, bottleneck_size=32, train_epochs=300, n_val_users=1000, max_history_len=1000):
        self.users = ItemId()
        self.items = ItemId()
        self.user_actions = defaultdict(lambda: [])
        self.model = None
        self.user_vectors = None
        self.matrix = None
        self.mean_user = None
        self.bottleneck_size = bottleneck_size
        self.train_epochs = train_epochs
        self.n_val_users = n_val_users
        self.max_history_length = max_history_len

    def name(self):
        return "GreedyMLPHistorical"

    def add_action(self, action):
        user_id_internal = self.users.get_id(action.user_id)
        action_id_internal = self.items.get_id(action.item_id)
        self.user_actions[user_id_internal].append((action.timestamp, action_id_internal))

    def user_actions_by_id_list(self, id_list):
        result = []
        for user_id in id_list:
            result.append(self.user_actions[user_id])
        return result

    def split_users(self):
        all_user_ids = list(range(0, self.users.size()))
        random.shuffle(all_user_ids)
        val_users = self.user_actions_by_id_list(all_user_ids[:self.n_val_users])
        train_users = self.user_actions_by_id_list(all_user_ids[self.n_val_users:])
        return train_users, val_users

    def sort_actions(self):
        for user_id in self.user_actions:
            self.user_actions[user_id].sort()

    def rebuild_model(self):
        self.sort_actions()
        train_users, val_users = self.split_users()
        val_generator = BatchGenerator(val_users, self.max_history_length, self.items.size())
        self.model = self.get_model(self.items.size())
        for epoch in range(self.train_epochs):
            print(f"epoch: {epoch}")
            generator = BatchGenerator(train_users, self.max_history_length, self.items.size())
            self.model.fit(generator, validation_data=val_generator)

    def get_model(self, n_movies):
        model = Sequential(name='MLP')
        model.add(layers.Embedding(n_movies + 1, 32, input_length=self.max_history_length))
        model.add(layers.Flatten())
        model.add(layers.Dense(256, name="dense1", activation="relu"))
        model.add(layers.Dense(128, name="dense2", activation="relu"))
        model.add(layers.Dense(self.bottleneck_size,
                               name="bottleneck", activation="relu"))
        model.add(layers.Dense(128, name="dense3", activation="relu"))
        model.add(layers.Dense(256, name="dense4", activation="relu"))
        model.add(layers.Dropout(0.5, name="dropout"))
        model.add(layers.Dense(n_movies, name="output", activation="sigmoid"))
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def get_next_items(self, user_id, limit):
        actions = self.user_actions[self.users.get_id(user_id)]
        items = [action[1] for action in actions]
        return self.get_model_predictions(items, limit)

    def get_model_predictions(self, items_list, limit):
        actions = [(0, action) for action in items_list]
        vector = actions_to_vector(actions, self.max_history_length, self.items.size())
        scores = self.model.predict(vector.reshape(1, self.max_history_length))[0]
        best_ids = np.argsort(scores)[::-1][:limit]
        result = [(self.items.reverse_id(id), scores[id]) for id in best_ids]
        return result

    def recommend_by_items(self, items_list, limit):
        items_iternal = []
        for item in items_list:
            item_id = self.items.get_id(item)
            items_iternal.append(item_id)
        return self.get_model_predictions(items_iternal, limit)

    def get_similar_items(self, item_id, limit):
        raise (NotImplementedError)

    def to_str(self):
        raise (NotImplementedError)

    def from_str(self):
        raise (NotImplementedError)


def actions_to_vector(user_actions, vector_size, special_value):
    if len(user_actions) >= vector_size:
        return np.array([action[1] for action in user_actions[-vector_size:]])
    else:
        n_special = vector_size - len(user_actions)
        result_list = [special_value] * n_special + [action[1] for action in user_actions]
        return np.array(result_list)


class BatchGenerator(Sequence):
    def __init__(self, user_actions, history_size, n_items, batch_size=1000):
        history, target = BatchGenerator.split_actions(user_actions)
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


    @staticmethod
    def one_hot_encoded_matrix(user_actions, n_items):
        rows = []
        cols = []
        vals = []
        for i in range(len(user_actions)):
            for action in user_actions[i]:
                rows.append(i)
                cols.append(action[1])
                vals.append(1.0)
        return csr_matrix((vals, (rows, cols)), shape=(len(user_actions), n_items))

    @staticmethod
    def split_actions(user_actions):
        history = []
        target = []
        for user in user_actions:
            user_history, user_target = BatchGenerator.split_user(user)
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
