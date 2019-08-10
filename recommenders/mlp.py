import sys
from aprec.utils.item_id import ItemId
from collections import defaultdict
from random import random,randint
from keras.models import Sequential
from keras.layers import Embedding, Lambda, BatchNormalization, Dense, Dropout
from keras import backend as K
import keras
from keras.optimizers import Adam

import numpy as np

class GreedyMLP():
    def __init__(self, sequence_length=100):
        self.sequence_length = sequence_length 
        self.users = ItemId()
        self.items = ItemId()
        self.user_actions = defaultdict(lambda: [])
        self.total_actions = 0
        self.model = None

    def name(self):
        return "GreedyMLPRecommender"

    def add_action(self, action):
        user_id = self.users.get_id(action.user_id)
        item_id = self.items.get_id(action.item_id)
        self.user_actions[user_id].append(item_id)
        self.total_actions += 1

    def get_batch(self, n_samples, n_history, n_target):
        data = []
        target = []
        for i in range(n_samples):
            user_id = randint(0, self.users.size()) 
            all_actions = self.user_actions[user_id]
            split_point = randint(0, len(all_actions))
            history = self.get_history(user_id, n_history, split_point)
            future = np.zeros(self.items.size())
            for action in all_actions[split_point: split_point + n_target]:
                future[action] = 1.0
            data.append(history)
            target.append(future)
        return np.array(data), np.array(target)

    def batch_iterator(self, n_samples, n_history, n_target):
        while True:
            yield self.get_batch(n_samples, n_history, n_target)

    def get_history(self, user_id_internal, n_history, split_point): 
            all_actions = self.user_actions[user_id_internal]
            history = all_actions[split_point - n_history:split_point]
            if len(history) < n_history:
                history = [-1]*(n_history - len(history)) + history
            return np.array(history)

    def rebuild_model(self):
        model = Sequential()
        embedding_length = 30
        target_items = 50
        batch_size = 10000
        n_epochs = 10 

        model.add(Embedding(self.items.size(), embedding_length, input_length=self.sequence_length))
        model.add(Dropout(0.5))
        model.add(Lambda(lambda v: K.sum(v, axis=1)))
        model.add(BatchNormalization())
        model.add(Dense(100, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
        model.add(Dense(100, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
        model.add(Dropout(0.5))
        model.add(Dense(self.items.size(), activation='softmax', kernel_regularizer=keras.regularizers.l2(0.001)))
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        generator = self.batch_iterator(batch_size, self.sequence_length, target_items) 
        steps_per_epoch = self.total_actions // batch_size + 1 
        model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=n_epochs)
        self.model = model
       

    def get_next_items(self, user_id, limit):
        user_id_internal = self.users.get_id(user_id) 
        split_point = len(self.user_actions[user_id_internal])
        history = np.array(self.get_history(user_id_internal, self.sequence_length, split_point)).reshape(1, self.sequence_length)
        scores = np.array(self.model.predict(history)).reshape(self.items.size())
        internal_movie_ids = np.argsort(scores)[::-1][:limit]
        result = []
        for id in internal_movie_ids:
            result.append((self.items.reverse_id(id), scores[id]))
        return result

    def get_similar_items(self, item_id, limit):
        raise(NotImplementedError)

    def to_str(self):
        raise(NotImplementedError)

    def from_str(self):
        raise(NotImplementedError)
