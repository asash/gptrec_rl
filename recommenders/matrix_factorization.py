import random
from collections import defaultdict

import numpy as np
from keras.layers import Flatten
from keras.utils.data_utils import Sequence
from scipy.sparse import csr_matrix

from aprec.recommenders.losses.bpr import BPRLoss
from aprec.recommenders.losses.lambdarank import LambdaRankLoss
from aprec.recommenders.losses.xendcg import XENDCGLoss
from aprec.recommenders.recommender import Recommender
from aprec.utils.item_id import ItemId
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


class MatrixFactorizationRecommender(Recommender):
    def __init__(self, embedding_size, num_epochs, loss, batch_size):
        self.users = ItemId()
        self.items = ItemId()
        self.user_actions = defaultdict(list)
        self.embedding_size = embedding_size
        self.num_epochs = num_epochs
        self.loss = loss
        self.batch_size = batch_size
        self.sigma = 1.0
        self.ndcg_at=40

    def add_action(self, action):
        self.user_actions[self.users.get_id(action.user_id)].append(self.items.get_id(action.item_id))


    def rebuild_model(self):
        loss = self.loss

        if loss == 'lambdarank':
            loss = self.get_lambdarank_loss()

        if loss == 'xendcg':
            loss = self.get_xendcg_loss()

        if loss == 'bpr':
            loss = self.get_bpr_loss()

        self.model = Sequential()
        self.model.add(Embedding(self.users.size(), self.embedding_size+1, input_length=1))
        self.model.add(Flatten())
        self.model.add(Dense(self.items.size()))
        self.model.compile(optimizer=Adam(), loss=loss)
        data_generator = DataGenerator(self.user_actions, self.users.size(), self.items.size(), self.batch_size)
        for epoch in range(self.num_epochs):
            print(f"epoch: {epoch}")
            data_generator.shuffle()
            self.model.fit(data_generator)

    def get_next_items(self, user_id, limit, features=None):
        model_input = np.array([[self.users.get_id(user_id)]])
        predictions = tf.nn.top_k(self.model.predict(model_input), limit)
        result = []
        for item_id, score in zip(predictions.indices[0], predictions.values[0]):
            result.append((self.items.reverse_id(int(item_id)), float(score)))
        return result

    def get_lambdarank_loss(self):
        return LambdaRankLoss(self.items.size(), self.batch_size)

    def get_xendcg_loss(self):
        return XENDCGLoss(self.items.size(), self.batch_size)

    def get_bpr_loss(self):
        return BPRLoss(max_positives=40)

class DataGenerator(Sequence):
    def __init__(self, user_actions, n_users, n_items, batch_size):
        rows = []
        cols = []
        vals = []

        for user in user_actions:
            for item in user_actions[user]:
                rows.append(user)
                cols.append(item)
                vals.append(1.0)
        self.full_matrix = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
        self.users = list(range(n_users))
        self.batch_size = batch_size

    def shuffle(self):
        random.shuffle(self.users)

    def __len__(self):
        return len(self.users) // self.batch_size

    def __getitem__(self, item):
        start = self.batch_size * item
        end = self.batch_size * (item+1)
        users = []
        targets = []
        for i in range(start, end):
            users.append([self.users[i]])
            targets.append(self.full_matrix[self.users[i]].todense())
        users = np.array(users)
        targets = np.reshape(np.array(targets), (self.batch_size, self.full_matrix.shape[1]))
        return users, targets