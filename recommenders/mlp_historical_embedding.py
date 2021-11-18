import gc
import time

import tensorflow.keras.backend as K
from collections import defaultdict

from aprec.losses.bpr import BPRLoss
from aprec.utils.item_id import ItemId
from aprec.recommenders.metrics.ndcg import KerasNDCG
from aprec.losses.lambdarank import LambdaRankLoss
from aprec.losses.xendcg import XENDCGLoss
from aprec.recommenders.recommender import Recommender
from aprec.recommenders.history_batch_generator import HistoryBatchGenerator
from aprec \
    .recommenders.history_batch_generator import actions_to_vector
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
import numpy as np


class GreedyMLPHistoricalEmbedding(Recommender):
    def __init__(self, bottleneck_size=32, train_epochs=300,
                 max_history_len=1000,
                 loss='binary_crossentropy',
                 output_layer_activation='sigmoid',
                 optimizer='adam',
                 batch_size=1000,
                 early_stop_epochs=100,
                 target_decay = 1.0,
                 sigma=1,
                 ndcg_at=30,
                 ):
        self.users = ItemId()
        self.items = ItemId()
        self.user_actions = defaultdict(lambda: [])
        self.model = None
        self.user_vectors = None
        self.matrix = None
        self.mean_user = None
        self.bottleneck_size = bottleneck_size
        self.train_epochs = train_epochs
        self.max_history_length = max_history_len
        self.loss = loss
        self.early_stop_epochs = early_stop_epochs
        self.optimizer = optimizer
        self.metadata = {}
        self.batch_size = batch_size
        self.sigma = sigma
        self.ndcg_at = ndcg_at
        self.output_layer_activation = output_layer_activation
        self.target_decay = target_decay
        self.val_users = None

    def get_metadata(self):
        return self.metadata

    def set_loss(self, loss):
        self.loss = loss

    def name(self):
        return "GreedyMLPHistoricalEmbedding"

    def add_action(self, action):
        user_id_internal = self.users.get_id(action.user_id)
        action_id_internal = self.items.get_id(action.item_id)
        self.user_actions[user_id_internal].append((action.timestamp, action_id_internal))

    def user_actions_by_id_list(self, id_list):
        result = []
        for user_id in id_list:
            result.append(self.user_actions[user_id])
        return result

    def sort_actions(self):
        for user_id in self.user_actions:
            self.user_actions[user_id].sort()

    def rebuild_model(self):
        self.sort_actions()
        train_users, val_users = self.train_val_split()

        print("train_users: {}, val_users:{}, items:{}".format(len(train_users), len(val_users), self.items.size()))
        val_generator = HistoryBatchGenerator(val_users, self.max_history_length, self.items.size(),
                                              batch_size=self.batch_size, validation=True,
                                              target_decay=self.target_decay)
        self.model = self.get_model(self.items.size())
        best_ndcg = 0
        steps_since_improved = 0
        best_epoch = -1
        best_weights = self.model.get_weights()
        val_ndcg_history = []
        start_time = time.time()
        for epoch in range(self.train_epochs):
            val_generator.reset()
            generator = HistoryBatchGenerator(train_users, self.max_history_length, self.items.size(),
                                              batch_size=self.batch_size, target_decay=self.target_decay)
            print(f"epoch: {epoch}")
            train_history = self.model.fit(generator, validation_data=val_generator)
            total_trainig_time = time.time() - start_time
            try:
                val_ndcg = train_history.history[f"val_ndcg_at_{self.ndcg_at}"][-1]
                val_ndcg_history.append((total_trainig_time, val_ndcg))
            except:
                print(f"self.ndcgg_at: {self.ndcg_at}")
                val_ndcg = 0
            steps_since_improved += 1
            if val_ndcg > best_ndcg:
                steps_since_improved = 0
                best_ndcg = val_ndcg
                best_epoch = epoch
                best_weights = self.model.get_weights()
            print(f"val_ndcg: {val_ndcg}, best_ndcg: {best_ndcg}, steps_since_improved: {steps_since_improved},"
                  f" total_training_time: {total_trainig_time}")
            if steps_since_improved >= self.early_stop_epochs:
                print(f"early stopped at epoch {epoch}")
                break
            K.clear_session()
            gc.collect()
        self.model.set_weights(best_weights)
        self.metadata = {"epochs_trained": best_epoch + 1, "best_val_ndcg": best_ndcg,
                         "val_ndcg_history": val_ndcg_history}
        print(self.get_metadata())
        print(f"taken best model from epoch{best_epoch}. best_val_ndcg: {best_ndcg}")

    def train_val_split(self):
        all_user_ids = set(range(0, self.users.size()))
        val_user_ids = [self.users.get_id(val_user) for val_user in self.val_users]
        train_user_ids = list(all_user_ids - set(val_user_ids))
        val_users = self.user_actions_by_id_list(val_user_ids)
        train_users = self.user_actions_by_id_list(train_user_ids)
        return train_users, val_users

    def get_model(self, n_movies):
        model = Sequential(name='MLP')
        model.add(layers.Embedding(n_movies + 1, 32, input_length=self.max_history_length))
        model.add(layers.Flatten())
        model.add(layers.Dense(256, name="dense1", activation="relu"))
        model.add(layers.Dense(128, name="dense2", activation="relu"))
        model.add(layers.Dense(self.bottleneck_size,
                               name="bottleneck", activation="relu"))
        model.add(layers.Dropout(0.5, name="dropout"))
        model.add(layers.Dense(128, name="dense3", activation="relu"))
        model.add(layers.Dense(256, name="dense4", activation="relu"))
        model.add(layers.Dense(n_movies, name="output", activation=self.output_layer_activation))
        ndcg_metric = KerasNDCG(self.ndcg_at)
        loss = self.loss
        if loss == 'lambdarank':
            loss = self.get_lambdarank_loss()

        if loss == 'xendcg':
            loss = self.get_xendcg_loss()

        if loss == 'bpr':
            loss = self.get_bpr_loss()

        model.compile(optimizer=self.optimizer, loss=loss, metrics=[ndcg_metric])
        return model

    def get_lambdarank_loss(self):
        return LambdaRankLoss(self.items.size(), self.batch_size, self.sigma, ndcg_at=self.ndcg_at)

    def get_xendcg_loss(self):
        return XENDCGLoss(self.items.size(), self.batch_size)

    def get_bpr_loss(self):
        return BPRLoss(max_positives=10)

    def get_next_items(self, user_id, limit, features=None):
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


