import gc
import time

import keras
import tensorflow.keras.backend as K
from collections import defaultdict

from aprec.recommenders.losses.bpr import BPRLoss
from aprec.utils.item_id import ItemId
from aprec.recommenders.metrics.ndcg import KerasNDCG
from aprec.recommenders.losses.lambdarank import LambdaRankLoss
from aprec.recommenders.losses.xendcg import XENDCGLoss
from aprec.recommenders.recommender import Recommender
from aprec.recommenders.salrec.data_generator import DataGenerator,  reverse_positions
from aprec \
    .recommenders.history_batch_generator import actions_to_vector
import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np


class SalrecRecommender(Recommender):
    def __init__(self, train_epochs=300,
                 max_history_len=1000,
                 loss='binary_crossentropy',
                 output_layer_activation='sigmoid',
                 optimizer='adam',
                 batch_size=64,
                 early_stop_epochs=100,
                 target_decay = 0.8,
                 sigma=1,
                 ndcg_at=30,
                 num_blocks=5,
                 num_heads = 5,
                 num_target_predictions=5
                 ):
        self.users = ItemId()
        self.items = ItemId()
        self.user_actions = defaultdict(lambda: [])
        self.model = None
        self.user_vectors = None
        self.matrix = None
        self.mean_user = None
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
        self.num_blocks = num_blocks
        self.val_users = None
        self.num_heads = num_heads
        self.num_target_predictions = num_target_predictions
        self.target_request = np.array([[self.max_history_length + 1] * num_target_predictions])

    def get_metadata(self):
        return self.metadata

    def set_loss(self, loss):
        self.loss = loss

    def name(self):
        return "SalRec"

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
        val_generator = DataGenerator(val_users, self.max_history_length, self.items.size(),
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
            generator = DataGenerator(train_users, self.max_history_length, self.items.size(),
                                              batch_size=self.batch_size, target_decay=self.target_decay)
            print(f"epoch: {epoch}")
            train_history = self.model.fit(generator, validation_data=val_generator)
            total_trainig_time = time.time() - start_time
            val_ndcg = train_history.history[f"val_ndcg_at_{self.ndcg_at}"][-1]
            val_ndcg_history.append((total_trainig_time, val_ndcg))
            steps_since_improved += 1
            if val_ndcg > best_ndcg:
                steps_since_improved = 0
                best_ndcg = val_ndcg
                best_epoch = epoch
                best_weights = self.model.get_weights()
            print(f"val_ndcg: {val_ndcg}, best_ndcg: {best_ndcg}, steps_since_improved: "
                  f"{steps_since_improved}, total_training_time: {total_trainig_time}")
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

    def get_model(self, n_items):
        embedding_size = 64

        pos_input = layers.Input(shape=(self.max_history_length))
        position_embedding_layer = layers.Embedding(self.max_history_length +
                                              self.num_target_predictions + 1, embedding_size )

        position_embedding = position_embedding_layer(pos_input)
        position_embedding = layers.Dense(embedding_size, activation='swish')(position_embedding)


        input = layers.Input(shape=(self.max_history_length))
        x = layers.Embedding(n_items + 1, embedding_size)(input)
        x = layers.Multiply()([x, position_embedding])

        for block_num in range(self.num_blocks):
             x = self.block(x)


        target_pos_input = layers.Input(shape=(self.num_target_predictions))
        target_pos_embedding = position_embedding_layer(target_pos_input)
        target_pos_embedding = layers.Dense(embedding_size, activation='swish')(target_pos_embedding)
        x = layers.MultiHeadAttention(self.num_heads, key_dim=x.shape[-1])(target_pos_embedding, x)
        x = layers.LayerNormalization()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(256, name="bottleneck", activation='swish')(x)
        x = layers.Dropout(0.5, name="dropout")(x)
        x = layers.Dense(256, name="bottleneck_after_dropout", activation='swish')(x)
        output = layers.Dense(n_items, name="output", activation=self.output_layer_activation)(x)
        model = keras.Model(inputs = [input, pos_input, target_pos_input], outputs=output)
        # model = keras.Model(inputs = [input], outputs=output)
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

        pos = np.array(reverse_positions(len(items_list), self.max_history_length)) \
            .reshape(1, self.max_history_length)

        scores = self.model.predict([vector.reshape(1, self.max_history_length), pos, self.target_request])[0]
        best_ids = np.argsort(scores)[::-1][:limit]
        result = [(self.items.reverse_id(id), scores[id]) for id in best_ids]
        return result

    def recommend_by_items(self, items_list, limit):
        items_iternal = []
        for item in items_list:
            item_id = self.items.get_id(item)
            items_iternal.append(item_id)
        return self.get_model_predictions(items_iternal, limit)

    def block(self, x):
        shortcut = x
        attention = layers.MultiHeadAttention(self.num_heads, key_dim=x.shape[-1])(x, x)
        attention = layers.Convolution1D(x.shape[-1], 1, activation='swish')(attention)
        output = layers.Multiply()([shortcut, attention])
        output = layers.LayerNormalization()(output)
        return output

    def get_similar_items(self, item_id, limit):
        raise (NotImplementedError)

    def to_str(self):
        raise (NotImplementedError)

    def from_str(self):
        raise (NotImplementedError)

    def train_val_split(self):
        all_user_ids = set(range(0, self.users.size()))
        val_user_ids = [self.users.get_id(val_user) for val_user in self.val_users]
        train_user_ids = list(all_user_ids - set(val_user_ids))
        val_users = self.user_actions_by_id_list(val_user_ids)
        train_users = self.user_actions_by_id_list(train_user_ids)
        return train_users, val_users
