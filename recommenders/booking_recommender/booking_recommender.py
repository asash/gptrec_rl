import gc
import tensorflow.keras.backend as K
import random
from collections import defaultdict

from aprec.utils.item_id import ItemId
from aprec.recommenders.metrics.ndcg import KerasNDCG
from aprec.recommenders.metrics.success import KerasSuccess
from aprec.recommenders.losses.lambdarank import LambdaRankLoss
from aprec.recommenders.losses.xendcg import XENDCGLoss
from aprec.recommenders.recommender import Recommender
from aprec.recommenders.booking_recommender.booking_history_batch_generator import BookingHistoryBatchGenerator,\
    ACTION_FEATURES
from aprec.recommenders.booking_recommender.booking_history_batch_generator import history_to_vector,\
    encode_additional_features, id_vector
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers
import numpy as np


class BookingRecommender(Recommender):
    def __init__(self, bottleneck_size=32, train_epochs=300, n_val_users=1000,
                 max_history_len=1000, 
                 loss = 'binary_crossentropy',
                 output_layer_activation = 'sigmoid',
                 optimizer = 'adam',
                 batch_size = 1000,
                 early_stop_epochs = 100,
                 sigma = 1,
                 ndcg_at = 30,
                 ):
        self.users = ItemId()
        self.items = ItemId()
        self.countries = ItemId()
        self.affiliates = ItemId()
        self.user_actions = defaultdict(lambda: [])
        self.model = None
        self.user_vectors = None
        self.matrix = None
        self.mean_user = None
        self.bottleneck_size = bottleneck_size
        self.train_epochs = train_epochs
        self.n_val_users = n_val_users
        self.max_history_length = max_history_len
        self.loss = loss
        self.early_stop_epochs = early_stop_epochs
        self.optimizer = optimizer
        self.metadata = {}
        self.batch_size = batch_size
        self.sigma = sigma
        self.ndcg_at = ndcg_at
        self.output_layer_activation = output_layer_activation

    def get_metadata(self):
        return self.metadata

    def set_loss(self, loss):
        self.loss = loss

    def name(self):
        return "BookingRecommender"

    def add_action(self, action):
        user_id_internal = self.users.get_id(action.user_id)
        item_id_internal = self.items.get_id(action.item_id)
        self.countries.get_id(action.data['booker_country'])
        self.countries.get_id(action.data['hotel_country'])
        self.affiliates.get_id(action.data['affiliate_id'])
        self.user_actions[user_id_internal].append((item_id_internal, action))

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
            self.user_actions[user_id].sort(key=lambda x: x[1].timestamp)

    def rebuild_model(self):
        self.sort_actions()
        train_users, val_users = self.split_users()
        print("train_users: {}, val_users:{}, items:{}".format(len(train_users), len(val_users), self.items.size()))
        val_generator = BookingHistoryBatchGenerator(val_users, self.max_history_length, self.items.size(),
                                              batch_size=self.batch_size, country_dict = self.countries,
                                                affiliates_dict = self.affiliates,
                                                     validation=True)
        self.model = self.get_model()
        best_success = 0
        steps_since_improved = 0
        best_epoch = -1 
        best_weights = self.model.get_weights()
        val_ndcg_history = []
        val_success_history = []
        for epoch in range(self.train_epochs):
            generator = BookingHistoryBatchGenerator(train_users, self.max_history_length, self.items.size(),
                                              batch_size=self.batch_size, country_dict = self.countries,
                                                     affiliates_dict = self.affiliates)
            print(f"epoch: {epoch}")
            train_history = self.model.fit(generator, validation_data=val_generator)
            val_ndcg = train_history.history[f"val_ndcg_at_{self.ndcg_at}"][-1]
            val_success = train_history.history[f"val_Success_at_4"][-1]
            val_ndcg_history.append(val_ndcg)
            val_success_history.append(val_success)

            steps_since_improved += 1
            if val_success > best_success:
                steps_since_improved = 0
                best_success = val_success
                best_epoch =  epoch
                best_weights = self.model.get_weights()
            print(f"best_val_success: {best_success}, steps_since_improved: {steps_since_improved}")
            if steps_since_improved >= self.early_stop_epochs:
                print(f"early stopped at epoch {epoch}")
                break
            K.clear_session()
            gc.collect()
        self.model.set_weights(best_weights)
        self.metadata = {"epochs_trained": best_epoch + 1, "best_val_ndcg": best_success, "val_ndcg_history":
            val_ndcg_history, "val_success_history": val_success_history}
        print(self.get_metadata())
        print(f"taken best model from epoch{best_epoch}. best_val_ndcg: {best_success}")

    def get_model(self):
        country_embedding = layers.Embedding(self.countries.size() + 1, 10)

        history_input = layers.Input(shape=(self.max_history_length))
        features_input = layers.Input(shape=(self.max_history_length, len(ACTION_FEATURES)))

        user_country_input = layers.Input(shape=(self.max_history_length))
        hotel_country_input = layers.Input(shape=(self.max_history_length))
        affiliate_id_input = layers.Input(shape=(self.max_history_length))

        history_embedding = layers.Embedding(self.items.size() + 1, 32)(history_input)

        user_country_embedding = country_embedding(user_country_input)
        hotel_country_embedding = country_embedding(hotel_country_input)
        affiliate_id_embedding = layers.Embedding(self.affiliates.size() + 1, 5)(affiliate_id_input)
        concatenated = layers.Concatenate()([history_embedding, features_input,
                                             user_country_embedding, hotel_country_embedding, affiliate_id_embedding])
        x = layers.BatchNormalization()(concatenated)
        x = layers.Attention()([x, x])
        x = layers.Attention()([x, x])
        x = layers.Attention()([x, x])
        x = layers.Flatten()(x)
        x = layers.Dense(self.bottleneck_size,
                               name="bottleneck", activation="swish")(x)
        x = layers.Dense(1000, name="dense4", activation="sigmoid")(x)
        x = layers.Dropout(0.5, name="dropout")(x)
        output = layers.Dense(self.items.size(), name="output", activation=self.output_layer_activation)(x)
        model = Model(inputs=[history_input, features_input, user_country_input,
                              hotel_country_input, affiliate_id_input], outputs=output)
        ndcg_metric = KerasNDCG(self.ndcg_at)
        success_4_metric = KerasSuccess(4)

        loss = self.loss
        if loss == 'lambdarank':
            loss = self.get_lambdarank_loss()

        if loss == 'xendcg':
            loss = self.get_xendcg_loss()

        model.compile(optimizer=self.optimizer, loss=loss, metrics=[ndcg_metric, success_4_metric])
        return model

    def get_lambdarank_loss(self):
        return LambdaRankLoss(self.items.size(), self.batch_size, self.sigma, ndcg_at=self.ndcg_at)

    def get_xendcg_loss(self):
        return XENDCGLoss(self.items.size(), self.batch_size)

    def get_next_items(self, user_id, limit):
        actions = self.user_actions[self.users.get_id(user_id)]
        items = [action[1] for action in actions]
        return self.get_model_predictions(items, limit)

    def get_model_predictions(self, items_list, limit):
        actions = [(self.items.get_id(action.item_id), action) for action in items_list]
        history_vector = history_to_vector(actions, self.max_history_length, self.items.size())\
            .reshape(1, self.max_history_length)
        additional_features = encode_additional_features(actions, self.max_history_length)\
            .reshape(1, self.max_history_length, len(ACTION_FEATURES))

        user_countries = id_vector(actions, self.max_history_length, self.countries, 'booker_country') \
            .reshape(1, self.max_history_length)
        hotel_countries = id_vector(actions, self.max_history_length, self.countries, 'hotel_country') \
            .reshape(1, self.max_history_length)
        affiliate_ids = id_vector(actions, self.max_history_length, self.affiliates, 'affiliate_id') \
            .reshape(1, self.max_history_length)
        scores = self.model.predict([history_vector, additional_features, user_countries,
                                     hotel_countries, affiliate_ids])[0]
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


