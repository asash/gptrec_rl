import gc
import tensorflow.keras.backend as K
import random
from collections import defaultdict

from aprec.recommenders.booking_recommender.candidates_recommender import BookingCandidatesRecommender
from aprec.utils.item_id import ItemId
from aprec.recommenders.metrics.ndcg import KerasNDCG
from aprec.recommenders.metrics.success import KerasSuccess
from aprec.losses.lambdarank import LambdaRankLoss
from aprec.losses.xendcg import XENDCGLoss
from aprec.recommenders.recommender import Recommender
from aprec.recommenders.booking_recommender.booking_history_batch_generator import BookingHistoryBatchGenerator, \
    ACTION_FEATURES, encode_action_features, CandidatesGenerator, direct_positions, reverse_positions
from aprec.recommenders.booking_recommender.booking_history_batch_generator import history_to_vector,\
    encode_additional_features, id_vector
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers
import numpy as np


NUM_AFFILIATES = 3612
NUM_CITIES = 39903
NUM_COUNTRIES = 197


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
                 target_decay=0.6,
                 min_target_value=0.0,
                 candidates_cnt = 50, epoch_size=20000, val_epoch_size=2000):
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
        self.target_decay = target_decay
        self.min_target_value = min_target_value
        self.output_layer_activation = output_layer_activation
        self.candidates_cnt = candidates_cnt
        self.city_country_mapping = {}
        self.candidates_recommender = BookingCandidatesRecommender()
        self.epoch_size = epoch_size
        self.val_epoch_size = val_epoch_size

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
        hotel_country_id = self.countries.get_id(action.data['hotel_country'])
        self.city_country_mapping[item_id_internal] = hotel_country_id
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

        print("rebuilding candidates generator...")
        for trip in train_users:
            for(_, action) in trip:
                self.candidates_recommender.add_action(action)

        self.candidates_recommender.rebuild_model()


        val_generator = BookingHistoryBatchGenerator(val_users, self.max_history_length, self.items.size(),
                                              batch_size=self.batch_size, country_dict = self.countries,
                                              candidates_recommender=self.candidates_recommender,
                                              city_dict=self.items,
                                              affiliates_dict = self.affiliates,
                                              validation=True,
                                              city_country_mapping = self.city_country_mapping,
                                              target_decay=self.target_decay,
                                              n_candidates=self.candidates_cnt,
                                              min_target_val=self.min_target_value, epoch_size=self.val_epoch_size)
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
                                              affiliates_dict = self.affiliates,
                                              candidates_recommender=self.candidates_recommender,
                                              city_dict=self.items,
                                              city_country_mapping=self.city_country_mapping,
                                              n_candidates=self.candidates_cnt,
                                              target_decay=self.target_decay,
                                              min_target_val=self.min_target_value, epoch_size=self.epoch_size)
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
        print(f"taken best model from epoch{best_epoch}. best_val_success: {best_success}")

    def block(self, x, num_heads=5):
        shortcut = x
        attention = layers.MultiHeadAttention(num_heads, key_dim=x.shape[-1])(x, x)
        attention = layers.Convolution1D(x.shape[-1], 1, activation='swish')(attention)
        output = layers.Multiply()([shortcut, attention])
        output = layers.LayerNormalization()(output)
        return output

    def get_model(self):
        direct_pos_input = layers.Input(shape=(self.max_history_length))
        direct_pos_embedding = layers.Embedding(self.max_history_length +1, 10)(direct_pos_input)

        reverse_pos_input = layers.Input(shape=(self.max_history_length))
        reverse_pos_embedding = layers.Embedding(self.max_history_length +1, 10)(reverse_pos_input)


        city_embedding = layers.Embedding(NUM_CITIES + 1, 32)
        country_embedding = layers.Embedding(NUM_COUNTRIES + 1, 32)
        target_features = layers.Input(shape=len(ACTION_FEATURES))
        target_features_encoded = layers.Dense(50, activation='swish')(target_features)
        target_features_encoded = layers.BatchNormalization()(target_features_encoded)
        target_features_encoded = layers.Dense(50, activation='swish')(target_features_encoded)
        target_features_encoded = layers.BatchNormalization()(target_features_encoded)
        affiliate_id_embedding = layers.Embedding(NUM_AFFILIATES + 1, 5)
        target_affiliate_id_input = layers.Input(shape=(1, 1))
        target_affiliate_id_embedding = affiliate_id_embedding(target_affiliate_id_input)
        target_affiliate_id_embedding = layers.Flatten()(target_affiliate_id_embedding)
        target_affiliate_id_embedding = layers.BatchNormalization()(target_affiliate_id_embedding)
        target_features_encoded = layers.Concatenate()([target_features_encoded, target_affiliate_id_embedding])

        history_input = layers.Input(shape=(self.max_history_length))
        features_input = layers.Input(shape=(self.max_history_length, len(ACTION_FEATURES)))


        user_country_input = layers.Input(shape=(self.max_history_length))
        hotel_country_input = layers.Input(shape=(self.max_history_length))
        affiliate_id_input = layers.Input(shape=(self.max_history_length))
        history_embedding = city_embedding(history_input)
        user_country_embedding = country_embedding(user_country_input)
        hotel_country_embedding = country_embedding(hotel_country_input)
        history_affiliates_embeddings = affiliate_id_embedding(affiliate_id_input)
        concatenated = layers.Concatenate()([history_embedding,
                                             features_input,
                                             user_country_embedding,
                                             hotel_country_embedding, history_affiliates_embeddings])



        position_embedding = layers.Concatenate()([direct_pos_embedding, reverse_pos_embedding])
        position_embedding = layers.Convolution1D(concatenated.shape[-1], 1)(position_embedding)

        x = layers.BatchNormalization()(concatenated)
        x = layers.Convolution1D(x.shape[-1], 1)(x)
        x = layers.Multiply()([x, position_embedding])
        x = self.block(x)
        # x = layers.Flatten()(x)
        # x = layers.Dense(self.bottleneck_size,
        #                        name="bottleneck", activation="swish")(x)
        # x = layers.Concatenate()([x, target_features_encoded])
        # x = layers.Dense(1000, name="dense4", activation="sigmoid")(x)
        # x = layers.Dropout(0.5, name="dropout")(x)

        candiate_city_input = layers.Input(shape=(self.candidates_cnt))
        target_city_emb = city_embedding(candiate_city_input)
        candidate_country_input = layers.Input(shape=(self.candidates_cnt))
        target_country_emb = country_embedding(candidate_country_input)
        candidate_features_input = layers.Input(shape=(self.candidates_cnt, self.candidates_recommender.n_features))

        target_embedding = layers.Concatenate()([target_city_emb, target_country_emb, candidate_features_input])
        target_embedding = layers.Convolution1D(x.shape[-1], 1, activation='swish')(target_embedding)
        target_embedding = self.block(target_embedding)

        target_attention = layers.MultiHeadAttention(num_heads=5, key_dim=x.shape[-1])(target_embedding, x)
        target_attention = layers.Multiply()([target_attention, target_embedding])
        target_attention = layers.Convolution1D(x.shape[-1], 1, activation='sigmoid')(target_attention)

        target_features_encoded =  layers.Dense(x.shape[-1], activation='swish')(target_features_encoded)
        target_features_encoded =  layers.Dense(x.shape[-1], activation='tanh')(target_features_encoded)
        target_features_encoded = layers.Dropout(0.5)(target_features_encoded)
        output = layers.Dot(axes=-1)([target_attention, target_features_encoded])

        model = Model(inputs=[history_input, direct_pos_input, reverse_pos_input, features_input, user_country_input,
                              hotel_country_input, affiliate_id_input, target_features, target_affiliate_id_input,
                              candiate_city_input, candidate_country_input, candidate_features_input], outputs=output)

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
        return LambdaRankLoss(self.candidates_cnt, self.batch_size, self.sigma, ndcg_at=self.ndcg_at)

    def get_xendcg_loss(self):
        return XENDCGLoss(self.items.size(), self.batch_size)

    def get_next_items(self, user_id, limit, features=None):
        actions = self.user_actions[self.users.get_id(user_id)]
        items = [action[1] for action in actions]
        return self.get_model_predictions(items, limit, target_action=features)

    def get_model_predictions(self, items_list, limit, target_action):
        actions = [(self.items.get_id(action.item_id), action) for action in items_list]
        history_vector = history_to_vector(actions, self.max_history_length, self.items.size())\
            .reshape(1, self.max_history_length)

        direct_pos = np.array(direct_positions(len(items_list), self.max_history_length))\
            .reshape(1, self.max_history_length)
        reverse_pos = np.array(reverse_positions(len(items_list), self.max_history_length)) \
            .reshape(1, self.max_history_length)


        additional_features = encode_additional_features(actions, self.max_history_length)\
            .reshape(1, self.max_history_length, len(ACTION_FEATURES))

        target_affiliate_id = np.array(self.affiliates.get_id(target_action.data['affiliate_id'])).reshape(1,1)

        target_features = encode_action_features(target_action).reshape(1, len(ACTION_FEATURES))

        user_countries = id_vector(actions, self.max_history_length, self.countries, 'booker_country') \
            .reshape(1, self.max_history_length)
        hotel_countries = id_vector(actions, self.max_history_length, self.countries, 'hotel_country') \
            .reshape(1, self.max_history_length)
        affiliate_ids = id_vector(actions, self.max_history_length, self.affiliates, 'affiliate_id') \
            .reshape(1, self.max_history_length)

        candidates_generator = CandidatesGenerator(self.items,
                                                   self.candidates_recommender,
                                                   self.candidates_cnt,
                                                   self.city_country_mapping)
        candidates, countries, candidate_features = candidates_generator(items_list)

        candiates = np.array(candidates).reshape(1, self.candidates_cnt)
        countries = np.array(countries).reshape(1, self.candidates_cnt)
        candidate_features = np.array(candidate_features)\
            .reshape(1, self.candidates_cnt, self.candidates_recommender.n_features)
        scores = self.model.predict([history_vector, direct_pos, reverse_pos, additional_features, user_countries,
                                     hotel_countries, affiliate_ids, target_features, target_affiliate_id,
                                     candiates, countries, candidate_features])[0]
        result = []
        for i in range(len(candiates[0])):
            item = self.items.reverse_id(candiates[0][i])
            score = scores[i]
            result.append((item, score))
        result = sorted(result, key=lambda x: -x[1])
        return result[:limit]

