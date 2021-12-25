import gc
import time

import keras
import mmh3
import tensorflow.keras.backend as K
from collections import defaultdict

from keras.regularizers import l2

from aprec.losses.lambda_gamma_rank import LambdarankLambdasSum, BCELambdasSum, LambdaGammaRankLoss
from aprec.losses.loss import Loss
from aprec.utils.item_id import ItemId
from aprec.recommenders.metrics.ndcg import KerasNDCG
from aprec.recommenders.recommender import Recommender
from aprec.recommenders.salrec.data_generator import DataGenerator,  reverse_positions
from aprec.recommenders.dnn_sequential_recommender.data_generator import actions_to_vector
import tensorflow.keras.layers as layers
import numpy as np


class SalrecRecommender(Recommender):
    def __init__(self, train_epochs=300, max_history_len=200,
                                  loss=LambdaGammaRankLoss(pred_truncate_at=2500, bce_grad_weight=0.975),
                 output_layer_activation='linear', optimizer='adam', batch_size=128, early_stop_epochs=100,
                 target_decay=1.0, num_blocks=3, num_heads=5, num_target_predictions=5,
                 positional=True, embedding_size=64, bottleneck_size=256, num_bottlenecks=2, regularization=0.0,
                 training_time_limit=None,  log_lambdas_len=False,
                 eval_ndcg_at=10,
                 num_user_cat_hashes=3, user_cat_features_space=1000, max_user_features_hashes=None):
        super().__init__()
        self.embedding_size = embedding_size
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
        self.output_layer_activation = output_layer_activation
        self.target_decay = target_decay
        self.num_blocks = num_blocks
        self.val_users = None
        self.num_heads = num_heads
        self.num_target_predictions = num_target_predictions
        self.target_request = np.array([[self.max_history_length + 1] * num_target_predictions])
        self.positional = positional
        self.regularization = regularization
        self.bottleneck_size = bottleneck_size
        self.num_bottlenecks = num_bottlenecks
        self.trainig_time_limit = training_time_limit
        self.num_user_cat_hashes = num_user_cat_hashes
        self.user_cat_features_space=user_cat_features_space
        self.user_feature_hashes = {}
        self.max_user_feature_hashes = max_user_features_hashes
        self.eval_ndcg_at=eval_ndcg_at
        assert(isinstance(self.loss, Loss))

        if log_lambdas_len and not (isinstance(loss, LambdaGammaRankLoss)):
            raise Exception("logging lambdas len is only possible with lambdarank loss")
        self.log_lambdas_len = log_lambdas_len

    def add_user(self, user):
        user_id = self.users.get_id(user.user_id)
        feature_hashes = self.get_feature_hashes(user.cat_features)
        self.user_feature_hashes[user_id] = feature_hashes

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
        self.loss.set_num_items(self.items.size())
        self.loss.set_batch_size(self.batch_size)
        if len(self.user_feature_hashes) == 0:
            self.max_user_feature_hashes = 0
        if self.max_user_feature_hashes is None:
            self.max_user_feature_hashes = int(np.max([len(self.user_feature_hashes[user])
                                                   for user in self.user_feature_hashes]))

        train_users, train_user_features,  val_users, val_user_features = self.train_val_split()
        print("train_users: {}, val_users:{}, items:{}".format(len(train_users), len(val_users), self.items.size()))
        val_generator = DataGenerator(val_users, self.max_history_length, self.items.size(),
                                      val_user_features,
                                      self.max_user_feature_hashes,
                                      batch_size=self.batch_size, validation=True,
                                      target_decay=self.target_decay,
                                      num_actions_to_predict=self.num_target_predictions,
                                      positional=self.positional)
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
                                              val_user_features,
                                              self.max_user_feature_hashes,
                                              batch_size=self.batch_size, target_decay=self.target_decay,
                                              num_actions_to_predict=self.num_target_predictions,
                                              positional=self.positional)
            print(f"epoch: {epoch}")
            X, y = generator[0]
            train_history = self.model.fit(generator, validation_data=val_generator)
            total_trainig_time = time.time() - start_time
            val_ndcg = train_history.history[f"val_ndcg_at_{self.eval_ndcg_at}"][-1]
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

            if self.trainig_time_limit is not None and total_trainig_time > self.trainig_time_limit:
                print(f"time limit stop triggered at epoch {epoch}")
                break

            K.clear_session()
            gc.collect()

        self.model.set_weights(best_weights)
        self.metadata = {"epochs_trained": best_epoch + 1, "best_val_ndcg": best_ndcg,
                         "val_ndcg_history": val_ndcg_history}
        print(self.get_metadata())
        print(f"taken best model from epoch{best_epoch}. best_val_ndcg: {best_ndcg}")

    def get_model(self, n_items):
        embedding_size = self.embedding_size
        model_inputs = []

        input = layers.Input(shape=(self.max_history_length))
        model_inputs.append(input)
        x = layers.Embedding(n_items + 1, embedding_size)(input)

        if self.max_user_feature_hashes > 0:
            features_input = layers.Input(shape=(self.max_user_feature_hashes))
            model_inputs.append(features_input)
            user_features_embedding = layers.Embedding(self.user_cat_features_space + 1, embedding_size)
            user_features = user_features_embedding(features_input)
            user_features = layers.AveragePooling1D(self.num_user_cat_hashes)(user_features)

        if self.positional:
            pos_input = layers.Input(shape=(self.max_history_length))
            model_inputs.append(pos_input)
            position_embedding_layer = layers.Embedding(2*self.max_history_length + 1, embedding_size )

            position_embedding = position_embedding_layer(pos_input)
            position_embedding = layers.Dense(embedding_size, activation='swish')(position_embedding)
            x = layers.Multiply()([x, position_embedding])

            target_pos_input = layers.Input(shape=(self.num_target_predictions))
            model_inputs.append(target_pos_input)
            target_pos_embedding = position_embedding_layer(target_pos_input)
            target_pos_embedding = layers.Dense(embedding_size, activation='swish')(target_pos_embedding)

        if self.max_user_feature_hashes > 0:
            x = layers.concatenate([x, user_features], axis=1)

        for block_num in range(self.num_blocks):
             x = self.block(x)

        if self.positional:
            x = layers.MultiHeadAttention(self.num_heads, key_dim=x.shape[-1])(target_pos_embedding, x)
            x = layers.LayerNormalization()(x)

        x = layers.Flatten()(x)
        for i in range(self.num_bottlenecks):
            x = layers.Dense(self.bottleneck_size, activation='swish')(x)
        output = layers.Dense(n_items, name="output", activation=self.output_layer_activation, bias_regularizer=l2(self.regularization),
                              kernel_regularizer=l2(self.regularization))(x)
        model = keras.Model(inputs = model_inputs, outputs=output)
        ndcg_metric = KerasNDCG(self.eval_ndcg_at)
        metrics = [ndcg_metric]
        if self.log_lambdas_len:
            metrics.append(LambdarankLambdasSum(self.loss))
            metrics.append(BCELambdasSum(self.loss))


        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=metrics)
        return model

    def recommend(self, user_id, limit, features=None):
        items, user_features = self.items_and_features_for_user(user_id)
        return self.get_model_predictions(items, user_features, limit)

    def get_item_rankings(self):
        result = {}
        for request in self.items_ranking_requests:
            user_id = request.user_id
            items, user_features = self.items_and_features_for_user(user_id)
            scores = self.get_all_scores(items, user_features)
            user_result = []
            for item_id in request.item_ids:
                if self.items.has_item(item_id):
                    user_result.append((item_id, scores[self.items.get_id(item_id)]))
                else:
                    user_result.append((item_id, float("-inf")))
            user_result.sort(key = lambda x: -x[1])
            result[user_id] = user_result
        return result


    def items_and_features_for_user(self, user_id):
        if user_id in self.user_actions:
            actions = self.user_actions[self.users.get_id(user_id)]
        else:
            actions = []
        items = [action[1] for action in actions]
        if self.max_user_feature_hashes > 0:
            user_features = [self.user_feature_hashes.get(self.users.get_id(user_id))]
        else:
            user_features = [[]]
        return items, user_features

    def get_model_predictions(self, items_list, user_features, limit):
        scores = self.get_all_scores(items_list, user_features)
        best_ids = np.argsort(scores)[::-1][:limit]
        result = [(self.items.reverse_id(id), scores[id]) for id in best_ids]
        return result

    def get_all_scores(self, items_list, user_features):
        actions = [(0, action) for action in items_list]
        history = actions_to_vector(actions, self.max_history_length, self.items.size()) \
            .reshape(1, self.max_history_length)
        model_inputs = [history]
        if self.max_user_feature_hashes > 0:
            features = DataGenerator.get_user_features_matrix(user_features, self.max_user_feature_hashes)
            model_inputs.append(features)
        if self.positional:
            pos = np.array(reverse_positions(len(items_list), self.max_history_length)) \
                .reshape(1, self.max_history_length)
            model_inputs.append(pos)
            model_inputs.append(self.target_request)
        scores = self.model.predict(model_inputs)[0]
        return scores

    def recommend_by_items(self, items_list, limit):
        items_iternal = []
        for item in items_list:
            item_id = self.items.get_id(item)
            items_iternal.append(item_id)
        user_features = [[]]
        return self.get_model_predictions(items_iternal, user_features, limit)

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
        all_user_ids = set(self.user_actions.keys())
        val_user_ids = []
        for user in self.val_users:
            val_user_id = self.users.get_id(user)
            val_user_ids.append(val_user_id)


        train_user_ids = list(all_user_ids - set(val_user_ids))


        train_users = self.user_actions_by_id_list(train_user_ids)
        train_user_features = self.user_features_by_id_list(train_user_ids)

        val_users = self.user_actions_by_id_list(val_user_ids)
        val_user_features = self.user_features_by_id_list(val_user_ids)

        return train_users, train_user_features, val_users, val_user_features

    def user_features_by_id_list(self, id_list):
        result = []
        for user_id in id_list:
            result.append(self.user_feature_hashes.get(user_id, ()))
        return result



    def get_feature_hashes(self, cat_features):
        result = []
        for feature in cat_features:
            for hash_num in range(self.num_user_cat_hashes):
                val = f"{feature}_" + str(cat_features[feature]) + f"_hash{hash_num}"
                hash_val = mmh3.hash(val) % self.user_cat_features_space + 1
                result.append(hash_val)
        return result
