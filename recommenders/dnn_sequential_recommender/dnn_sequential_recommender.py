import gc
import time

import tensorflow.keras.backend as K
from collections import defaultdict

from aprec.utils.item_id import ItemId
from aprec.recommenders.metrics.ndcg import KerasNDCG
from aprec.recommenders.recommender import Recommender
from aprec.recommenders.dnn_sequential_recommender.data_generator import DataGenerator
from aprec.recommenders.dnn_sequential_recommender.data_generator import actions_to_vector
from aprec.recommenders.dnn_sequential_recommender.models.sequential_recsys_model import SequentialRecsysModel
from aprec.losses.loss import Loss
from aprec.losses.bce import BCELoss

import numpy as np


class DNNSequentialRecommender(Recommender):
    def __init__(self, model_arch: SequentialRecsysModel, loss: Loss = BCELoss(), users_featurizer=None,
                 train_epochs=300, optimizer='adam', batch_size=1000, early_stop_epochs=100, target_decay=1.0,
                 train_on_last_item_only=False, training_time_limit=None, sigma=1, eval_ndcg_at=40):
        super().__init__()
        self.model_arch = model_arch
        self.users = ItemId()
        self.items = ItemId()
        self.user_actions = defaultdict(lambda: [])
        self.model = None
        self.user_vectors = None
        self.matrix = None
        self.mean_user = None
        self.train_epochs = train_epochs
        self.loss = loss
        self.early_stop_epochs = early_stop_epochs
        self.optimizer = optimizer
        self.metadata = {}
        self.batch_size = batch_size
        self.sigma = sigma
        self.target_decay = target_decay
        self.val_users = None
        self.eval_ndcg_at = 40
        self.eval_ndcg_at = eval_ndcg_at
        self.train_on_last_item_only = train_on_last_item_only
        self.training_time_limit = training_time_limit
        self.users_featurizer = users_featurizer
        self.user_features = {}
        self.users_with_actions = set()
        self.max_user_features = 0
        self.max_user_feature_val = 0

    def add_user(self, user):
        if self.users_featurizer is None:
            pass
        else:
            self.user_features[self.users.get_id(user.user_id)] = self.users_featurizer(user)

    def get_metadata(self):
        return self.metadata

    def set_loss(self, loss):
        self.loss = loss

    def name(self):
        return self.model

    def add_action(self, action):
        user_id_internal = self.users.get_id(action.user_id)
        action_id_internal = self.items.get_id(action.item_id)
        self.users_with_actions.add(user_id_internal)
        self.user_actions[user_id_internal].append((action.timestamp, action_id_internal))

    # exclude last action for val_users
    def user_actions_by_id_list(self, id_list, val_user_ids=None):
        val_users = set()
        if val_user_ids is not None:
            val_users = set(val_user_ids)
        result = []
        for user_id in id_list:
            if user_id not in val_users:
                result.append(self.user_actions[user_id])
            else:
                result.append(self.user_actions[user_id][:-1])
        return result

    def sort_actions(self):
        for user_id in self.user_actions:
            self.user_actions[user_id].sort()

    def rebuild_model(self):
        self.sort_actions()
        self.loss.set_num_items(self.items.size())
        self.loss.set_batch_size(self.batch_size)
        self.max_user_features, self.max_user_feature_val = self.get_max_user_features()

        train_users, train_user_ids, train_features, val_users, val_user_ids, val_features = self.train_val_split()

        print("train_users: {}, val_users:{}, items:{}".format(len(train_users), len(val_users), self.items.size()))
        val_generator = DataGenerator(val_users, val_user_ids, val_features, self.model_arch.max_history_length,
                                      self.items.size(),
                                      batch_size=self.batch_size, last_item_only=True,
                                      target_decay=self.target_decay,
                                      user_id_required=self.model_arch.requires_user_id,
                                      max_user_features=self.max_user_features,
                                      user_features_required=not (self.users_featurizer is None)
                                      )
        self.model = self.get_model()
        best_ndcg = 0
        steps_since_improved = 0
        best_epoch = -1
        best_weights = self.model.get_weights()
        val_ndcg_history = []
        start_time = time.time()
        for epoch in range(self.train_epochs):
            val_generator.reset()
            generator = DataGenerator(train_users, train_user_ids, train_features, self.model_arch.max_history_length,
                                      self.items.size(),
                                      batch_size=self.batch_size, target_decay=self.target_decay,
                                      user_id_required=self.model_arch.requires_user_id,
                                      last_item_only=self.train_on_last_item_only,
                                      max_user_features=self.max_user_features,
                                      user_features_required=not (self.users_featurizer is None)
                                      )
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
            print(f"val_ndcg: {val_ndcg}, best_ndcg: {best_ndcg}, steps_since_improved: {steps_since_improved},"
                  f" total_training_time: {total_trainig_time}")
            if steps_since_improved >= self.early_stop_epochs:
                print(f"early stopped at epoch {epoch}")
                break

            if self.training_time_limit is not None and total_trainig_time > self.training_time_limit:
                print(f"time limit stop triggered at epoch {epoch}")
                break

            K.clear_session()
            gc.collect()
        self.model.set_weights(best_weights)
        self.metadata = {"epochs_trained": best_epoch + 1, "best_val_ndcg": best_ndcg,
                         "val_ndcg_history": val_ndcg_history}
        print(self.get_metadata())
        print(f"taken best model from epoch{best_epoch}. best_val_ndcg: {best_ndcg}")

    def train_val_split(self):
        all_user_ids = self.users_with_actions
        val_user_ids = [self.users.get_id(val_user) for val_user in self.val_users]
        train_user_ids = list(all_user_ids)
        val_users = self.user_actions_by_id_list(val_user_ids)
        train_users = self.user_actions_by_id_list(train_user_ids, val_user_ids)
        train_features = [self.user_features.get(id, list()) for id in train_user_ids]
        val_features = [self.user_features.get(id, list()) for id in val_user_ids]
        return train_users, train_user_ids, train_features, val_users, val_user_ids, val_features

    def get_model(self):
        self.max_user_features, self.max_user_feature_val = self.get_max_user_features()
        self.model_arch.set_common_params(num_items=self.items.size(),
                                          num_users=self.users.size(),
                                          max_user_features=self.max_user_features,
                                          user_feature_max_val=self.max_user_feature_val)
        model = self.model_arch.get_model()
        ndcg_metric = KerasNDCG(self.eval_ndcg_at)
        metrics = [ndcg_metric]
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=metrics)
        return model

    def recommend(self, user_id, limit, features=None):
        scores = self.get_all_item_scores(user_id)
        best_ids = np.argsort(scores)[::-1][:limit]
        result = [(self.items.reverse_id(id), scores[id]) for id in best_ids]
        return result

    def get_item_rankings(self):
        result = {}
        for request in self.items_ranking_requests:
            user_id = request.user_id
            scores = self.get_all_item_scores(user_id)
            user_result = []
            for item_id in request.item_ids:
                user_result.append((item_id, scores[self.items.get_id(item_id)]))
            user_result.sort(key = lambda x: -x[1])
            result[user_id] = user_result
        return result

    def get_all_item_scores(self, user_id):
        actions = self.user_actions[self.users.get_id(user_id)]
        items_list = [action[1] for action in actions]
        model_actions = [(0, action) for action in items_list]
        session = actions_to_vector(model_actions, self.model_arch.max_history_length, self.items.size())
        session = session.reshape(1, self.model_arch.max_history_length)
        model_inputs = [session]
        if (self.model_arch.requires_user_id):
            model_inputs.append(np.array([[self.users.get_id(user_id)]]))

        if self.users_featurizer is not None:
            user_features = self.user_features.get(self.users.get_id(user_id), list())
            features_vector = DataGenerator.get_features_matrix([user_features], self.max_user_features)
            model_inputs.append(features_vector)

        scores = self.model.predict(model_inputs)[0]
        return scores

    def get_max_user_features(self):
        result = 0
        max_val = 0
        for user_id in self.user_features:
            features = self.user_features[user_id]
            result = max(result, len(features))
            for feature in features:
                max_val = max(feature, max_val)
        return result, max_val
