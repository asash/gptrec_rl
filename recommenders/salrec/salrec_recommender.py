import gc
import time

import mmh3
import tensorflow.keras.backend as K
from collections import defaultdict

from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from aprec.losses.lambda_gamma_rank import LambdarankLambdasSum, BCELambdasSum, LambdaGammaRankLoss
from aprec.losses.loss import Loss
from aprec.recommenders.salrec.salrec_model import SalrecModel
from aprec.utils.item_id import ItemId
from aprec.recommenders.metrics.ndcg import KerasNDCG
from aprec.recommenders.recommender import Recommender
from aprec.recommenders.salrec.data_generator import DataGenerator, reverse_positions, actions_to_vector
import numpy as np


class SalrecRecommender(Recommender):
    def __init__(self, train_epochs=300, max_history_len=200,
                                  loss=LambdaGammaRankLoss(pred_truncate_at=2500, bce_grad_weight=0.975),
                 output_layer_activation='linear', optimizer=Adam(), batch_size=128, early_stop_epochs=100,
                 target_decay=1.0, num_blocks=3, num_heads=5, num_target_predictions=5,
                 positional=True, embedding_size=64, bottleneck_size=256, num_bottlenecks=2, regularization=0.0,
                 training_time_limit=None,  log_lambdas_len=False,
                 eval_ndcg_at=10,
                 users_featurizer=None,
                 num_user_cat_hashes=3, user_cat_features_space=1000,
                 debug = False):
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
        self.users_featurizer = users_featurizer
        self.user_features = {}
        self.eval_ndcg_at=eval_ndcg_at
        self.debug = debug
        assert(isinstance(self.loss, Loss))

        self.users_with_actions = set()

        if log_lambdas_len and not (isinstance(loss, LambdaGammaRankLoss)):
            raise Exception("logging lambdas len is only possible with lambdarank loss")
        self.log_lambdas_len = log_lambdas_len

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
        return "SalRec"

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
        if len(self.user_features) == 0:
            self.max_user_feature_hashes = 0
        if self.max_user_feature_hashes is None:
            self.max_user_feature_hashes = int(np.max([len(self.user_features[user])
                                                   for user in self.user_features]))

        train_users, train_ids, train_user_features,  val_users, val_ids ,val_user_features = self.train_val_split()
        print("train_users: {}, val_users:{}, items:{}".format(len(train_users), len(val_users), self.items.size()))
        val_generator = DataGenerator(val_users, self.max_history_length, self.items.size(),
                                      val_user_features,
                                      self.max_user_feature_hashes,
                                      batch_size=self.batch_size, validation=True,
                                      target_decay=self.target_decay,
                                      num_actions_to_predict=self.num_target_predictions,
                                      positional=self.positional)
        self.model = SalrecModel(n_items=self.items.size(),
                                 num_heads=self.num_heads,
                                 num_blocks=self.num_blocks,
                                 embedding_size=self.embedding_size,
                                 num_bottlenecks=self.num_bottlenecks,
                                 max_user_feature_hashes=self.max_user_feature_hashes,
                                 output_layer_activation=self.output_layer_activation,
                                 user_cat_features_space=self.user_cat_features_space,
                                 max_history_length=self.max_history_length,
                                 num_user_cat_hashes=self.num_user_cat_hashes,
                                 bottleneck_size=self.bottleneck_size,
                                 regularization=self.regularization,
                                 positional=self.positional)

        ndcg_metric = KerasNDCG(self.eval_ndcg_at)
        metrics = [ndcg_metric]
        if self.log_lambdas_len:
            metrics.append(LambdarankLambdasSum(self.loss))
            metrics.append(BCELambdasSum(self.loss))
        if not self.debug:
            self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=metrics)

        #initialize model weights
        X, y = val_generator[0]
        self.model(X)

        best_ndcg = 0
        steps_since_improved = 0
        best_epoch = -1
        best_weights = self.model.get_weights()
        val_ndcg_history = []
        start_time = time.time()
        for epoch in range(self.train_epochs):
            val_generator.current_position=0
            generator = DataGenerator(train_users, self.max_history_length, self.items.size(),
                                              train_user_features,
                                              self.max_user_feature_hashes,
                                              batch_size=self.batch_size, target_decay=self.target_decay,
                                              num_actions_to_predict=self.num_target_predictions,
                                              positional=self.positional)
            print(f"epoch: {epoch}")
            val_ndcg = self.train_epoch(generator, val_generator, ndcg_metric)
            total_training_time = time.time() - start_time
            val_ndcg_history.append((total_training_time, val_ndcg))
            steps_since_improved += 1
            if val_ndcg > best_ndcg:
                steps_since_improved = 0
                best_ndcg = val_ndcg
                best_epoch = epoch
                best_weights = self.model.get_weights()
            print(f"val_ndcg: {val_ndcg}, best_ndcg: {best_ndcg}, steps_since_improved: "
                  f"{steps_since_improved}, total_training_time: {total_training_time}")
            if steps_since_improved >= self.early_stop_epochs:
                print(f"early stopped at epoch {epoch}")
                break

            if self.trainig_time_limit is not None and total_training_time > self.trainig_time_limit:
                print(f"time limit stop triggered at epoch {epoch}")
                break

            K.clear_session()
            gc.collect()

        self.model.set_weights(best_weights)
        self.metadata = {"epochs_trained": best_epoch + 1, "best_val_ndcg": best_ndcg,
                         "val_ndcg_history": val_ndcg_history}
        print(self.get_metadata())
        print(f"taken best model from epoch{best_epoch}. best_val_ndcg: {best_ndcg}")


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
            user_features = [self.user_features.get(self.users.get_id(user_id))]
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



    def train_epoch(self, generator, val_generator, ndcg_metric):
        if not self.debug:
            return self.train_epoch_prod(generator, val_generator)
        else:
            return self.train_epoch_debug(generator, ndcg_metric, val_generator)


    def train_epoch_prod(self, generator, val_generator):
        train_history = self.model.fit(generator, validation_data=val_generator)
        return train_history.history[f"val_ndcg_at_{self.eval_ndcg_at}"][-1]

    def train_epoch_debug(self, generator, ndcg_metric, val_generator):
        pbar = tqdm(generator, ascii=True)
        variables = self.model.variables
        loss_sum = 0
        ndcg_sum = 0
        num_batches = 0
        for X, y_true in pbar:
            num_batches += 1
            with tf.GradientTape() as tape:
                y_pred = self.model(X)
                loss_val = tf.reduce_mean(self.loss(y_true, y_pred))
            grad = tape.gradient(loss_val, variables)
            self.optimizer.apply_gradients(zip(grad, variables))
            ndcg = ndcg_metric(y_true=y_true, y_pred=y_pred)
            loss_sum += loss_val
            ndcg_sum += ndcg
            pbar.set_description(f"loss: {loss_sum/num_batches:.5f}, "
                                 f"ndcg_at_{self.eval_ndcg_at}: {ndcg_sum/num_batches:.5f}")
        val_loss_sum = 0
        val_ndcg_sum = 0
        num_val_samples = 0
        num_batches = 0
        for X, y_true in val_generator:
            num_batches += 1
            y_pred = self.model(X)
            loss_val = self.loss(y_true, y_pred)
            ndcg = ndcg_metric(y_true, y_pred)
            val_ndcg_sum += ndcg
            val_loss_sum += loss_val
            num_val_samples += y_true.shape[0]
        val_ndcg = val_ndcg_sum / num_batches
        return val_ndcg


    def get_similar_items(self, item_id, limit):
        raise (NotImplementedError)

    def to_str(self):
        raise (NotImplementedError)

    def from_str(self):
        raise (NotImplementedError)

    def train_val_split(self):
        all_user_ids = self.users_with_actions
        val_user_ids = [self.users.get_id(val_user) for val_user in self.val_users]
        train_user_ids = list(all_user_ids)
        val_users = self.user_actions_by_id_list(val_user_ids)
        train_users = self.user_actions_by_id_list(train_user_ids, val_user_ids)
        train_features = [self.user_features.get(id, list()) for id in train_user_ids]
        val_features = [self.user_features.get(id, list()) for id in val_user_ids]
        return train_users, train_user_ids, train_features, val_users, val_user_ids, val_features

