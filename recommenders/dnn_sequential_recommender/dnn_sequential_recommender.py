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
    def __init__(self,
                 model_arch: SequentialRecsysModel,
                 loss: Loss=BCELoss(),
                 train_epochs=300,
                 optimizer='adam',
                 batch_size=1000,
                 early_stop_epochs=100,
                 target_decay = 1.0,
                 train_on_last_item_only = False,
                 training_time_limit = None,
                 sigma=1,
                 eval_ndcg_at=40,
                 ):
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
        self.eval_ndcg_at=40
        self.eval_ndcg_at = eval_ndcg_at
        self.train_on_last_item_only = train_on_last_item_only
        self.training_time_limit = training_time_limit

    def get_metadata(self):
        return self.metadata

    def set_loss(self, loss):
        self.loss = loss

    def name(self):
        return self.model

    def add_action(self, action):
        user_id_internal = self.users.get_id(action.user_id)
        action_id_internal = self.items.get_id(action.item_id)
        self.user_actions[user_id_internal].append((action.timestamp, action_id_internal))

    #exclude last action for val_users
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

        train_users, train_user_ids, val_users, val_user_ids = self.train_val_split()

        print("train_users: {}, val_users:{}, items:{}".format(len(train_users), len(val_users), self.items.size()))
        val_generator = DataGenerator(val_users, val_user_ids, self.model_arch.max_history_length, self.items.size(),
                                      batch_size=self.batch_size, last_item_only=True,
                                      target_decay=self.target_decay,
                                      user_id_required = self.model_arch.requires_user_id
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
            generator = DataGenerator(train_users, train_user_ids, self.model_arch.max_history_length, self.items.size(),
                                      batch_size=self.batch_size, target_decay=self.target_decay,
                                      user_id_required = self.model_arch.requires_user_id,
                                      last_item_only=self.train_on_last_item_only
                                      )
            print(f"epoch: {epoch}")
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
        all_user_ids = set(range(0, self.users.size()))
        val_user_ids = [self.users.get_id(val_user) for val_user in self.val_users]
        train_user_ids = list(all_user_ids)
        val_users = self.user_actions_by_id_list(val_user_ids)
        train_users = self.user_actions_by_id_list(train_user_ids, val_user_ids)
        return train_users, train_user_ids, val_users, val_user_ids

    def get_model(self):
        self.model_arch.set_common_params(num_items=self.items.size(), num_users=self.users.size())
        model = self.model_arch.get_model()
        ndcg_metric = KerasNDCG(self.eval_ndcg_at)
        metrics = [ndcg_metric]
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=metrics)
        return model

    def get_next_items(self, user_id, limit, features=None):
        actions = self.user_actions[self.users.get_id(user_id)]
        items_list = [action[1] for action in actions]
        model_actions = [(0, action) for action in items_list]
        session = actions_to_vector(model_actions, self.model_arch.max_history_length, self.items.size())
        session = session.reshape(1, self.model_arch.max_history_length)
        model_inputs = [session]
        if (self.model_arch.requires_user_id):
            model_inputs.append(np.array([[self.users.get_id(user_id)]]))
        scores = self.model.predict(model_inputs)[0]
        best_ids = np.argsort(scores)[::-1][:limit]
        result = [(self.items.reverse_id(id), scores[id]) for id in best_ids]
        return result


