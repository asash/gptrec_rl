import random

import lightgbm
import numpy as np
from collections import defaultdict
from lightgbm import LGBMRanker

from aprec.recommenders.booking_recommender.candidates_recommender import BookingCandidatesRecommender
from aprec.recommenders.booking_recommender.neural_ranker import NeuralRanker
from aprec.utils.item_id import ItemId
from aprec.recommenders.recommender import Recommender
from aprec.recommenders.booking_recommender.booking_history_batch_generator import BookingHistoryBatchGenerator, \
    CandidatesGenerator, encode_action_features, ACTION_FEATURES

NUM_AFFILIATES = 3612
NUM_CITIES = 39903
NUM_COUNTRIES = 197



query_group = 0
def process_batch(candidate_features, target_features, target_batch):
    batch_size = len(target_batch)
    result_x = []
    result_y = []
    for query_group in range(batch_size):
        query_group_size = len(target_batch[query_group])
        for candidate in range(query_group_size):
            query_features = target_features[query_group]
            pair_features = candidate_features[query_group][candidate]
            features = np.concatenate([query_features, pair_features])
            target = int((target_batch[query_group][candidate]) * 31)
            result_x.append(features)
            result_y.append(target)
    return result_x, result_y



class BookingRecommenderLTR(Recommender):
    def __init__(self,n_val_users=1000,
                 batch_size = 1000,
                 candidates_cnt = 50, epoch_size=20000,
                 val_epoch_size=2000,
                 num_training_samples = 1000000,
                 model_type = 'lightgbm', attention=False, lgbm_boosting_type='gbdt', lgbm_objective='lambdarank'):
        print(f"Creating LTR recommender. Model_type{model_type}, "
              f"Lgbm_boosting_type:{lgbm_boosting_type}, lgbm_objective:{lgbm_objective}")
        self.users = ItemId()
        self.items = ItemId()
        self.countries = ItemId()
        self.affiliates = ItemId()
        self.user_actions = defaultdict(lambda: [])
        self.model = None
        self.user_vectors = None
        self.matrix = None
        self.mean_user = None
        self.n_val_users = n_val_users
        self.metadata = {}
        self.batch_size = batch_size
        self.candidates_cnt = candidates_cnt
        self.city_country_mapping = {}
        self.candidates_recommender = BookingCandidatesRecommender()
        self.epoch_size = epoch_size
        self.val_epoch_size = val_epoch_size
        self.target_decay = 0.5
        self.min_target_value = 0.0
        self.num_training_samples = num_training_samples
        self.model_type = model_type
        self.attention=attention
        self.lgbm_boosting_type = lgbm_boosting_type
        self.lgbm_objective = lgbm_objective


    def get_metadata(self):
        return self.metadata

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


        val_generator = BookingHistoryBatchGenerator(val_users, 0, self.items.size(),
                                                     batch_size=self.batch_size, country_dict = self.countries,
                                                     candidates_recommender=self.candidates_recommender,
                                                     city_dict=self.items,
                                                     affiliates_dict = self.affiliates,
                                                     validation=True,
                                                     city_country_mapping = self.city_country_mapping,
                                                     target_decay=self.target_decay,
                                                     n_candidates=self.candidates_cnt,
                                                     min_target_val=self.min_target_value, epoch_size=self.val_epoch_size)
        val_x, val_y = [], []
        for sample_x, sample_y in self.dataset_generator(val_generator):
            val_x.append(sample_x)
            val_y.append(sample_y)
        val_x = np.array(val_x)
        val_y = np.array(val_y)
        val_qg = [self.candidates_cnt] * (len(val_y) // self.candidates_cnt)

        x = []
        y = []
        while(len(y)) < self.num_training_samples:
            generator = BookingHistoryBatchGenerator(train_users, 0, self.items.size(),
                                                     batch_size=self.batch_size, country_dict = self.countries,
                                                     affiliates_dict = self.affiliates,
                                                     candidates_recommender=self.candidates_recommender,
                                                     city_dict=self.items,
                                                     city_country_mapping=self.city_country_mapping,
                                                     n_candidates=self.candidates_cnt,
                                                     target_decay=self.target_decay,
                                                     min_target_val=self.min_target_value, epoch_size=self.epoch_size)

            for sample_x, sample_y in self.dataset_generator(generator):
                x.append(sample_x)
                y.append(sample_y)
        qg = [self.candidates_cnt] * (len(y) // self.candidates_cnt)
        x = np.array(x)
        y = np.array(y)
        if self.model_type == 'lightgbm':
            bagging_fraction, bagging_freq = None, None
            if self.lgbm_boosting_type == 'rf':
                bagging_fraction=0.1
                bagging_freq=5

            callback = lightgbm.early_stopping(40, first_metric_only=True, verbose=True)

            self.model = LGBMRanker(n_estimators=500, boosting_type=self.lgbm_boosting_type,
                                    objective=self.lgbm_objective, bagging_fraction=bagging_fraction,
                                    bagging_freq=bagging_freq)
            self.model.fit(x, y, group=qg, eval_set=[(val_x, val_y)], eval_group=[val_qg],
                           eval_metric='ndcg', eval_at=[40], callbacks=[callback])
        elif self.model_type == 'neural':
            self.model = NeuralRanker(x.shape[-1], self.candidates_cnt, self.batch_size, attention=self.attention)
            self.model.fit(x, y, val_x, val_y)


    def get_next_items(self, user_id, limit, features=None):
        actions = self.user_actions[self.users.get_id(user_id)]
        items = [action[1] for action in actions]
        return self.get_model_predictions(items, limit, target_action=features)

    def get_model_predictions(self, items_list, limit, target_action):
        candidates_generator = CandidatesGenerator(self.items,
                                                   self.candidates_recommender,
                                                   self.candidates_cnt,
                                                   self.city_country_mapping)
        candidates, countries, candidate_features = candidates_generator(items_list)
        target_features = encode_action_features(target_action).reshape(len(ACTION_FEATURES))
        target_batch = [0] * self.candidates_cnt
        x, y = process_batch([candidate_features], [target_features], [target_batch])
        scores = self.model.predict(x)
        result = []
        for i in range(len(candidates)):
            item = self.items.reverse_id(candidates[i])
            score = scores[i]
            result.append((item, score))
        result = sorted(result, key=lambda x: -x[1])
        return result[:limit]



    def dataset_generator(self, batch_generator):
        for data_batch, target_batch in batch_generator:
            (history, direct_pos, reverse_pos, features, user_countries, hotel_countries, affiliates,
             target_features, target_affiliate_ids,
             candidate_cities, candidate_countries, candidate_features) = data_batch
            x, y =  process_batch(candidate_features, target_features, target_batch)
            for i in range(len(y)):
                yield (x[i], y[i])


