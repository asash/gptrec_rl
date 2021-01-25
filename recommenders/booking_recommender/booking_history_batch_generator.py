import copy
import math
import random

import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map, thread_map

ACTION_FEATURES = [
    ['is_desktop', lambda action: int(action.data['device_class'] == 'desktop')],
    ['is_mobile', lambda action: int(action.data['device_class'] == 'mobile')],
    ['n_nights', lambda action: (action.data['checkout_date'] - action.data['checkin_date']).days],
    ['checkin_day_of_year', lambda action: ((action.data['checkin_date'].timetuple().tm_yday) + 1) / 365],
    ['checkout_day_of_year', lambda action: ((action.data['checkout_date'].timetuple().tm_yday) + 1) / 365],
    ['checkin_year_2016', lambda action: action.data['checkin_date'].year == 2015],
    ['checkin_year_2016', lambda action: action.data['checkin_date'].year == 2016],
    ['checkin_year_2017', lambda action: action.data['checkin_date'].year == 2017],
    ['checkin_weekday', lambda action: ((action.data['checkin_date'].weekday()) + 1) / 7],
    ['checkout_weekday', lambda action: ((action.data['checkout_date'].weekday()) + 1) / 7],
    ['over_weekend', lambda action: over_weekend(action.data['checkin_date'], action.data['checkout_date'])],
    ['same_country', lambda action: action.data['booker_country'] == action.data['hotel_country']],
    ['from_start', lambda action: action.data.get('from_start', 0)],
    ['remain', lambda action: action.data.get('remain', 0)]
]


def over_weekend(start_date, end_date):
    n_nighs = (end_date - start_date).days
    start_weekday = start_date.weekday()
    end_day = start_weekday + n_nighs
    if end_day > 13 or (start_weekday < 5 and end_day > 6):
        return 1
    return 0


class BookingHistoryBatchGenerator(Sequence):
    def __init__(self, user_actions, history_size, n_items, affiliates_dict,
                 country_dict,
                 city_dict,
                 candidates_recommender,
                 city_country_mapping,
                 n_candidates,
                 batch_size=1000, validation=False, target_decay=0.6,
                 min_target_val=0.03, epoch_size = 20000):
        self.user_actions = np.random.choice(user_actions, epoch_size, replace=False)
        self.history_size = history_size
        self.n_items = n_items
        self.batch_size = batch_size
        self.history_matrix = None
        self.target_matrix = None
        self.validation = validation
        self.city_dict = city_dict
        self.target_decay = target_decay
        self.min_target_val = min_target_val
        self.country_dict = country_dict
        self.n_candidates = n_candidates
        self.affiliates_dict = affiliates_dict
        self.candidates_recommender = candidates_recommender
        self.city_country_mapping = city_country_mapping
        self.reset()

    def reset(self):
        history, target = self.split_actions(self.user_actions)
        self.history_matrix = self.matrix_for_embedding(history, self.history_size, self.n_items)
        self.additional_features = self.additional_features(history, self.history_size)
        self.direct_position, self.reverse_position = self.positions(history, self.history_size)
        self.user_country_matrix = self.id_vectors(history, self.history_size, self.country_dict, 'booker_country')
        self.hotel_country_matrix = self.id_vectors(history, self.history_size, self.country_dict, 'hotel_country')
        self.affiliate_id_matrix = self.id_vectors(history, self.history_size, self.affiliates_dict, 'affiliate_id')
        self.target_features = self.get_target_trip_features(target)
        self.target_affiliate_ids = self.get_target_affiliate_ids(target, self.affiliates_dict)
        self.candidates, self.candidate_countries, self.candidate_features = self.get_candidates(history)
        self.target_matrix = self.get_target_matrix(target, self.candidates)
        self.current_position = 0
        self.max = self.__len__()
        pass

    def positions(self, trips, history_size):
        result_direct, result_reverse = [], []
        for trip in trips:
            result_direct.append(direct_positions(len(trip), history_size))
            result_reverse.append(reverse_positions(len(trip), history_size))
        return np.array(result_direct), np.array(result_reverse)

    def get_candidates(self, user_actions):
        result_candidate_cities = []
        result_candidate_countries = []
        result_candidate_features = []
        print('generating candidates...')
        candidates_generator = CandidatesGenerator(self.city_dict,
                                                   self.candidates_recommender,
                                                   self.n_candidates,
                                                   self.city_country_mapping)
        result = []
        for trip in tqdm(user_actions):
            actions = [item[1] for item in trip]
            result.append(candidates_generator(actions))

        for candidates, countries, features in result:
            result_candidate_cities.append(candidates)
            result_candidate_countries.append(countries)
            result_candidate_features.append(features)
        return np.array(result_candidate_cities), np.array(result_candidate_countries), np.array(result_candidate_features)

    @staticmethod
    def get_target_trip_features(user_actions):
        result = []
        for actions in user_actions:
            target_action = copy.deepcopy(actions[0][1])
            target_action.data['hotel_country'] = ""
            result.append(encode_action_features(target_action))
        return np.array(result)

    @staticmethod
    def get_target_affiliate_ids(user_actions, affiliates_dict):
        result = []
        for actions in user_actions:
            id = affiliates_dict.get_id(actions[0][1].data['affiliate_id'])
            result.append(id)
        return np.array(result)

    @staticmethod
    def additional_features(user_actions, history_size):
        additional_features = []
        for actions in user_actions:
            additional_features.append(encode_additional_features(actions, history_size))
        return np.array(additional_features)

    @staticmethod
    def id_vectors(user_actions, histor_size, translate_dict, property):
        result = []
        for actions in user_actions:
            result.append(id_vector(actions, histor_size, translate_dict, property))
        return np.array(result)

    @staticmethod
    def matrix_for_embedding(user_actions, history_size, n_items):
        result = []
        for actions in user_actions:
            result.append(history_to_vector(actions, history_size, n_items))
        return np.array(result)

    def get_target_matrix(self, user_actions, candidates):
        result = []
        for i in range(len(user_actions)):
            user_result = []
            target_val = {}
            cur_val = 0.99
            user_sum = 0
            for action_num in range(len(user_actions[i])):
                action = user_actions[i][action_num]
                target_val[action[0]] = cur_val
                cur_val *= self.target_decay
                if cur_val < self.min_target_val:
                    cur_val = self.min_target_val
            for j in range(self.n_candidates):
                city = candidates[i][j]
                if city in target_val:
                    user_result.append(target_val[city])
                    user_sum += target_val[city]
                else:
                    user_result.append(0.0)
            if user_sum == 0:
                user_result[random.randint(0, len(user_result) - 1)] = 0.001
            result.append(user_result)
        return np.array(result)

    def split_actions(self, user_actions):
        history = []
        target = []
        for user in user_actions:
            user_history, user_target = self.split_user(user)
            history.append(user_history)
            target.append(user_target)
        return history, target

    def split_user(self, user):
        if not self.validation:
            history_fraction = random.random()
            n_history_actions = max(1, int(len(user) * history_fraction))
        else:
            n_history_actions = len(user) - 1
        n_target_actions = len(user) - n_history_actions
        history_actions = user[:n_history_actions]
        target_actions = user[-n_target_actions:]
        return history_actions, target_actions

    def __len__(self):
        return math.floor(self.history_matrix.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        history = self.history_matrix[start:end]
        features = self.additional_features[start:end]
        user_countries = self.user_country_matrix[start:end]
        hotel_countries = self.hotel_country_matrix[start:end]
        affiliates = self.affiliate_id_matrix[start:end]
        target_features = self.target_features[start:end]
        target_affiliate_ids = self.target_affiliate_ids[start:end]
        target = self.target_matrix[start:end]
        candidate_cities = self.candidates[start:end]
        candidate_countries = self.candidate_countries[start:end]
        direct_pos = self.direct_position[start:end]
        reverse_pos = self.reverse_position[start:end]
        candidate_features = self.candidate_features[start:end]
        return [history, direct_pos, reverse_pos, features, user_countries, hotel_countries, affiliates, target_features, target_affiliate_ids,
                candidate_cities, candidate_countries, candidate_features], target

    def __next__(self):
        if self.current_position >= self.max:
            raise StopIteration()
        result = self.__getitem__(self.current_position)
        self.current_position += 1
        return result


def history_to_vector(user_actions, vector_size, special_value):
    if len(user_actions) >= vector_size:
        return np.array([action[0] for action in user_actions[-vector_size:]])
    else:
        n_special = vector_size - len(user_actions)
        result_list = [special_value] * n_special + [action[0] for action in user_actions]
        return np.array(result_list)


def encode_action_features(action):
    result = np.array([feature[1](action) for feature in ACTION_FEATURES])
    return np.array(result)


def encode_additional_features(actions, history_size):
    action_vectors = []
    taken_actions = actions[-history_size:]
    for i in range(len(taken_actions)):
        action = taken_actions[i]
        action[1].data['from_start'] = i + 1
        action[1].data['remain'] = len(taken_actions) - i
        action_vectors.append(encode_action_features(action[1]))
    result = np.array(action_vectors).reshape((len(taken_actions), len(ACTION_FEATURES)))
    n_pad = history_size - len(actions)
    if n_pad > 0:
        result = np.pad(result, ((n_pad, 0), (0, 0)), mode='constant', constant_values=0)
    return result


def id_vector(actions, history_size, translate_dict, property):
    take_actions = actions[-history_size:]
    result = np.array([translate_dict.get_id(action[1].data[property]) for action in take_actions])
    n_pad = history_size - len(actions)
    if n_pad > 0:
        result = np.pad(result, ((n_pad, 0)), mode='constant', constant_values=translate_dict.size())
    return result


class CandidatesGenerator(object):
    def __init__(self, city_dict, candidates_recommender, n_candidates, city_country_mapping):
        self.city_dict = city_dict
        self.candidates_recommender = candidates_recommender
        self.n_candidates = n_candidates
        self.city_country_mapping = city_country_mapping

    def __call__(self, trip):
        candidates_with_features = self.candidates_recommender.get_candidates_with_features(trip, self.n_candidates)
        candidates = [self.city_dict.get_id(candidate[0]) for candidate
                      in candidates_with_features]
        countries = [self.city_country_mapping[city] for city in candidates]
        candidate_features = [item[1] for item in candidates_with_features]
        return candidates, countries, candidate_features


def direct_positions(trip_len, history_size):
    if trip_len >= history_size:
        return list(range(1, history_size + 1))
    else:
        return [0] * (history_size - trip_len) + list(range(1, trip_len + 1))


def reverse_positions(trip_len, history_size):
    if trip_len >= history_size:
        return list(range(history_size, 0, -1))
    else:
        return [0] * (history_size - trip_len) + list(range(trip_len, 0, -1))
