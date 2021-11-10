import os
import random
import sys
import importlib.util
import json
import copy 
import mmh3
import gzip

from aprec.utils.os_utils import shell

from split_actions import split_actions, leave_one_out
from evaluate_recommender import evaluate_recommender
from filter_cold_start import filter_cold_start
from tqdm import tqdm
import time
import tensorflow as tf



def config():
    """ from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path"""

    spec = importlib.util.spec_from_file_location("config", sys.argv[1])

    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    if len(sys.argv) > 2:
        config.out_file = open(sys.argv[2], 'w')
        config.out_dir = os.path.dirname(sys.argv[2])
    else:
        config.out_file = sys.stdout
        config.out_dir = os.getcwd()

    return config


class RecommendersEvaluator(object):
    def __init__(self, actions, recommenders, metrics, out_dir, data_splitter,  n_val_users, recommendations_limit, callbacks=()):
        self.actions = actions
        self.metrics = metrics
        self.recommenders = recommenders
        self.data_splitter = data_splitter
        self.callbacks = callbacks
        self.out_dir = out_dir
        self.features_from_test = None
        self.n_val_users = n_val_users
        self.train, self.test = self.data_splitter(actions)
        self.save_split(self.train, self.test)
        self.test = filter_cold_start(self.train, self.test)
        all_train_user_ids = list(set([action.user_id for action in self.train]))
        self.recommendations_limit = recommendations_limit
        random.shuffle(all_train_user_ids)
        self.val_user_ids = all_train_user_ids[:self.n_val_users]

    def set_features_from_test(self, features_from_test):
        self.features_from_test = features_from_test

    def __call__(self):
        result = {"recommenders": {}}

        for recommender_name in self.recommenders:
            print("evaluating {}".format(recommender_name))
            recommender = self.recommenders[recommender_name]()
            print("adding train actions...")
            for action in tqdm(self.train, ascii=True):
                recommender.add_action(action)
            recommender.set_val_users(self.val_user_ids)
            print("rebuilding model...")
            build_time_start = time.time()
            recommender.rebuild_model()
            build_time_end = time.time()
            print("done")

            print("calculating metrics...")
            evaluate_time_start = time.time()
            evaluation_result = evaluate_recommender(recommender, self.test,
                                                     self.metrics, self.out_dir,
                                                     recommender_name, self.features_from_test, 
                                                     recommendations_limit=self.recommendations_limit)
            evaluate_time_end = time.time()
            print("calculating metrics...")
            evaluation_result['model_build_time'] =  build_time_end - build_time_start
            evaluation_result['model_inference_time'] =  evaluate_time_end - evaluate_time_start
            evaluation_result['model_metadata'] = copy.deepcopy(recommender.get_metadata())
            print("done")
            print(json.dumps(evaluation_result))
            result['recommenders'][recommender_name] = evaluation_result
            for callback in self.callbacks:
                callback(recommender, recommender_name, evaluation_result, config)
            del(recommender)
        return result

    def save_split(self, train, test):
        self.save_actions(train, "train.json.gz")
        self.save_actions(test, "test.json.gz")

    def save_actions(self, actions, filename):
        with gzip.open(os.path.join(self.out_dir, filename), 'w') as output:
            for action in actions:
                output.write(action.to_json().encode('utf-8') + b"\n")

def real_hash(obj):
    str_val = str(obj)
    result = (mmh3.hash(str_val) + (1 << 31)) * 1.0 / ((1 << 32) - 1)
    return result

def run_experiment(config):
    result = []
    all_actions = list(config.DATASET)
    callbacks = ()
    if hasattr(config, 'CALLBACKS'):
        callbacks = config.CALLBACKS

    for users_fraction in config.USERS_FRACTIONS:
        every_user = 1/users_fraction
        print("read data...")
        print("use one out of every {} users ({}% fraction)".format(every_user, users_fraction*100))
        actions = list(filter( lambda action: real_hash(action.user_id)  < users_fraction,
                         all_actions))
        print("actions in dataset: {}".format(len(actions)))
        item_id_set = set([action.item_id for action in actions])
        user_id_set = set([action.user_id for action in actions])

        if hasattr(config, 'N_VAL_USERS'):
            n_val_users = config.N_VAL_USERS
        else:
            n_val_users = len(user_id_set) // 10

        if hasattr(config, 'RECOMMENDATIONS_LIMIT'):
            recommendations_limit = config.RECOMMENDATIONS_LIMIT
        else:
            recommendations_limit = 900

        print("number of items in the dataset: {}".format(len(item_id_set)))
        print("number of users in the dataset: {}".format(len(user_id_set)))
        print("number of val_users: {}".format(n_val_users))
        print("evaluating...")

        data_splitter = get_data_splitter(config)

        recommender_evaluator = RecommendersEvaluator(actions,
                                     config.RECOMMENDERS,
                                     config.METRICS,
                                     config.out_dir,
                                     data_splitter,
                                     n_val_users,
                                     recommendations_limit,
                                     callbacks)
        if  hasattr(config, 'FEATURES_FROM_TEST'):
            recommender_evaluator.set_features_from_test(config.FEATURES_FROM_TEST)
        result_for_fraction = recommender_evaluator()
        result_for_fraction['users_fraction'] = users_fraction
        result_for_fraction['num_items'] = len(item_id_set)
        result_for_fraction['num_users'] = len(user_id_set)
        result.append(result_for_fraction)
        write_result(config, result)
        shell(f"python3 statistical_signifficance_test.py --predictions-dir={config.out_dir}/predictions/ "
              f"--output-file={config.out_dir}/statistical_signifficance.json")

def get_data_splitter(config):
    if config.SPLIT_STRATEGY == "TEMPORAL_GLOBAL":
        split_fraction = config.FRACTION_TO_SPLIT
        return lambda actions: split_actions(actions, (split_fraction, 1 - split_fraction))
    elif config.SPLIT_STRATEGY == "LEAVE_ONE_OUT":
        return leave_one_out

def write_result(config, result):
    if config.out_file != sys.stdout:
        config.out_file.seek(0)
    config.out_file.write(json.dumps(result, indent=4))
    if config.out_file != sys.stdout:
        config.out_file.truncate()
        config.out_file.flush()


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    config = config()
    run_experiment(config)
    
