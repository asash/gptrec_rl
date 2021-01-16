import os
import sys
import importlib.util
import json
import copy 
import mmh3

from split_actions import split_actions, leave_one_out
from evaluate_recommender import evaluate_recommender
from filter_cold_start import filter_cold_start
from tqdm import tqdm
import time

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
    def __init__(self, actions, recommenders, metrics, data_splitter, callbacks=()):
        self.actions = actions
        self.metrics = metrics
        self.recommenders = recommenders
        self.data_splitter = data_splitter
        self.callbacks = callbacks
        self.features_from_test = None

    def set_features_from_test(self, features_from_test):
        self.features_from_test = features_from_test

    def __call__(self):
        result = {"recommenders": {}}
        train, test = self.data_splitter(self.actions)
        test = filter_cold_start(train, test)
        for recommender_name in self.recommenders:
            print("evaluating {}".format(recommender_name))
            recommender = self.recommenders[recommender_name]()
            print("adding train actions...")
            for action in tqdm(train):
                recommender.add_action(action)
            print("rebuilding model...")
            build_time_start = time.time()
            recommender.rebuild_model()
            build_time_end = time.time()
            print("done")

            print("calculating metrics...")
            evaluate_time_start = time.time()
            evaluation_result = evaluate_recommender(recommender, test, self.metrics, self.features_from_test)
            evaluate_time_end = time.time()
            print("calculating metrics...")
            evaluation_result['model_build_time'] =  build_time_end - build_time_start
            evaluation_result['model_inference_time'] =  evaluate_time_end - evaluate_time_start
            evaluation_result['model_metadata'] = copy.deepcopy(recommender.get_metadata())
            print("done")
            print(json.dumps(evaluation_result))
            result['recommenders'][recommender_name] = evaluation_result
            for callback in self.callbacks:
                callback(recommender, recommender_name, config)
            del(recommender)
        return result


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
        print("number of items in the dataset: {}".format(len(item_id_set)))
        print("number of users in the dataset: {}".format(len(user_id_set)))
        print("evaluating...")

        data_splitter = get_data_splitter(config)

        recommender_evaluator = RecommendersEvaluator(actions,
                                     config.RECOMMENDERS,
                                     config.METRICS,
                                     data_splitter,
                                     callbacks)
        if  hasattr(config, 'FEATURES_FROM_TEST'):
            recommender_evaluator.set_features_from_test(config.FEATURES_FROM_TEST)
        result_for_fraction = recommender_evaluator()
        result_for_fraction['users_fraction'] = users_fraction
        result_for_fraction['num_items'] = len(item_id_set)
        result_for_fraction['num_users'] = len(user_id_set)
        result.append(result_for_fraction)
        write_result(config, result)

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
    config = config()
    run_experiment(config)


