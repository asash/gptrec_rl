import sys
import importlib.util
import json
import copy 
import mmh3

from split_actions import split_actions
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
    else:
        config.out_file = sys.stdout
    return config

class RecommendersEvaluator(object):
    def __init__(self, actions, recommenders, metrics):
        self.actions = actions
        self.metrics = metrics
        self.recommenders = recommenders

    def __call__(self, split_fraction):
        result = {"train_fraction": split_fraction, "recommenders": {}}
        train, test = split_actions(self.actions, (split_fraction, 1 - split_fraction))
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
            evaluation_result = evaluate_recommender(recommender, test, self.metrics)
            evaluate_time_end = time.time()
            print("calculating metrics...")
            evaluation_result['model_build_time'] =  build_time_end - build_time_start
            evaluation_result['model_inference_time'] =  evaluate_time_end - evaluate_time_start
            evaluation_result['model_metadata'] = copy.deepcopy(recommender.get_metadata())
            print("done")
            print(json.dumps(evaluation_result))
            result['recommenders'][recommender_name] = evaluation_result
            del(recommender)
        return result


def real_hash(obj):
    str_val = str(obj)
    result = (mmh3.hash(str_val) + (1 << 31)) * 1.0 / ((1 << 32) - 1)
    return result

def run_experiment(config):
    result = []
    all_actions = list(config.DATASET)
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
        recommender_evaluator = RecommendersEvaluator(actions,
                                     config.RECOMMENDERS,
                                     config.METRICS)
        split_fraction = config.FRACTION_TO_SPLIT
        print("evaluating for split fraction {:.3f}".format(split_fraction))
        result_for_fraction = recommender_evaluator(split_fraction)
        result_for_fraction['users_fraction'] = users_fraction
        result_for_fraction['split_fraction'] = split_fraction
        result_for_fraction['num_items'] = len(item_id_set)
        result_for_fraction['num_users'] = len(user_id_set)
        result.append(result_for_fraction)
        write_result(config, result)


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


