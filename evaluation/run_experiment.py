import sys
import importlib.util
import json
import copy 
import mmh3

from split_actions import split_actions
from evaluate_recommender import evaluate_recommender
from filter_cold_start import filter_cold_start
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
            for action in train:
                recommender.add_action(action)
            build_time_start = time.time()
            recommender.rebuild_model()
            build_time_end = time.time()

            evaluate_time_start = time.time()
            evaluation_result = evaluate_recommender(recommender, test, self.metrics)
            evaluate_time_end = time.time()
            evaluation_result['model_build_time'] =  build_time_end - build_time_start
            evaluation_result['model_inference_time'] =  evaluate_time_end - evaluate_time_start
            evaluation_result['model_metadata'] = copy.deepcopy(recommender.get_metadata())
            result['recommenders'][recommender_name] = evaluation_result
            del(recommender)
        return result

def run_experiment(config):
    every_user = int (1/config.USERS_FRACTION)
    print("read data...")
    print("use every {} user".format(every_user))
    actions = list(filter( lambda action: mmh3.hash(action.user_id) % every_user == 0,
                     config.DATASET))
    print("actions in dataset: {}".format(len(actions)))
    item_id_set = set([action.item_id for action in actions])
    print("number of items in dataset: {}".format(len(item_id_set)))
    print("evaluating...")
    recommender_evaluator = RecommendersEvaluator(actions,
                                 config.RECOMMENDERS, 
                                 config.METRICS)
    result = []
    for fraction in config.FRACTIONS_TO_SPLIT:
        print("evaluating for split fraction {:.3f}".format(fraction))
        result.append(recommender_evaluator(fraction))
    return list(result)

if __name__ == "__main__":
    config = config()
    result = run_experiment(config)
    config.out_file.write(json.dumps(result, indent=4))
    config.out_file.close()
            

