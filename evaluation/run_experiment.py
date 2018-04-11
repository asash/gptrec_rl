import sys
import importlib.util
import json
import copy 

from multiprocessing import Pool

from split_actions import split_actions
from evaluate_recommender import evaluate_recommender
from n_actions_for_user import n_actions_for_user

def config():
    """ from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path"""

    spec = importlib.util.spec_from_file_location("config", sys.argv[1])
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

class RecommendersEvaluator(object):
    def __init__(self, actions, recommenders, metrics):
        self.actions = actions
        self.metrics = metrics
        self.recommenders = recommenders

    def __call__(self, split_fraction):
        result = {"train_fraction": split_fraction, "recommenders": {}}
        train, test = split_actions(self.actions, (split_fraction, 1 - split_fraction))
        for recommender_name in self.recommenders:
            recommender = self.recommenders[recommender_name]()
            for action in train:
                recommender.add_action(action)
            recommender.rebuild_model()
            evaluation_result = evaluate_recommender(recommender, test, self.metrics)
            result['recommenders'][recommender_name] = evaluation_result
        return result

def run_experiment(config):
    print("read data...")
    actions = list(config.DATASET) 
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
    print(json.dumps(result, indent=4))
            

