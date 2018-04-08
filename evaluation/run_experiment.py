import sys
import importlib.util
import json

from split_actions import split_actions
from evaluate_recommender import evaluate_recommender
from n_actions_for_user import n_actions_for_user

def config():
    """ from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path"""

    spec = importlib.util.spec_from_file_location("config", sys.argv[1])
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def run_experiment(config):
    actions = list(config.DATASET) 
    result = []
    for fraction in config.FRACTIONS_TO_SPLIT:
        result.append({"train_fraction": fraction, "recommenders": {}})
        train, test = split_actions(actions, (fraction, 1 - fraction))
        test = n_actions_for_user(test, config.TEST_ACTION_PER_USER)
        for recommender_name in config.RECOMMENDERS:
            recommender = config.RECOMMENDERS[recommender_name]()
            for action in train:
                recommender.add_action(action)
            recommender.rebuild_model()
            evaluation_result = evaluate_recommender(recommender, test, config.METRICS)
            result[-1]['recommenders'][recommender_name] = evaluation_result
    return result

if __name__ == "__main__":
    config = config()
    result = run_experiment(config)
    print(json.dumps(result, indent=4))
            

