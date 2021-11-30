import gzip
import json
from collections import defaultdict
from tqdm import tqdm

from aprec.utils.os_utils import mkdir_p


def group_by_user(actions):
    result = defaultdict(list)
    for action in actions:
        result[action.user_id].append(action)
    return result


def evaluate_recommender(recommender, test_actions,
                         metrics, out_dir, recommender_name,
                         features_from_test=None, recommendations_limit=900):

    test_actions_by_user = group_by_user(test_actions)
    metric_sum = defaultdict(lambda: 0.0)
    all_user_ids = list(test_actions_by_user.keys())
    requests = []
    for user_id in all_user_ids:
        if features_from_test is not None:
            requests.append((user_id, features_from_test(test_actions)))
        else:
            requests.append((user_id, None))


    print("generating predictions...")
    all_predictions = recommender.recommend_batch(requests, recommendations_limit)

    print('calculating metrics...')
    user_docs = []
    for i in tqdm(range(len(all_user_ids)), ascii=True):
        user_id = all_user_ids[i]
        predictions = all_predictions[i]
        user_test_actions = test_actions_by_user[user_id]
        user_doc = {"user_id": user_id,
                    "metrics": {},
                    "test_actions": [action.to_json() for action in user_test_actions],
                    "predictions": [(prediction[0], float(prediction[1])) for prediction in predictions],
                    }
        for metric in metrics:
            metric_value = metric(predictions, test_actions_by_user[user_id])
            metric_sum[metric.name] += metric_value
            user_doc["metrics"][metric.name] = metric_value
        user_docs.append(user_doc)

    mkdir_p(f"{out_dir}/predictions/")
    predictions_filename = f"{out_dir}/predictions/{recommender_name}.json.gz"
    with gzip.open(predictions_filename, "w") as output:
        for user_doc in user_docs:
            try:
                output.write(json.dumps(user_doc).encode("utf-8") + b"\n")
            except:
                pass

    result = {}
    for metric in metric_sum:
        result[metric] = metric_sum[metric]/len(test_actions_by_user)
    return result

    
