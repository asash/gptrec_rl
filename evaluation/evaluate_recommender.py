from collections import defaultdict
from tqdm import tqdm

def group_by_user(actions):
    result = defaultdict(list)
    for action in actions:
        result[action.user_id].append(action)
    return result

class GetPredictions(object):
    def __init__(self, recommender, limit):
        self.recommender = recommender
        self.limit = limit

    def __call__(self, request):
        user_id, features = request
        return self.recommender.get_next_items(user_id, self.limit, features)

def evaluate_recommender(recommender, actions, metrics, features_from_test=None, recommendations_limit=900):
    by_user = group_by_user(actions)
    metric_sum = defaultdict(lambda: 0.0)
    get_predictions = GetPredictions(recommender, recommendations_limit)
    all_user_ids = list(by_user.keys())
    requests = []
    for user_id in all_user_ids:
        if features_from_test is not None:
            requests.append((user_id, features_from_test(actions)))
        else:
            requests.append((user_id, None))


    print("generating predictions...")
    all_predictions = list(tqdm(map(get_predictions, requests), total=len(all_user_ids)))

    print('calculating metrics...')
    for i in tqdm(range(len(all_user_ids))):
        user_id = all_user_ids[i]
        predictions = all_predictions[i]
        for metric in metrics:
            metric_sum[metric.name] += metric(predictions, by_user[user_id])

    result = {}
    for metric in metric_sum:
        result[metric] = metric_sum[metric]/len(by_user)
    return result

    
