from collections import defaultdict

def group_by_user(actions):
    result = defaultdict(lambda: [])
    for action in actions:
        result[action.user_id].append(action)
    return result

def evaluate_recommender(recommender, actions, metrics, recommendations_limit=50):
    by_user = group_by_user(actions)
    metric_sum = defaultdict(lambda: 0.0)
    for user_id in by_user:
        for metric in metrics:
            metric_sum[metric.name] += metric(recommender.get_next_items(user_id, recommendations_limit), by_user[user_id])

    result = {}
    for metric in metric_sum:
        result[metric] = metric_sum[metric]/len(by_user)
    return result

    
