from collections import defaultdict

def group_by_user(actions, max_actions_per_user):
    result = defaultdict(lambda: [])
    for action in actions:
        if len(result[action.user_id]) < max_actions_per_user:
            result[action.user_id].append(action)
        else:
            if action.timestamp  < max([a.timestamp for a in result[action.user_id]]):
                raise Exception("actions were not sorted in chronological order")
    return result

def evaluate_recommender(recommender, actions, metrics, max_test_actions_per_user, recommendations_limit=50):
    by_user = group_by_user(actions, max_test_actions_per_user)
    metric_sum = defaultdict(lambda: 0.0)
    for user_id in by_user:
        for metric in metrics:
            metric_sum[metric.name] += metric(recommender.get_next_items(user_id, recommendations_limit), by_user[user_id])

    result = {}
    for metric in metric_sum:
        result[metric] = metric_sum[metric]/len(by_user)
    return result

    
