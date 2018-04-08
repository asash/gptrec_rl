def precision(recommendations, actual_actions, k):
    actual_set = set([action.item_id for action in actual_actions])
    recommended = set([recommendation[0] for recommendation in recommendations[:k]])
    cool = recommended.intersection(actual_set)
    return len(cool) / len(recommended)
