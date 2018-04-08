def precision(recommendations, actual_items, k):
    actual_set = set(actual_items)
    recommended = set([recommendation[0] for recommendation in recommendations[:k]])
    cool = recommended.intersection(actual_set)
    return len(cool) / len(recommended)
