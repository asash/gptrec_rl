from collections import Counter


def filter_popular_items(actions_generator, max_actions):
    actions = []
    items_counter = Counter()
    for action in actions_generator:
        actions.append(action)
        items_counter[action.item_id] += 1
    popular_items = set([item_id for (item_id, cnt) in items_counter.most_common(max_actions)])
    return filter(lambda action: action.item_id in popular_items, actions)