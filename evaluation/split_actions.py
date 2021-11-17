from collections import defaultdict
import numpy as np

def split_actions(actions, fractions):
    """split actions into n lists by timestamp in chronological order"""

    fractions_sum = sum(fractions)
    fractions_real = [fraction / fractions_sum for fraction in fractions]
    actions_list = sorted([action for action in actions],
                           key = lambda action: action.timestamp)
    cummulative = 0.0
    num_actions = len(actions_list)
    result = []
    for fraction in fractions_real[:-1]:
        left_border = int(cummulative * num_actions)
        right_border = int((cummulative + fraction) * num_actions)
        result.append(actions_list[left_border:right_border])
        cummulative += fraction

    left_border = int(cummulative * num_actions)
    right_border = int(num_actions)
    result.append(actions_list[left_border:right_border])
    return result


def get_control_users(actions):
    result = set()
    for action in actions:
        if 'is_control' in action.data and action.data['is_control']:
            result.add(action.user_id)
    return result

def leave_one_out(actions, max_test_users=4000):
    sorted_actions = sorted(actions, key=lambda x: x.timestamp)
    users = defaultdict(list)
    for action in sorted_actions:
        users[action.user_id].append(action)
    train = []
    test = []
    control_users = get_control_users(actions)
    valid_user_selection = users.keys() - control_users
    test_user_ids = set(np.random.choice(list(valid_user_selection), max_test_users, replace=False))
    for user_id in users:
        if user_id in test_user_ids:
            train += users[user_id][:-1]
            test.append(users[user_id][-1])
        else:
            train += users[user_id]
    return sorted(train, key=lambda x: x.timestamp), sorted(test, key=lambda x: x.timestamp)

def random_split(actions, test_fraction = 0.3, max_test_users=4000):
    sorted_actions = sorted(actions, key=lambda x: x.timestamp)
    users = defaultdict(list)
    for action in sorted_actions:
        users[action.user_id].append(action)
    train = []
    test = []
    control_users = get_control_users(actions)
    valid_user_selection = users.keys() - control_users
    test_user_ids = set(np.random.choice(list(valid_user_selection), max_test_users, replace=False))
    for user_id in users:
        if user_id in test_user_ids:
            num_user_actions = len(users[user_id])
            num_test_actions = int(min(num_user_actions * test_fraction, 1))
            test_action_indices = set(np.random.choice(range(num_test_actions), num_test_actions, replace=False))
            for action_id in range(num_user_actions):
                if action_id in test_action_indices:
                    test.append(users[user_id][action_id])
                else:
                    train.append(users[user_id][action_id])
        else:
            train += users[user_id]
    return sorted(train, key=lambda x: x.timestamp), sorted(test, key=lambda x: x.timestamp)

