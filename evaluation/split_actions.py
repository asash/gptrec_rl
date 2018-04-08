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
