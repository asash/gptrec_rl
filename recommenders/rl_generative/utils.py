import numpy as np

class TrialResult(object):
    def __init__(self, reward, seq, recs, recs_with_scores, gt_action, logged_probs, mask_original) -> None:
        self.reward = reward
        self.seq = seq
        self.recs = recs
        self.gt_action = gt_action
        self.recs_with_scores = recs_with_scores
        self.logged_probs = logged_probs
        self.mask_original = mask_original
        

def get_seq_with_gt(all_actions, sep_item_id, gt_action_index):
    ground_truth = all_actions[gt_action_index][1]
    actions = all_actions[:gt_action_index]
    sequence = [action[1] for action in actions]
    sequence.append(sep_item_id)
    return sequence, ground_truth

def build_trial_result(gt_action, recommendations, seq, items, reward_metric, logged_probs, mask_original):
        recs_with_scores = [] 
        for i in range(len(recommendations)):
            score = 1/np.log2(i+2)
            item_id = items.reverse_id(recommendations[i])
            recs_with_scores.append((item_id, score))
        reward = reward_metric(recs_with_scores, [gt_action])
        return TrialResult(reward=reward, seq=seq, recs=recommendations, recs_with_scores=recs_with_scores, 
                           gt_action=gt_action, logged_probs=logged_probs, mask_original=mask_original)