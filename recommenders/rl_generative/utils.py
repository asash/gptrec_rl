import numpy as np

class TrialResult(object):
    def __init__(self, reward, seq, recs, recs_with_scores, gt_action, logged_probs, mask_original, full_probs) -> None:
        self.reward = reward
        self.seq = seq
        self.recs = recs
        self.gt_action = gt_action
        self.recs_with_scores = recs_with_scores
        self.logged_probs = logged_probs
        self.mask_original = mask_original
        self.full_probs = full_probs
        

def get_seq_with_gt(all_actions, sep_item_id, gt_action_index):
    ground_truth = all_actions[gt_action_index][1]
    actions = all_actions[:gt_action_index]
    sequence = [action[1] for action in actions]
    sequence.append(sep_item_id)
    return sequence, ground_truth

def build_trial_result(gt_action, recommendations, seq, items, reward_metric, logged_probs, mask_original, full_probs):
        recs_with_scores = [] 
        for i in range(len(recommendations)):
            score = 1/np.log2(i+2)
            item_id = items.reverse_id(recommendations[i])
            recs_with_scores.append((item_id, score))
        reward = reward_metric(recs_with_scores, [gt_action])
        return TrialResult(reward=reward, seq=seq, recs=recommendations, recs_with_scores=recs_with_scores, 
                           gt_action=gt_action, logged_probs=logged_probs, mask_original=mask_original, full_probs=full_probs)

def get_latest_checkpoint(checkpoint_dir):
    import os
    import glob
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*'))
    latest_checkpoint = None
    latest_timestamp = None
    for checkpoint in checkpoints:
        if os.path.isfile(checkpoint + "/__success__"):
            try:
                timestamp = os.path.getmtime(checkpoint + "/__success__")
                if latest_timestamp is None or timestamp > latest_timestamp:
                    latest_timestamp = timestamp
                    latest_checkpoint = checkpoint
            except FileNotFoundError: #file could be deleted by another process 
                continue
    return latest_checkpoint

