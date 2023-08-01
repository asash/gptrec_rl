
import tensorflow as tf
from aprec.api.action import Action
from aprec.recommenders.rl_generative.generator import Generator
import random

import numpy as np

from aprec.recommenders.rl_generative.utils import TrialResult, build_trial_result, get_seq_with_gt



class TrialsGenerator(object):
    def __init__(self, user_actions, items, users, model_config, recommender_config,
                 model_checkpoint_file, 
                val_users, reward_metric,
                tuning_batch_size,
                filter_seen=True, 
                gen_limit=10
                ):
        self.user_actions = user_actions
        self.items = items
        self.users = users
        self.val_users = val_users
        self.reward_metric = reward_metric
        self.filter_seen = filter_seen
        self.generator = Generator(model_config, model_checkpoint_file, items, recommender_config, gen_limit=gen_limit)
        self.trials_generator = self.trial_generator_func()
        self.batch_size = tuning_batch_size
        
    def trial_generator_func(self) -> TrialResult:
        while True:
            all_user_ids = list(self.user_actions.keys())
            random.shuffle(all_user_ids)
            for internal_user_id in all_user_ids:
                if len(self.user_actions[internal_user_id]) == 0:
                    continue
                #ignore val users
                if self.users.reverse_id(internal_user_id) in self.val_users:
                    continue

                trial_result = self.get_trial_result(internal_user_id)
                yield trial_result 

    def get_trial_result(self, internal_user_id):
        sequence, ground_truth = self.get_tuning_sequence(internal_user_id)
        gt_action = Action(user_id=self.users.reverse_id(internal_user_id), item_id=self.items.reverse_id(ground_truth), timestamp=0)
        sep_item_id = self.items.get_id('<SEP>')
        recommendations, seq = self.generator.generate(sequence, self.filter_seen, sep_item_id, greedy=False, train=False)
        items = self.items 
        reward_metric=self.reward_metric
        return build_trial_result(gt_action, recommendations, seq, items, reward_metric)



    def get_tuning_sequence(self, internal_user_id):
        all_actions = self.user_actions[internal_user_id]
        sep_item_id = self.items.get_id('<SEP>')
        gt_action_index = np.random.randint(len(all_actions))
        return get_seq_with_gt(all_actions, sep_item_id, gt_action_index) 


    
    def next_tuning_batch(self):
        batch_rewards = []
        batch_seqs = []
        batch_recs = []
        for i in range(self.batch_size):
            trial_result = next(self.trials_generator)
            batch_rewards.append(trial_result.reward)
            batch_seqs.append(trial_result.seq)
            batch_recs.append(trial_result.recs)
        batch_rewards = tf.stack(batch_rewards, 0)
        batch_seqs = tf.stack(batch_seqs, 0)
        batch_recs = tf.stack(batch_recs, 0)
        return [batch_rewards, batch_seqs, batch_recs]