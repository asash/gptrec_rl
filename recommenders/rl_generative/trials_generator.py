
from collections import deque
from typing import Any, List
from aprec.api.action import Action
from multiprocessing.context import SpawnProcess, SpawnContext
import random

import numpy as np

from aprec.recommenders.rl_generative.utils import build_trial_result, get_seq_with_gt



class TrialsGenerator(object):
    def __init__(self, user_actions, items, users, model_config, pred_history_vectorizer,
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
        self.generator = None 
        self.model_config = model_config
        self.pred_history_vectorizer = pred_history_vectorizer
        self.model_checkpoint_file = model_checkpoint_file
        self.gen_limit = gen_limit
        self.batch_size = tuning_batch_size
        self.all_user_ids = list(self.user_actions.keys())

    def ensure_generator(self):
        if self.generator is None:
            from aprec.recommenders.rl_generative.generator import Generator
            self.generator = Generator(self.model_config, self.model_checkpoint_file, self.items, 
                                       self.pred_history_vectorizer, gen_limit=self.gen_limit)
        
        
    def random_trial(self):
        internal_user_id = self.get_random_user()
        trial_result = self.get_trial_result(internal_user_id)
        return trial_result 

    def get_random_user(self):
        while True:
            internal_user_id = random.choice(self.all_user_ids) 
            if len(self.user_actions[internal_user_id]) == 0:
                continue
            #ignore val users
            if self.users.reverse_id(internal_user_id) in self.val_users:
                continue
            break
        return internal_user_id

    def get_trial_result(self, internal_user_id):
        sequence, ground_truth = self.get_tuning_sequence(internal_user_id)
        gt_action = Action(user_id=self.users.reverse_id(internal_user_id), item_id=self.items.reverse_id(ground_truth), timestamp=0)
        sep_item_id = self.items.get_id('<SEP>')
        self.ensure_generator()
        recommendations, seq, logged_probs, mask_original = self.generator.generate(sequence, self.filter_seen, sep_item_id, greedy=False, train=False)
        items = self.items 
        reward_metric=self.reward_metric
        return build_trial_result(gt_action, recommendations, seq, items, reward_metric, logged_probs, mask_original)



    def get_tuning_sequence(self, internal_user_id):
        all_actions = self.user_actions[internal_user_id]
        sep_item_id = self.items.get_id('<SEP>')
        gt_action_index = np.random.randint(len(all_actions))
        return get_seq_with_gt(all_actions, sep_item_id, gt_action_index) 


    
    def next_tuning_batch(self):
        import tensorflow as tf
        batch_rewards = []
        batch_seqs = []
        batch_recs = []
        batch_logged_probs = []
        batch_mask_original = []
        
        for i in range(self.batch_size):
            trial_result = self.random_trial() 
            batch_rewards.append(trial_result.reward)
            batch_seqs.append(trial_result.seq)
            batch_recs.append(trial_result.recs)
            batch_logged_probs.append(trial_result.logged_probs)
            batch_mask_original.append(trial_result.mask_original)
            
        batch_rewards = tf.stack(batch_rewards, 0)
        batch_seqs = tf.stack(batch_seqs, 0)
        batch_recs = tf.stack(batch_recs, 0)
        batch_logged_probs = tf.stack(batch_logged_probs, 0)
        batch_mask_original = tf.stack(batch_mask_original, 0)
        return [batch_rewards, batch_seqs, batch_recs, batch_logged_probs, batch_mask_original]

class TrialsGenerratorProcess(object):
    def __init__(self, queue, *args, **kwargs):
        self.generator = TrialsGenerator(*args, **kwargs)
        self.queue = queue
    
    def __call__(self):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        while True:
            batch = self.generator.next_tuning_batch()
            self.queue.put(batch)
         

class TrialsGeneratorMultiprocess(object):
    def __init__(self, sampling_processess,sampling_queue_size, batch_cache_size=100, *args, **kwargs) -> None:
        ctx = SpawnContext()
        self.result_queue = ctx.Queue(sampling_queue_size)
        self.trials_generator = TrialsGenerratorProcess(self.result_queue, *args, **kwargs)
        self.sampling_processess = sampling_processess
        self.batches_cache = deque(maxlen=batch_cache_size)
        

    def __enter__(self):
        self.processors:List[SpawnProcess] = []
        for i in range(self.sampling_processess):
            self.processors.append(SpawnProcess(target=self.trials_generator))
            self.processors[-1].daemon = True 
            self.processors[-1].start()

        #add first batch to cache
        self.batches_cache.append(self.result_queue.get())
        return self

    def next_tuning_batch(self):
        while not self.result_queue.empty():
            self.batches_cache.append(self.result_queue.get())
        print(f"batch_cache_size: {len(self.batches_cache)}")
        result =  random.choice(self.batches_cache)
        return result
            

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for p in self.processors:
            p.terminate()
            p.join()

