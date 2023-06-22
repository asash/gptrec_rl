import numpy as np
import tensorflow as tf
from aprec.api.action import Action
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.recommenders.recommender import Recommender
from aprec.recommenders.sequential.models.generative.gpt_rec import GPT2RecModel
from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender
from transformers import TFGPT2LMHeadModel

import tensorflow as tf

class FlagProxy(object):
    def __init__(self, recommender: SequentialRecommender) -> None:
        self.recommender = recommender
   
    def __getitem__(self, key):
        return self.recommender.flags[key] 

    def __setitem__(self, key, value):
        self.recommender.flags[key] = value


class TunedGPTRec(Recommender):
    def __init__(self, gpt_rec: SequentialRecommender, generate_length=10, reward_metric=NDCG(10)):
        self.gpt_rec = gpt_rec
        self.flags = FlagProxy(gpt_rec)
        self.generate_length = generate_length
        self.reward_metric = reward_metric
        self.pre_defined_scores =np.array([2**-i for i in range(0, generate_length)])
        
    def recommend(self, user_id, limit: int, features=None):
        return self.gpt_rec.recommend(user_id, limit, features)

    def recommend_batch(self, recommendation_requests, limit):
        return self.gpt_rec.recommend_batch(recommendation_requests, limit)
    
    def add_action(self, action: Action):
        self.gpt_rec.add_action(action)

    def set_val_users(self, val_users):
        return self.gpt_rec.set_val_users(val_users)
    
    def add_test_items_ranking_request(self, request):
        self.gpt_rec.add_test_items_ranking_request(request)
        
    def rebuild_model(self):
        self.gpt_rec.rebuild_model()
        self.tune_model()

    def tune_model(self):
        model: GPT2RecModel = self.gpt_rec.model
        gpt: TFGPT2LMHeadModel  = model.gpt
        while True:
            seq, truth = self.get_tuning_sequence()
            n_pre_generate = np.random.randint(0, self.generate_length-1)
            if n_pre_generate == 0:
                prompt=seq
            else:
                prompt = gpt.generate(seq, do_sample=False, num_beams=1, max_new_tokens=n_pre_generate) # assume first n items in the list are already generated. 
            ground_truth = [Action(user_id=0, item_id=truth, timestamp=0)]
            variables = gpt.trainable_variables
            with tf.GradientTape() as tape:
                logits = gpt(prompt)['logits'][0, -1, :][:self.gpt_rec.items.size()] 
                samples = tf.random.categorical([logits], 2)
                samples = tf.cast(tf.reshape(samples, [2, 1]), 'int64')
                expanded_prompt = tf.cast(tf.tile(prompt, [2, 1]), 'int64')
                full_prompt = tf.concat([expanded_prompt, samples], axis=1)
                generated = tf.stop_gradient(gpt.generate(full_prompt, do_sample=False, num_beams=1, max_new_tokens=self.generate_length - n_pre_generate - 1))[:,-self.generate_length:]
                
                recs1 = list(zip(generated[0].numpy(), self.pre_defined_scores))
                recs2 = list(zip(generated[1].numpy(), self.pre_defined_scores))

                score_1 = logits[samples[0, 0]]
                score_2 = logits[samples[1, 0]]

                reward1 = self.reward_metric(recs1, ground_truth)
                reward2 = self.reward_metric(recs2, ground_truth)
                
                if  reward1 == reward2:
                   continue # samples are equal, no need to tune the model 
                
                if reward1 > reward2:
                    loss = -tf.math.log_sigmoid(score_1 - score_2)
                
                else:
                    loss = -tf.math.log_sigmoid(score_2 - score_1)
                pass
        pass
        


    def get_tuning_sequence(self):
        while True:
            internal_id = np.random.choice(self.gpt_rec.users.size())
            external_id = self.gpt_rec.users.reverse_id(internal_id)
            is_val = external_id in self.gpt_rec.val_users
            actions = self.get_all_actions(external_id, is_val)
            if(len(actions) == 0):
                continue
            split_point = np.random.choice(len(actions))
            input_actions = actions[:split_point]
            target_action  = actions[split_point]
            eos_action = (0, self.gpt_rec.items.size() + 1)
            input_actions.append(eos_action)
            input_session = self.gpt_rec.config.pred_history_vectorizer(input_actions)
            return np.expand_dims(input_session, 0), target_action[-1]
        

    def get_all_actions(self, user_id, is_val=False):
        if not is_val:
            actions = self.gpt_rec.user_actions[self.gpt_rec.users.get_id(user_id)]
        else:
            actions = self.gpt_rec.user_actions[self.gpt_rec.users.get_id(user_id)][:-1]
        items_list = [action[1] for action in actions]
        model_actions = [(0, action) for action in items_list]
        return model_actions
        #session = self.config.pred_history_vectorizer(model_actions)
        #session = session.reshape(1, self.config.sequence_length)
        #model_inputs = [session]
        #return model_inputs