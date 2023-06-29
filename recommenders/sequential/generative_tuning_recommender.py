import random
import numpy as np
import tensorflow as tf
import tqdm
from aprec.api.action import Action
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.recommenders.recommender import Recommender
from aprec.recommenders.sequential.models.generative.gpt_rec_rl import RLGPT2RecConfig, RLGPT2RecModel
from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender
from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig


class GenerativeTuningRecommender(SequentialRecommender):
    def __init__(self, config: SequentialRecommenderConfig, pre_train_recommender_factory,
                 gen_limit=10, 
                 filter_seen=True,
                 clip_eps=0.2, reward_metric = NDCG(10), 
                 tuning_batch_size=8, 
                 max_tuning_steps = 1000,
                 validate_every_steps = 100
                 ):
        if (type(config.model_config) != RLGPT2RecConfig):
            raise ValueError("GenerativeTuningRecommender only works with RLGPT2Rec model")
        super().__init__(config)
        self.pre_train_recommender:Recommender = pre_train_recommender_factory()
        self.gen_limit = gen_limit
        self.filter_seen = filter_seen
        self.model: RLGPT2RecModel
        self.clip_eps = clip_eps
        self.reward_metric = reward_metric
        self.last_action_hold_out = {}
        self.tuning_batch_size = tuning_batch_size
        self.max_tuning_steps = max_tuning_steps
        self.validate_every_steps = validate_every_steps

    
    def add_action(self, action):
        super().add_action(action)

    def rebuild_model(self):
        self.sort_actions()
        for user in self.users.straight:
            for ts, internal_item_id in self.user_actions[self.users.get_id(user)][:-1]:
                item_id = self.items.reverse_id(internal_item_id)
                self.pre_train_recommender.add_action(Action(user_id=user, item_id=item_id, timestamp=ts))
        self.pre_train_recommender.rebuild_model()
        for user in self.users.straight:
            internal_user_id = self.users.get_id(user)
            last_action = self.user_actions[internal_user_id][-1]
            self.user_actions[internal_user_id] = self.user_actions[internal_user_id][:-1]
            self.last_action_hold_out[internal_user_id] = last_action
            last_user_timestamp = last_action[0] 
            sep_action = Action(user_id=user, item_id='<SEP>', timestamp=last_user_timestamp + 1)
            super().add_action(sep_action)
            user_recommendations = self.pre_train_recommender.recommend(user, self.gen_limit)
            current_timestamp = last_user_timestamp + 2
            for (item_id, score) in user_recommendations:
                action = Action(user_id=user, item_id=item_id, timestamp=current_timestamp)
                self.add_action(action)
                current_timestamp += 1
        super().rebuild_model()
        self.cleanup_pretraining_actions()
        self.tune()

    def get_seq_reverse_ids(self, seq):
        return [self.items.reverse_id(item_id) for ts, item_id in seq]
    
    def tune(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        for step in range(1, self.max_tuning_steps + 1): 
            print("Tuning step", step)
            print("generating...")
            with tf.GradientTape() as tape:
                tape.watch(self.model.trainable_variables)
                batch_ratios = []
                batch_rewards = []
                for i in tqdm.tqdm(range(self.tuning_batch_size),  ascii=True, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', position=0, leave=True, ncols=70):
                    ratio, reward = next(self.tune_seq_generator())
                    batch_ratios.append(ratio)
                    batch_rewards.append(reward)

                batch_ratios = tf.stack(batch_ratios, 0)
                batch_rewards = tf.cast(tf.expand_dims(tf.stack(batch_rewards, 0), axis=-1), 'float32')
                ratio_reward = batch_rewards * batch_ratios
                clipped_batch_ratios = tf.clip_by_value(batch_ratios, 1-self.clip_eps, 1+self.clip_eps)
                clipped_batch_ratio_reward = batch_rewards * clipped_batch_ratios
                ppo_loss = -tf.reduce_mean(tf.minimum(ratio_reward, clipped_batch_ratio_reward))
                grads = tape.gradient(ppo_loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                mean_reward = tf.reduce_mean(batch_rewards)
                print(f"Step {step}. Mean reward", mean_reward.numpy())
            if step % self.validate_every_steps == 0:
                self.validate(step)

    def validate(self, step):
        print("Validating...")
        rewards = []
        for user in tqdm.tqdm(self.val_users,  ascii=True, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', position=0, leave=True, ncols=70):
            internal_user_id = self.users.get_id(user)
            reward, ratio = self.get_ratio_reward(internal_user_id, greedy=True)
            rewards.append(reward)
        mean_reward = tf.reduce_mean(rewards)
        print(f"Validation at {step}. Mean reward", mean_reward.numpy())
    
    def tune_seq_generator(self):
        while True:
            all_user_ids = list(self.user_actions.keys())
            random.shuffle(all_user_ids)
            for internal_user_id in self.user_actions:
                if len(self.user_actions[internal_user_id]) == 0:
                    continue
                #ignore val users
                if self.users.reverse_id(internal_user_id) in self.val_users:
                    continue

                reward, ratio = self.get_ratio_reward(internal_user_id)
                yield ratio, reward

    def get_ratio_reward(self, internal_user_id, greedy=False):
        sequence, ground_truth = self.get_tuning_sequence(internal_user_id)
        gt_action = Action(user_id=self.users.reverse_id(internal_user_id), item_id=self.items.reverse_id(ground_truth), timestamp=0)
                
        sep_item_id = self.items.get_id('<SEP>')
        generated_seq, seq_logprob = self.generate(sequence, self.filter_seen, sep_item_id, greedy=greedy)

        recs = [] 
        for i in range(len(generated_seq)):
            score = 1/np.log2(i+2)
            item_id = self.items.reverse_id(generated_seq[i])
            recs.append((item_id, score))
        reward = self.reward_metric(recs, [gt_action])
        ratio = self.prob_ratio(seq_logprob)
        return reward,ratio

    #it always return one, but we are interested in gradiets, not in actual value
    def prob_ratio(self, logprob_tensor):
        old_logprob = tf.stop_gradient(logprob_tensor)
        result = tf.exp(logprob_tensor - old_logprob)
        return result

    def recommend(self, user_id, limit):
        internal_user_id = self.users.get_id(user_id)
        seq = self.get_pred_sequence(internal_user_id)
        sep_item_id = self.items.get_id('<SEP>')
        generated_seq, seq_logprob = self.generate(seq, self.filter_seen, sep_item_id, greedy=True)

        recs = [] 
        for i in range(len(generated_seq)):
            score = 1/np.log2(i+2)
            item_id = self.items.reverse_id(generated_seq[i])
            recs.append((item_id, score))
        return recs[:limit]

    def recommend_batch(self, recommendation_requests, limit):
        results = []
        for user_id, features in tqdm(recommendation_requests, ascii=True):
            results.append(self.recommend(user_id, limit, features))
        return results
   
    def generate(self, seq, filter_seen, sep_item_id, greedy=False):
        model_actions = [(0, action) for action in seq]
        mask = np.zeros([self.model.tokenizer.vocab_size+1], dtype='float32')
        mask[sep_item_id] = 1.0
        for i in range (len(model_actions)):
            mask[model_actions[i][1]] = 1.0
        mask[self.items.size():] = 1.0
        generated_tokens = []
        resulting_logprobs = []
        for i in range(self.gen_limit):
            seq = self.config.pred_history_vectorizer(model_actions) 
            tokens = self.model.tokenizer(seq, 1, self.model.data_parameters.sequence_length)
            attention_mask = tf.cast((tokens != -100), 'float32')
            next_token_logits = self.model.gpt(seq, attention_mask=attention_mask).logits[-1, :] 
            mask_score = min(tf.reduce_min(next_token_logits), 0) - 1e6 
            masked_logits = tf.where(mask, mask_score, next_token_logits) 
            sep_token_id = self.items.get_id('<SEP>')
            if not greedy:
                next_token = self.items.get_id('<SEP>')
                while next_token >= sep_token_id: #we don't want to generate SEP token or any other special tokens. Usually, this loop should only run once
                    next_token = tf.random.categorical(tf.expand_dims(masked_logits, 0), num_samples=1)[-1,0].numpy()
            else:
                next_token = tf.argmax(masked_logits[:sep_token_id]).numpy()
            model_actions.append((i+1, next_token))
            generated_tokens.append(next_token)
            log_probs = tf.nn.log_softmax(masked_logits)    
            resulting_logprobs.append(log_probs[next_token])
            mask[next_token] = 1.0
        return generated_tokens, tf.stack(resulting_logprobs, -1)

    def get_pred_sequence(self, internal_user_id):
        actions = self.user_actions[internal_user_id]
        sep_item_id = self.items.get_id('<SEP>')
        sequence = [action[1] for action in actions]
        sequence.append(sep_item_id)
        return sequence

    def get_tuning_sequence(self, internal_user_id):
        actions = self.user_actions[internal_user_id]
        ground_truth = actions[-1][1]
        sep_item_id = self.items.get_id('<SEP>')
        sequence = [action[1] for action in actions[:-1]]
        sequence.append(sep_item_id)
        return sequence, ground_truth 

    def cleanup_pretraining_actions(self):
        for internal_id in self.user_actions:
            self.user_actions[internal_id] = self.user_actions[internal_id][:-self.gen_limit-1]
            self.user_actions[internal_id].append(self.last_action_hold_out[internal_id])

            


        
        
    def set_val_users(self, val_users):
        self.pre_train_recommender.set_val_users(val_users)
        super().set_val_users(val_users)