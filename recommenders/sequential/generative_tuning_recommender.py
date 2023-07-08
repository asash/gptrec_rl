import random
import numpy as np
import tensorflow as tf
import tqdm
from aprec.api.action import Action
from aprec.recommenders.sequential.models.generative.reward_metrics.ndcg_reward import NDCGReward
from aprec.recommenders.recommender import Recommender
from aprec.recommenders.sequential.models.generative.gpt_rec_rl import RLGPT2RecConfig, RLGPT2RecModel
from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender
from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig
from aprec.utils.os_utils import mkdir_p


class GenerativeTuningRecommender(SequentialRecommender):
    def __init__(self, config: SequentialRecommenderConfig, pre_train_recommender_factory,
                 gen_limit=10, 
                 filter_seen=True,
                 clip_eps=0.2, reward_metric = NDCGReward(10), 
                 tuning_batch_size=8, 
                 max_tuning_steps = 1000,
                 validate_every_steps = 100,
                 gae_lambda = 0.7,
                 gae_gamma = 0.8,
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
        self.best_val_revard = float('-inf')
        self.best_weights = None
        self.gae_gamma = gae_gamma
        self.gae_lambda = gae_lambda

    
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
        self.save_best_weights()
        tensorboard_dir = self.get_tensorboard_dir() 
        mkdir_p(tensorboard_dir)
        tensorboard_writer = tf.summary.create_file_writer(tensorboard_dir)
        policy_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        value_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        tune_seq_generator = self.tune_seq_generator()
        self.value_model = RLGPT2RecModel.from_config(self.model.get_config())
        self.value_model.gpt.set_weights(self.model.gpt.get_weights())
        self.value_model.value_head = tf.keras.layers.Dense(1, name='value_head')

        for step in range(1, self.max_tuning_steps + 1): 
            print("Tuning step", step)
            print("generating...")
            
            #generate sequences and make a ppo step
            with tf.GradientTape() as policy_tape:
                with tf.GradientTape() as value_tape:
                    value_tape.watch(self.value_model.trainable_variables)
                    policy_tape.watch(self.model.trainable_variables)
                    batch_ratios = []
                    batch_rewards = []
                    batch_seqs = []
                    for i in tqdm.tqdm(range(self.tuning_batch_size),  ascii=True, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', position=0, leave=True, ncols=70):
                        ratio, reward, seq = next(tune_seq_generator)
                        batch_ratios.append(ratio)
                        batch_rewards.append(reward)
                        batch_seqs.append(seq)
                    batch_ratios = tf.stack(batch_ratios, 0)
                    batch_rewards = tf.stack(batch_rewards, 0)
                    batch_seqs = tf.stack(batch_seqs, 0)
                    gae_advantages, values = self.get_gae_advantages(batch_seqs, batch_rewards)
                    discounted_rewards = self.discount_rewards(batch_rewards)
                    value_loss = tf.reduce_mean(tf.square(discounted_rewards - values))
                    value_grads = value_tape.gradient(value_loss, self.value_model.trainable_variables)
                    value_optimizer.apply_gradients(zip(value_grads, self.value_model.trainable_variables))

                    ratio_advantage = gae_advantages * batch_ratios
                    clipped_batch_ratios = tf.clip_by_value(batch_ratios, 1-self.clip_eps, 1+self.clip_eps)
                    clipped_batch_ratio_advantage = gae_advantages * clipped_batch_ratios

                    ppo_loss = -tf.reduce_mean(tf.minimum(ratio_advantage, clipped_batch_ratio_advantage))

                    policy_grads = policy_tape.gradient(ppo_loss, self.model.trainable_variables)
                    policy_optimizer.apply_gradients(zip(policy_grads, self.model.trainable_variables))

                    mean_reward = tf.reduce_mean(tf.reduce_sum(batch_rewards, -1))
                    print(f"Step {step}. Mean reward", mean_reward.numpy())
                    with tensorboard_writer.as_default(step=step):
                        tf.summary.scalar('tuning_train/ppo_loss', ppo_loss)
                        tf.summary.scalar('tuning_train/mean_reward', mean_reward)
                        tf.summary.scalar('tuning_train/value_loss', value_loss)

                        if step % self.validate_every_steps == 0:
                            self.validate(step)
                            tensorboard_writer.flush()
        self.model.set_weights(self.best_weights)
        
    def get_gae_advantages(self, batch_seqs, batch_rewards):
        tokens = self.model.tokenizer(batch_seqs, batch_seqs.shape[0], self.model.data_parameters.sequence_length)
        attention_mask = tf.cast((tokens != -100), 'float32')
        output = self.value_model.gpt(input_ids=tokens, attention_mask=attention_mask, output_hidden_states=True, return_dict=True).hidden_states[-1]
        value_embeddings = output[:,-batch_rewards.shape[1]:,:]
        values = tf.squeeze(self.value_model.value_head(value_embeddings), -1)
        values_shifted = tf.concat([values[:,1:], tf.zeros([values.shape[0], 1], dtype='float32')], -1)
        deltas = batch_rewards + self.gae_gamma * values_shifted - values
        gae_advantages = self.compute_gae_advantages(deltas)
        return gae_advantages, values
    
    def compute_gae_advantages(self, deltas):
        gae_advantages = []
        for i in range(deltas.shape[1] -1, -1, -1):
            if i == deltas.shape[1] -1:
                gae_advantages.append(deltas[:, i])
            else:
                gae_advantages.append(deltas[:, i] + self.gae_gamma * self.gae_lambda * gae_advantages[-1])
        gae_advantages = tf.stack(gae_advantages[::-1], -1)
        return gae_advantages

    def discount_rewards(self, batch_rewards):
        discounted_rewards = []
        for i in range(batch_rewards.shape[1] -1, -1, -1):
            if i == batch_rewards.shape[1] -1:
                discounted_rewards.append(batch_rewards[:, i])
            else:
                discounted_rewards.append(batch_rewards[:, i] + self.gae_gamma * discounted_rewards[-1])
        discounted_rewards = tf.stack(discounted_rewards[::-1], -1)
        return discounted_rewards

    def validate(self, step):
        print("Validating...")
        rewards = []
        for user in tqdm.tqdm(self.val_users,  ascii=True, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', position=0, leave=True, ncols=70):
            internal_user_id = self.users.get_id(user)
            reward, ratio, seq = self.get_ratio_reward(internal_user_id, greedy=True)
            rewards.append(reward)
        mean_reward = tf.reduce_mean(tf.reduce_sum(rewards, -1))
        print(f"Validation at {step}. Mean reward", mean_reward.numpy())
        tf.summary.scalar('tuning_val/mean_reward', mean_reward)
        self.try_update_best_model(mean_reward, step)
        
    def save_best_weights(self):
        self.best_weights = self.model.get_weights()
        
    #if mean reward improves, we save the model weights    
    def try_update_best_model(self, mean_reward, step):
        if mean_reward > self.best_val_revard:
            self.best_val_revard = mean_reward
            self.save_best_weights()
            print(f"New best model at step {step}. Mean reward", mean_reward.numpy())
            tf.summary.scalar('tuning_val/best_mean_reward', mean_reward)
    
        
    
    def tune_seq_generator(self):
        while True:
            all_user_ids = list(self.user_actions.keys())
            random.shuffle(all_user_ids)
            for internal_user_id in all_user_ids:
                if len(self.user_actions[internal_user_id]) == 0:
                    continue
                #ignore val users
                if self.users.reverse_id(internal_user_id) in self.val_users:
                    continue

                reward, ratio, seq = self.get_ratio_reward(internal_user_id)
                yield ratio, reward, seq

    def get_ratio_reward(self, internal_user_id, greedy=False):
        sequence, ground_truth = self.get_tuning_sequence(internal_user_id)
        gt_action = Action(user_id=self.users.reverse_id(internal_user_id), item_id=self.items.reverse_id(ground_truth), timestamp=0)
                
        sep_item_id = self.items.get_id('<SEP>')
        recommendations, seq_logprob, seq = self.generate(sequence, self.filter_seen, sep_item_id, greedy=greedy)

        recs = [] 
        for i in range(len(recommendations)):
            score = 1/np.log2(i+2)
            item_id = self.items.reverse_id(recommendations[i])
            recs.append((item_id, score))
        reward = self.reward_metric(recs, [gt_action])
        ratio = self.prob_ratio(seq_logprob)
        return reward,ratio,seq

    #it always return one, but we are interested in gradiets, not in actual value
    def prob_ratio(self, logprob_tensor):
        old_logprob = tf.stop_gradient(logprob_tensor)
        result = tf.exp(logprob_tensor - old_logprob)
        return result

    def recommend(self, user_id, limit, features=None):
        internal_user_id = self.users.get_id(user_id)
        seq = self.get_pred_sequence(internal_user_id)
        sep_item_id = self.items.get_id('<SEP>')
        generated_seq, seq_logprob, seq = self.generate(seq, self.filter_seen, sep_item_id, greedy=True)

        recs = [] 
        for i in range(len(generated_seq)):
            score = 1/np.log2(i+2)
            item_id = self.items.reverse_id(generated_seq[i])
            recs.append((item_id, score))
        return recs[:limit]

    def recommend_batch(self, recommendation_requests, limit):
        results = []
        for user_id, features in tqdm.tqdm(recommendation_requests, ascii=True):
            results.append(self.recommend(user_id, limit, features))
        return results
   
    def generate(self, input_seq, filter_seen, sep_item_id, greedy=False):
        model_actions = [(0, action) for action in input_seq]
        mask = np.zeros([self.model.tokenizer.vocab_size+1], dtype='float32')
        mask[sep_item_id] = 1.0
        if filter_seen:
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
        return generated_tokens, tf.stack(resulting_logprobs, -1), seq 

    def get_pred_sequence(self, internal_user_id):
        actions = self.user_actions[internal_user_id]
        sep_item_id = self.items.get_id('<SEP>')
        sequence = [action[1] for action in actions]
        sequence.append(sep_item_id)
        return sequence

    def get_tuning_sequence(self, internal_user_id):
        all_actions = self.user_actions[internal_user_id]
        gt_action_index = np.random.randint(len(all_actions))
        ground_truth = all_actions[gt_action_index][1]
        actions = all_actions[:gt_action_index]
        sep_item_id = self.items.get_id('<SEP>')
        sequence = [action[1] for action in actions]
        sequence.append(sep_item_id)
        return sequence, ground_truth 

    def cleanup_pretraining_actions(self):
        for internal_id in self.user_actions:
            self.user_actions[internal_id] = self.user_actions[internal_id][:-self.gen_limit-1]
            self.user_actions[internal_id].append(self.last_action_hold_out[internal_id])

            


        
        
    def set_val_users(self, val_users):
        self.pre_train_recommender.set_val_users(val_users)
        super().set_val_users(val_users)