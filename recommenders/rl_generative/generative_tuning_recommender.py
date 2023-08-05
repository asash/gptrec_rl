from collections import defaultdict
import json
import os
import tempfile
from typing import List

import numpy as np
import tensorflow as tf
import tqdm
from aprec.api.action import Action
from aprec.recommenders.rl_generative.generator import static_generate
from aprec.recommenders.rl_generative.plot_utils import plot_rewards_per_pos, plot_tradeoff_trajectory
from aprec.recommenders.rl_generative.pre_train_targets_builder import PreTrainTargetsBuilder
from aprec.recommenders.rl_generative.trials_generator import TrialsGeneratorMultiprocess 
from aprec.recommenders.rl_generative.utils import build_trial_result, get_seq_with_gt
from aprec.recommenders.rl_generative.value_model import ValueModel
from aprec.recommenders.sequential.models.generative.reward_metrics.ndcg_reward import NDCGReward
from aprec.recommenders.recommender import Recommender
from aprec.recommenders.sequential.models.generative.gpt_rec_rl import RLGPT2RecConfig, RLGPT2RecModel
from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender
from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig
from aprec.utils.os_utils import mkdir_p



class GenerativeTuningRecommender(SequentialRecommender):
    def __init__(self, config: SequentialRecommenderConfig, 
                 pre_train_recommender_factory=None,
                 pre_trained_checkpoint_dir=None,
                 gen_limit=10, 
                 filter_seen=True,
                 clip_eps=0.2, reward_metric = NDCGReward(10), 
                 tuning_batch_size=8, 
                 max_tuning_steps = 16000,
                 validate_every_steps = 100,
                 gae_lambda = 0.1,
                 gae_gamma = 0.1,
                 tradeoff_monitoring_rewards = [],
                 ppo_lr = 1e-4,
                 value_lr = 1e-4,
                 checkpoint_every_steps = 20, 
                 sampling_processessess = 8, 
                 sampling_que_size = 16,
                 validate_before_tuning = True
                 ):
        if (type(config.model_config) != RLGPT2RecConfig):
            raise ValueError("GenerativeTuningRecommender only works with RLGPT2Rec model")
        super().__init__(config)

        #only one pre_train_recommender_factory or pre_trained_checkpoint_dir should be provided
        if pre_train_recommender_factory is not None and pre_trained_checkpoint_dir is not None:
           raise ValueError("Only one pre_train_recommender_factory or pre_trained_checkpoint_dir should be provided")
        if pre_train_recommender_factory is not None:
            self.pre_train_recommender:Recommender = pre_train_recommender_factory()
        self.do_pre_train = pre_train_recommender_factory is not None
        self.pre_trained_checkpoint_dir = pre_trained_checkpoint_dir
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
        self.gae_gamma = gae_gamma
        self.gae_lambda = gae_lambda
        self.tradeoff_monitoring_rewards = tradeoff_monitoring_rewards
        self.tradeoff_trajectiories = defaultdict(list)
        self.ppo_lr = ppo_lr
        self.value_lr = value_lr
        self.checkpoint_every_steps = checkpoint_every_steps
        self.sampling_processessess = sampling_processessess
        self.sampling_queue_size = sampling_que_size
        self.tuning_step = None
        self.value_model = None
        self.last_validation_step = None
        self.data_stats = None
        self.best_checkpoint_name = None
        self.actions:List[Action] = []
        self.validate_before_tuning = validate_before_tuning
        
        


    def compute_data_stats(self):
        #we use this data for sanity check when we load model from checkpoint
        self.data_stats = {}
        self.data_stats["users"] = self.users.straight
        self.data_stats["SEP_ITEM_ID"] = self.items.get_id('<SEP>')
        self.data_stats["items"] = self.items.straight 
        self.data_stats["val_users"] = self.val_users
        self.data_stats["model_config"] = self.config.model_config.as_dict()        
        seq_lens = {}
        for user in self.users.straight:
            seq_lens[user] = len(self.user_actions[self.users.get_id(user)])
        self.data_stats["seq_lens"] = seq_lens
         
    
    def add_action(self, action):
        self.actions.append(action)

    #sort by user_id first, then by timestamp, then by item_id
    def sort_actions(self):
        self.actions.sort(key=lambda action: (action.user_id, action.timestamp, action.item_id))
        self.user_actions = defaultdict(list)
        for action in self.actions:
            self.user_actions[self.users.get_id(action.user_id)].append((action.timestamp, self.items.get_id(action.item_id)))
        

    def rebuild_model(self):
        self.sort_actions()
        self.compute_data_stats()
        if self.do_pre_train:
            self.pre_train()
        elif self.pre_trained_checkpoint_dir is not None:
            self.load_pre_trained_checkpoint(self.pre_trained_checkpoint_dir)
            self.pass_parameters()
        self.ensure_value_model() 
        self.tune()
    
    def pre_train(self):
        #hold_out last action for val users, so that pre-training doesn't use it (and doesn't overfit on it via early stopping mechanism)
        #we will restore it after pre-training and user them for tuning. 
        
        for user in self.val_users:
            self.last_action_hold_out[user] = self.user_actions[self.users.get_id(user)][-1]
            self.user_actions[self.users.get_id(user)] = self.user_actions[self.users.get_id(user)][:-1]

        for user in self.users.straight:
            #we don't pre-train on val users
            if user in self.val_users:
                continue
            for ts, internal_item_id in self.user_actions[self.users.get_id(user)][:-1]:
                item_id = self.items.reverse_id(internal_item_id)
                self.pre_train_recommender.add_action(Action(user_id=user, item_id=item_id, timestamp=ts))

        self.pre_train_recommender.rebuild_model()
        self.config.targets_builder = lambda: PreTrainTargetsBuilder(self.pre_train_recommender,
                                                                     sequence_vectorizer=self.config.pred_history_vectorizer,
                                                                     item_ids=self.items,
                                                                     generation_limit=self.gen_limit)
        self.items.get_id('<SEP>') #ensure that we have special item id for <SEP>

        super().rebuild_model()

        #restore last action for val users
        for user in self.val_users:
            self.user_actions[self.users.get_id(user)].append(self.last_action_hold_out[user])



    def save_pre_trained_checkpoint(self, checkpoint_name):
        checkpoint_dir = self.get_out_dir() + "/checkpoints/" + checkpoint_name
        mkdir_p(checkpoint_dir)
        with open(checkpoint_dir + "/data_stats.json", 'w') as f:
            json.dump(self.data_stats, f)
        self.model.save_weights(checkpoint_dir + "/model.h5")
        if self.value_model is not None:
            self.value_model.save_weights(checkpoint_dir + "/value_model.h5")

    def ensure_value_model(self):
        if self.value_model is None:
            self.value_model = ValueModel.from_config(self.model.get_config())
            self.value_model.gpt.set_weights(self.model.gpt.get_weights())
        
    def load_pre_trained_checkpoint(self, checkpoint_dir):
        print("Loading pre-trained checkpoint from", checkpoint_dir)
        with open(checkpoint_dir + "/data_stats.json", 'r') as f:
            other_data_stats = json.load(f)
            if other_data_stats != self.data_stats:
                raise ValueError("Data stats from checkpoint and from current data don't match")

        if self.model is not None:
            del self.model
            self.model = None
            
        self.model = self.get_model()
        self.model.load_weights(checkpoint_dir + "/model.h5")
        if os.path.isfile(checkpoint_dir + "/value_model.h5"):
            self.ensure_value_model()
            self.value_model.load_weights(checkpoint_dir + "/value_model.h5")


    def tune(self):
        self.save_best_weights()
        tensorboard_dir = self.get_tensorboard_dir() 
        mkdir_p(tensorboard_dir)
        tensorboard_writer = tf.summary.create_file_writer(tensorboard_dir)
        policy_optimizer = tf.keras.optimizers.Adam(learning_rate=self.ppo_lr)
        value_optimizer = tf.keras.optimizers.Adam(learning_rate=self.value_lr)
        checkpoints_dir = tempfile.mkdtemp()
        self.model.save_weights(checkpoints_dir + "/latest.h5")
        with TrialsGeneratorMultiprocess(sampling_processess=self.sampling_processessess, sampling_queue_size=self.sampling_queue_size, 
                                           user_actions=self.user_actions, items=self.items, users=self.users,
                                           model_config=self.model.get_config(), pred_history_vectorizer=self.config.pred_history_vectorizer,
                                           model_checkpoint_file=checkpoints_dir + "/latest.h5",
                                           reward_metric=self.reward_metric,
                                           tuning_batch_size=self.tuning_batch_size,
                                           filter_seen=self.filter_seen,
                                           val_users=self.val_users,
                                           gen_limit=self.gen_limit) as trials_generator:
            self.tuning_step = 1 
            while self.tuning_step < self.max_tuning_steps + 1: 
                with tensorboard_writer.as_default(step=self.tuning_step):
                    if ((self.tuning_step - 1) % self.validate_every_steps == 0) and  ((self.tuning_step > 1) or self.validate_before_tuning):
                            self.validate()
                            tensorboard_writer.flush()

                print("Tuning step", self.tuning_step)
                print("generating...")
                
                batch_rewards, batch_seqs, batch_recs, batch_logged_probs, batch_mask_original = trials_generator.next_tuning_batch()
                position_ids =  tf.concat([tf.range(self.model.data_parameters.sequence_length, -1, -1), tf.range(self.model.data_parameters.sequence_length + 1, self.model.data_parameters.sequence_length + self.gen_limit)], 0)
                position_ids = tf.tile(tf.expand_dims(position_ids, 0), [self.tuning_batch_size, 1])
                        
                with tf.GradientTape() as value_tape:
                    value_tape.watch(self.value_model.trainable_variables)
                    gae_advantages, values = self.get_gae_advantages(batch_seqs, batch_rewards)
                    discounted_rewards = self.discount_rewards(batch_rewards)
                    value_loss = tf.reduce_mean(tf.square(discounted_rewards - values))
                    value_grads = value_tape.gradient(value_loss, self.value_model.trainable_variables)
                    value_optimizer.apply_gradients(zip(value_grads, self.value_model.trainable_variables))

                with tf.GradientTape() as policy_tape:
                    tokens = self.model.tokenizer(batch_seqs, self.tuning_batch_size, self.model.data_parameters.sequence_length + self.gen_limit)
                    attention_mask = tf.cast((tokens != -100), 'float32')
                    logits = self.model.gpt(input_ids=tf.nn.relu(tokens), position_ids=position_ids, attention_mask=attention_mask, training=True).logits[:,-self.gen_limit:,:]
                    mask_score =  -1e6 
                    masked_logits = tf.where(batch_mask_original > 0, mask_score, logits) 
                    probs = tf.nn.softmax(masked_logits, -1)
                    rec_probs = tf.gather(probs, batch_recs, batch_dims=2) 
                    batch_ratios = tf.math.divide_no_nan(rec_probs, batch_logged_probs)
                    ratio_advantage = gae_advantages * batch_ratios
                    clipped_batch_ratios = tf.clip_by_value(batch_ratios, 1-self.clip_eps, 1+self.clip_eps)
                    clipped_batch_ratio_advantage = gae_advantages * clipped_batch_ratios
                    ppo_loss = -tf.reduce_mean(tf.minimum(ratio_advantage, clipped_batch_ratio_advantage))
                    policy_grads = policy_tape.gradient(ppo_loss, self.model.trainable_variables)
                    policy_optimizer.apply_gradients(zip(policy_grads, self.model.trainable_variables))
                    mean_reward = tf.reduce_mean(tf.reduce_sum(batch_rewards, -1))
                    print(f"Step {self.tuning_step}. Mean reward", mean_reward.numpy(), "ppo loss", ppo_loss.numpy(), "value loss", value_loss.numpy())
                    with tensorboard_writer.as_default(self.tuning_step):
                        tf.summary.scalar('tuning_train/ppo_loss', ppo_loss)
                        tf.summary.scalar('tuning_train/mean_reward', mean_reward)
                        tf.summary.scalar('tuning_train/value_loss', value_loss)


                if (self.tuning_step + 1) % self.checkpoint_every_steps == 0:
                    self.model.save_weights(checkpoints_dir + "/current.h5")    
                    os.rename(checkpoints_dir + "/current.h5", checkpoints_dir + "/latest.h5")
                self.tuning_step += 1

            #no need in final validation if we didn't do any tuning
            if (self.max_tuning_steps > 0) or self.validate_before_tuning:
                #final validation
                self.validate()
                tensorboard_writer.flush()       
        self.load_pre_trained_checkpoint(self.get_out_dir() + "/checkpoints/" + self.best_checkpoint_name)

    def get_gae_advantages(self, batch_seqs, batch_rewards):
        tokens = self.model.tokenizer(batch_seqs, batch_seqs.shape[0], self.model.data_parameters.sequence_length + self.gen_limit)
        attention_mask = tf.cast((tokens != -100), 'float32')
        sequence_positions = tf.range(self.model.data_parameters.sequence_length, -1, -1)
        recommendation_positions = tf.range(self.model.data_parameters.sequence_length + 1, self.model.data_parameters.sequence_length + self.gen_limit)
        positions = tf.tile(tf.expand_dims(tf.concat([sequence_positions, recommendation_positions], 0), 0), [self.tuning_batch_size, 1])
        output = self.value_model.gpt(input_ids=tf.nn.relu(tokens), 
                                      position_ids=positions,
                                      attention_mask=attention_mask,
                                      output_hidden_states=True, return_dict=True).hidden_states[-1]
        value_embeddings = output[:,-self.gen_limit:,:]
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

    def validate(self):
        print("Validating...")
        rewards = []
        recs_gts = []
        for user in tqdm.tqdm(self.val_users,  ascii=True, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', position=0, leave=True, ncols=70):
            internal_user_id = self.users.get_id(user)
            all_actions = self.user_actions[internal_user_id]
            sep_item_id = self.items.get_id('<SEP>')
            gt_action_index = len(all_actions) - 1
            seq, gt = get_seq_with_gt(all_actions, sep_item_id, gt_action_index)
            gt_action = Action(user, item_id=self.items.reverse_id(gt), timestamp=0) 
            val_recommendations, val_seq, val_logged_probs, val_mask = static_generate(seq, self.filter_seen, sep_item_id, greedy=True, 
                                                           train=False, items=self.items, gen_limit=self.gen_limit,
                                                            pred_history_vectorizer=self.config.pred_history_vectorizer,
                                                            model=self.model)
            trial_result = build_trial_result(gt_action, val_recommendations, val_seq, self.items, self.reward_metric, val_logged_probs, val_mask)                                 
            rewards.append(trial_result.reward)
            recs_gts.append((trial_result.recs_with_scores, trial_result.gt_action))
        mean_reward = tf.reduce_mean(tf.reduce_sum(rewards, -1))
        print(f"Validation at {self.tuning_step}. Mean reward", mean_reward.numpy())
        tf.summary.scalar('tuning_val/mean_reward', mean_reward)
        reward_distr = plot_rewards_per_pos(rewards)
        tf.summary.image("tuning_val/reward_distr", reward_distr, step=self.tuning_step)

        self.plot_tradeoffs(recs_gts)
        self.try_update_best_model(mean_reward, self.tuning_step)

    def plot_tradeoffs(self, recs_gts):
        for (metric1, metric2) in self.tradeoff_monitoring_rewards:
            tradeoff_name = f"{metric1.name}:::{metric2.name}"
            m1_sum = 0
            m2_sum = 0
            rewards1 = []
            rewards2 = []
            for recs in recs_gts:
                rewards1.append(metric1(recs[0], [recs[1]]))
                rewards2.append(metric2(recs[0], [recs[1]]))
                m1_sum += np.sum(rewards1[-1]) 
                m2_sum += np.sum(rewards2[-1]) 
            m1_sum /= len(recs_gts)
            m2_sum /= len(recs_gts)
            self.tradeoff_trajectiories[tradeoff_name].append((m1_sum, m2_sum))
            trajectory = plot_tradeoff_trajectory(self.tradeoff_trajectiories[tradeoff_name], tradeoff_name)
            print(f"{metric1.name}::{metric2.name} tradeoff at {self.tuning_step}. {metric1.name}", m1_sum, f"{metric2.name}", m2_sum)
            tf.summary.image(f"tuning_val/{tradeoff_name}_tradeoff", trajectory, step=self.tuning_step)
            metric1_distr = plot_rewards_per_pos(rewards1)
            metric2_distr = plot_rewards_per_pos(rewards2)
            tf.summary.image(f"tuning_val/{metric1.name}_distr", metric1_distr, step=self.tuning_step)
            tf.summary.image(f"tuning_val/{metric2.name}_distr", metric2_distr, step=self.tuning_step)


        
    def save_best_weights(self):
        best_checkpoint_name = f"checkpoint_step_{self.tuning_step}_mean_reward_{self.best_val_revard}"
        print("saving new best weights to", best_checkpoint_name)
        self.save_pre_trained_checkpoint(best_checkpoint_name)
        self.best_checkpoint_name = best_checkpoint_name
        
    #if mean reward improves, we save the model weights    
    def try_update_best_model(self, mean_reward, step):
        if mean_reward > self.best_val_revard:
            self.best_val_revard = mean_reward
            self.save_best_weights()
            print(f"New best model at step {step}. Mean reward", mean_reward.numpy())
            tf.summary.scalar('tuning_val/best_mean_reward', mean_reward)
    


    #it always return one, but we are interested in gradiets, not in actual value
    def prob_ratio(self, logprob_tensor):
        old_logprob = tf.stop_gradient(logprob_tensor)
        result = tf.exp(logprob_tensor - old_logprob)
        return result

    def recommend(self, user_id, limit, features=None):
        internal_user_id = self.users.get_id(user_id)
        seq = self.get_pred_sequence(internal_user_id)
        sep_item_id = self.items.get_id('<SEP>')
        user_recs = static_generate(seq, self.filter_seen, sep_item_id, greedy=True,
                                              train=False, items=self.items, gen_limit=self.gen_limit,
                                              pred_history_vectorizer=self.config.pred_history_vectorizer,
                                              model=self.model)[0]
        recs = [] 
        for i in range(len(user_recs)):
            score = 1/np.log2(i+2)
            item_id = self.items.reverse_id(user_recs[i])
            recs.append((item_id, score))
        return recs[:limit]

    def recommend_batch(self, recommendation_requests, limit):
        results = []
        for user_id, features in tqdm.tqdm(recommendation_requests,  ascii=True, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', position=0, leave=True, ncols=70):
            results.append(self.recommend(user_id, limit, features))
        return results

    def get_pred_sequence(self, internal_user_id):
        actions = self.user_actions[internal_user_id]
        sep_item_id = self.items.get_id('<SEP>')
        sequence = [action[1] for action in actions]
        sequence.append(sep_item_id)
        return sequence

    def set_val_users(self, val_users):
        if self.do_pre_train:
            self.pre_train_recommender.set_val_users(val_users)
        super().set_val_users(val_users)