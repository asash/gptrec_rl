from collections import defaultdict
import gc
import io
import os
import random
import tempfile
from matplotlib import pyplot as plt

import numpy as np
import tensorflow as tf
import tqdm
from aprec.api.action import Action
from aprec.recommenders.rl_generative.generator import static_generate
from aprec.recommenders.rl_generative.pre_train_targets_builder import PreTrainTargetsBuilder
from aprec.recommenders.rl_generative.trials_generator import TrialsGeneratorMultiprocess 
from aprec.recommenders.rl_generative.utils import build_trial_result, get_seq_with_gt
from aprec.recommenders.sequential.models.generative.reward_metrics.ndcg_reward import NDCGReward
from aprec.recommenders.recommender import Recommender
from aprec.recommenders.sequential.models.generative.gpt_rec_rl import RLGPT2RecConfig, RLGPT2RecModel
from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender
from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig
from aprec.utils.os_utils import mkdir_p


def plot_to_image(func):
  def wrapper(*args, **kwargs):
        figure = func(*args, **kwargs)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        return image
  return wrapper

@plot_to_image
def plot_rewards_per_pos(rewards):
    rewards = np.array(rewards)
    fig =plt.figure(figsize=(4, 2.5))
    ax = fig.add_subplot()
    ax.violinplot(rewards, showmeans=True)
    ax.set_xlabel("Position")
    ax.set_ylabel("Reward")
    ax.set_yscale('log')
    return fig
    
 
class GenerativeTuningRecommender(SequentialRecommender):
    def __init__(self, config: SequentialRecommenderConfig, pre_train_recommender_factory,
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
                 sampling_que_size = 16
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
        self.tradeoff_monitoring_rewards = tradeoff_monitoring_rewards
        self.tradeoff_trajectiories = defaultdict(list)
        self.ppo_lr = ppo_lr
        self.value_lr = value_lr
        self.checkpoint_every_steps = checkpoint_every_steps
        self.sampling_processessess = sampling_processessess
        self.sampling_queue_size = sampling_que_size

    
    def add_action(self, action):
        super().add_action(action)

    def rebuild_model(self):
        self.sort_actions()
        self.pre_train()
        self.tune()

    def pre_train(self):
        for user in self.users.straight:
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

    def tune(self):
        self.save_best_weights()
        tensorboard_dir = self.get_tensorboard_dir() 
        mkdir_p(tensorboard_dir)
        tensorboard_writer = tf.summary.create_file_writer(tensorboard_dir)
        policy_optimizer = tf.keras.optimizers.Adam(learning_rate=self.ppo_lr)
        value_optimizer = tf.keras.optimizers.Adam(learning_rate=self.value_lr)
        checkpoints_dir = tempfile.mkdtemp()
        self.model.save_weights(checkpoints_dir + "/latest.h5")
        self.value_model = RLGPT2RecModel.from_config(self.model.get_config())
        self.value_model.gpt.set_weights(self.model.gpt.get_weights())
        self.value_model.value_head = tf.keras.layers.Dense(1, name='value_head')
        with TrialsGeneratorMultiprocess(sampling_processess=self.sampling_processessess, sampling_queue_size=self.sampling_queue_size, 
                                           user_actions=self.user_actions, items=self.items, users=self.users,
                                           model_config=self.model.get_config(), pred_history_vectorizer=self.config.pred_history_vectorizer,
                                           model_checkpoint_file=checkpoints_dir + "/latest.h5",
                                           reward_metric=self.reward_metric,
                                           tuning_batch_size=self.tuning_batch_size,
                                           filter_seen=self.filter_seen,
                                           val_users=self.val_users,
                                           gen_limit=self.gen_limit) as trials_generator:
            step = 1 
            while step < self.max_tuning_steps + 1: 
                with tensorboard_writer.as_default(step=step):
                    if (step - 1) % self.validate_every_steps == 0:
                        self.validate(step)
                        tensorboard_writer.flush()

                print("Tuning step", step)
                print("generating...")
                
                batch_rewards, batch_seqs, batch_recs = trials_generator.next_tuning_batch()
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
                    probs = tf.nn.softmax(logits, -1)
                    rec_probs = tf.gather(probs, batch_recs, batch_dims=2) 
                    batch_ratios = rec_probs / tf.stop_gradient(rec_probs)
                    ratio_advantage = gae_advantages * batch_ratios
                    clipped_batch_ratios = tf.clip_by_value(batch_ratios, 1-self.clip_eps, 1+self.clip_eps)
                    clipped_batch_ratio_advantage = gae_advantages * clipped_batch_ratios
                    ppo_loss = -tf.reduce_mean(tf.minimum(ratio_advantage, clipped_batch_ratio_advantage))
                    policy_grads = policy_tape.gradient(ppo_loss, self.model.trainable_variables)
                    policy_optimizer.apply_gradients(zip(policy_grads, self.model.trainable_variables))
                    mean_reward = tf.reduce_mean(tf.reduce_sum(batch_rewards, -1))
                    print(f"Step {step}. Mean reward", mean_reward.numpy(), "ppo loss", ppo_loss.numpy(), "value loss", value_loss.numpy())
                    with tensorboard_writer.as_default(step=step):
                        tf.summary.scalar('tuning_train/ppo_loss', ppo_loss)
                        tf.summary.scalar('tuning_train/mean_reward', mean_reward)
                        tf.summary.scalar('tuning_train/value_loss', value_loss)


                if (step + 1) % self.checkpoint_every_steps == 0:
                    self.model.save_weights(checkpoints_dir + "/current.h5")    
                    os.rename(checkpoints_dir + "/current.h5", checkpoints_dir + "/latest.h5")
                step += 1

            #final validation
            self.validate(step)
            tensorboard_writer.flush()       

        self.model.set_weights(self.best_weights)

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

    def validate(self, step):
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
            val_recommendations, val_seq = static_generate(seq, self.filter_seen, sep_item_id, greedy=True, 
                                                           train=False, items=self.items, gen_limit=self.gen_limit,
                                                            pred_history_vectorizer=self.config.pred_history_vectorizer,
                                                            model=self.model)
            trial_result = build_trial_result(gt_action, val_recommendations, val_seq, self.items, self.reward_metric)                                 
            rewards.append(trial_result.reward)
            recs_gts.append((trial_result.recs_with_scores, trial_result.gt_action))
        mean_reward = tf.reduce_mean(tf.reduce_sum(rewards, -1))
        print(f"Validation at {step}. Mean reward", mean_reward.numpy())
        tf.summary.scalar('tuning_val/mean_reward', mean_reward)
        reward_distr = plot_rewards_per_pos(rewards)
        tf.summary.image("tuning_val/reward_distr", reward_distr, step=step)

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
            trajectory = self.plot_tradeoff_trajectory(tradeoff_name)
            print(f"{metric1.name}::{metric2.name} tradeoff at {step}. {metric1.name}", m1_sum, f"{metric2.name}", m2_sum)
            tf.summary.image(f"tuning_val/{tradeoff_name}_tradeoff", trajectory, step=step)
            metric1_distr = plot_rewards_per_pos(rewards1)
            metric2_distr = plot_rewards_per_pos(rewards2)
            tf.summary.image(f"tuning_val/{metric1.name}_distr", metric1_distr, step=step)
            tf.summary.image(f"tuning_val/{metric2.name}_distr", metric2_distr, step=step)
        self.try_update_best_model(mean_reward, step)

    @plot_to_image
    def plot_tradeoff_trajectory(self, tradeoff_name):
        trajectory = self.tradeoff_trajectiories[tradeoff_name]
        n = len(trajectory)
        indices = list(range(n))  # Original indices
        
        # Ensure the trajectory has at most 1000 points while preserving the first and last point
        if n > 1000:
            step = n // 1000
            trajectory = [trajectory[i] for i in range(0, n, step)]
            indices = [indices[i] for i in range(0, n, step)]
            if trajectory[-1] != trajectory[n-1]:
                trajectory.append(trajectory[n-1])
                indices.append(n-1)
        
        metric1_values = [x[0] for x in trajectory]
        metric2_values = [x[1] for x in trajectory]
        
        # Create a color map from dark green to red
        cmap = plt.cm.RdYlGn_r

        fig, ax = plt.subplots(figsize=(8, 6))

        # Create a scatter plot instead of a line plot
        scatter = ax.scatter(metric1_values, metric2_values, c=indices, cmap=cmap, s=10)

        # Increase size of the last point and mark it in pure red
        ax.scatter(metric1_values[-1], metric2_values[-1], color='red', marker='o', s=100)

        metric1_name = tradeoff_name.split(':::')[0]
        metric2_name = tradeoff_name.split(':::')[1]
        ax.set_xlabel(metric1_name)
        ax.set_ylabel(metric2_name)
        
        # Add a colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Step in trajectory', rotation=270, labelpad=15)
        
        ax.grid()
        return fig
            
        
    def save_best_weights(self):
        self.best_weights = self.model.get_weights()
        
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
        user_recs, user_seq = static_generate(seq, self.filter_seen, sep_item_id, greedy=True,
                                              train=False, items=self.items, gen_limit=self.gen_limit,
                                              pred_history_vectorizer=self.config.pred_history_vectorizer,
                                              model=self.model)
        recs = [] 
        for i in range(len(user_recs)):
            score = 1/np.log2(i+2)
            item_id = self.items.reverse_id(user_recs[i])
            recs.append((item_id, score))
        return recs[:limit]

    def recommend_batch(self, recommendation_requests, limit):
        results = []
        for user_id, features in tqdm.tqdm(recommendation_requests, ascii=True):
            results.append(self.recommend(user_id, limit, features))
        return results

    def get_pred_sequence(self, internal_user_id):
        actions = self.user_actions[internal_user_id]
        sep_item_id = self.items.get_id('<SEP>')
        sequence = [action[1] for action in actions]
        sequence.append(sep_item_id)
        return sequence

    def set_val_users(self, val_users):
        self.pre_train_recommender.set_val_users(val_users)
        super().set_val_users(val_users)