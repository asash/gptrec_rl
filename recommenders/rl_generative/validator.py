from collections import defaultdict
from multiprocessing.context import SpawnProcess
import os
import time
from typing import List
import numpy as np




class Validator(object):
    def __init__(self, model_config, model_checkpoint_path, 
                 items, 
                 users, 
                 val_users,
                 user_actions,
                 pred_history_vectorizer, 
                 tensorboard_dir,
                 filter_seen,
                 reward_metric,
                 gen_limit=10,
                 tradeoff_monitoring_rewards = []): 
        self.model = None
        self.items = items
        self.pred_history_vectorizer = pred_history_vectorizer 
        self.gen_limit = gen_limit
        self.last_update_timestamp = None
        self.model_checkpoint_path = model_checkpoint_path
        self.best_val_revard = float('-inf')
        self.tradeoff_monitoring_rewards = tradeoff_monitoring_rewards
        self.users = users
        self.val_users = val_users
        self.user_actions = user_actions
        self.validation_step = 0
        self.tensorboard_dir = tensorboard_dir 
        self.filter_seen = filter_seen
        self.reward_metric = reward_metric
        self.tensroboard_writer = None
        self.last_checkpoint = None
        self.model_config = model_config
        self.tradeoff_trajectiories = defaultdict(list)

    def ensure_model(self):
        from aprec.recommenders.sequential.models.generative.gpt_rec_rl import RLGPT2RecModel
        if self.model is None: 
            self.model = RLGPT2RecModel.from_config(self.model_config)

    def try_update_weights(self):
        from aprec.recommenders.rl_generative.utils import get_latest_checkpoint
        self.ensure_model()
        latest_checkpoint = get_latest_checkpoint(self.model_checkpoint_path)
        file_timestamp = os.path.getmtime(latest_checkpoint + "/__success__")
        if self.last_update_timestamp is None or file_timestamp > self.last_update_timestamp:
            self.last_update_timestamp = file_timestamp
            self.model.load_weights(latest_checkpoint + "/model.h5")
            self.validation_step = int(latest_checkpoint.split('_')[-1])
            print("Validation model weights updated from step ", self.validation_step)
            self.last_checkpoint = latest_checkpoint
            return True
        return False
        
    def ensure_tensorboard_writer(self):
        import tensorflow as tf
        if self.tensroboard_writer is None:
            self.tensroboard_writer = tf.summary.create_file_writer(self.tensorboard_dir)

    def __call__(self):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        while True:
            if self.try_update_weights():
                print("Validator: Model weights updated")
                self.validate()
            else:
                time.sleep(1)

    def validate(self):
        from aprec.recommenders.rl_generative.generator import static_generate
        from aprec.recommenders.rl_generative.plot_utils import plot_rewards_per_pos
        from aprec.recommenders.rl_generative.utils import build_trial_result, get_seq_with_gt
        from aprec.api.action import Action
        import tensorflow as tf

        self.ensure_tensorboard_writer()
        rewards = []
        recs_gts = []
        cnt = 0
        for user in self.val_users:
            internal_user_id = self.users.get_id(user)
            all_actions = self.user_actions[internal_user_id]
            sep_item_id = self.items.get_id('<SEP>')
            gt_action_index = len(all_actions) - 1
            seq, gt = get_seq_with_gt(all_actions, sep_item_id, gt_action_index)
            gt_action = Action(user, item_id=self.items.reverse_id(gt), timestamp=0) 
            val_recommendations, val_seq, val_logged_probs, val_mask = static_generate(seq, self.filter_seen, sep_item_id, greedy=True, 
                                                           train=False, items=self.items, gen_limit=self.gen_limit,
                                                            pred_history_vectorizer=self.pred_history_vectorizer,
                                                            model=self.model)
            trial_result = build_trial_result(gt_action, val_recommendations, val_seq, self.items, self.reward_metric, val_logged_probs, val_mask)                                 
            rewards.append(trial_result.reward)
            recs_gts.append((trial_result.recs_with_scores, trial_result.gt_action))
            cnt += 1
            if cnt % 10 == 0:
                print(f"Validator: validation at {self.validation_step}. Processed {cnt} users")
                
        mean_reward = tf.reduce_mean(tf.reduce_sum(rewards, -1))
        print(f"Validator: validation at {self.validation_step}. Mean reward", mean_reward.numpy())
        with self.tensroboard_writer.as_default(step=self.validation_step):
            tf.summary.scalar('tuning_val/mean_reward', mean_reward)
            reward_distr = plot_rewards_per_pos(rewards)
            tf.summary.image("tuning_val/reward_distr", reward_distr, step=self.validation_step)
            self.plot_tradeoffs(recs_gts)
        validation_file = self.model_checkpoint_path + "/validations.csv"
        with open(validation_file, "a") as f:
            f.write(f"{self.last_checkpoint},{self.validation_step},{mean_reward.numpy()}\n")
            f.flush()
            print(f"Validator: Validation file {validation_file} updated")

    def plot_tradeoffs(self, recs_gts):
        from aprec.recommenders.rl_generative.plot_utils import plot_tradeoff_trajectory
        from aprec.recommenders.rl_generative.plot_utils import plot_rewards_per_pos
        import tensorflow as tf
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
            print(f"{metric1.name}::{metric2.name} tradeoff at {self.validation_step}. {metric1.name}", m1_sum, f"{metric2.name}", m2_sum)
            tf.summary.image(f"tuning_val/{tradeoff_name}_tradeoff", trajectory, step=self.validation_step)
            metric1_distr = plot_rewards_per_pos(rewards1)
            metric2_distr = plot_rewards_per_pos(rewards2)
            tf.summary.image(f"tuning_val/{metric1.name}_distr", metric1_distr, step=self.validation_step)
            tf.summary.image(f"tuning_val/{metric2.name}_distr", metric2_distr, step=self.validation_step)
 
class ValidatorProcess(object):
    def __init__(self, *args, **kwargs) -> None:
        self.validator = Validator(*args, **kwargs)

    def __enter__(self):
        self.validator_process = SpawnProcess(target=self.validator)
        self.validator_process.daemon = True
        self.validator_process.start()
        return self
            

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.validator_process.terminate()
        self.validator_process.join()

