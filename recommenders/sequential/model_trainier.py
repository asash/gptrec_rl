from __future__ import annotations
from typing import TYPE_CHECKING

from collections import defaultdict
import random
import time
from typing import List
import tensorflow as tf
from tqdm import tqdm
from aprec.api.action import Action
from aprec.recommenders.sequential.data_generator.data_generator import DataGenerator, DataGeneratorAsyncFactory, MemmapDataGenerator

if TYPE_CHECKING:
    from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender



class ValidationResult(object):
    def __init__(self, val_loss, val_metric, extra_val_metrics, train_metric, extra_train_metrics, train_loss=None) -> None:
        self.val_loss = val_loss
        self.val_metric = val_metric
        self.extra_val_metrics = extra_val_metrics
        self.train_metric = train_metric
        self.extra_train_metrics = extra_train_metrics
        self.train_loss = train_loss

    def set_train_loss(self, train_loss):
        self.train_loss = train_loss

class ModelTrainer(object):
    def __init__(self, recommender: SequentialRecommender):
        self.recommender = recommender
        tensorboard_dir = self.recommender.get_tensorboard_dir()
        print(f"writing tensorboard logs to {tensorboard_dir}")
        self.tensorboard_writer = tf.summary.create_file_writer(tensorboard_dir)
        self.train_users, self.val_users = self.train_val_split()
        
        self.val_recommendation_requets = [(user_id, None) for user_id in self.recommender.val_users]
        self.val_seen, self.val_ground_truth = self.leave_one_out(self.recommender.val_users)

        self.train_users_pool = list(self.recommender.users.straight.keys() - set(self.recommender.val_users))
        self.train_users_sample = random.choices(self.train_users_pool, k=len(self.val_users)) #these users will be used for calculating metric on train
        self.train_sample_recommendation_requests = [(user_id, None) for user_id in self.train_users_sample]
        self.train_sample_seen, self.train_sample_ground_truth = self.leave_one_out(self.train_users_sample)

        self.targets_builder = self.recommender.config.targets_builder()
        self.targets_builder.set_n_items(self.recommender.items.size())
        self.targets_builder.set_train_sequences(self.train_users)
        print("train_users: {}, val_users:{}, items:{}".format(len(self.train_users), len(self.val_users), self.recommender.items.size()))
        self.model_is_compiled = False
        self.recommender.model = self.recommender.get_model()
        self.recommender.model.fit_biases(self.train_users)
        
        if self.recommender.config.val_metric.less_is_better:
            self.best_metric_val = float('inf')
        else:
            self.best_metric_val = float('-inf')

        self.best_val_loss = float('inf')

        self.steps_metric_not_improved = 0
        self.steps_loss_not_improved = 0
        self.best_epoch = -1
        self.best_weights = self.recommender.model.get_weights()
        self.val_generator = self.get_val_generator()
        self.early_stop_flag = False
        self.current_epoch = None 
        self.history: List[ValidationResult] = []

    def train(self):
        self.start_time = time.time()
        with self.get_train_generator_factory() as train_data_generator_factory:
            for epoch_num in range(self.recommender.config.train_epochs):
                print(f"epoch {epoch_num}")
                self.current_epoch = epoch_num
                train_generator = train_data_generator_factory.next_generator()
                self.epoch(train_generator)
                train_generator.cleanup()
                if self.early_stop_flag:
                    break
            self.recommender.model.set_weights(self.best_weights)
            print(f"taken best model from epoch{self.best_epoch}. best_val_{self.recommender.config.val_metric.name}: {self.best_metric_val}")
    
    def get_train_generator_factory(self):
        return DataGeneratorAsyncFactory(self.recommender.config,
                                      self.train_users,
                                      self.recommender.items.size(),
                                      targets_builder=self.targets_builder, 
                                      shuffle_data=True)
    def get_val_generator(self):
        return DataGenerator(self.recommender.config, self.val_users, 
                                      self.recommender.items.size(),
                                      targets_builder=self.targets_builder,
                                      shuffle_data=False)


    def epoch(self, generator: MemmapDataGenerator):
        train_loss = self.train_epoch(generator)
        validation_result = self.validate()
        validation_result.set_train_loss(train_loss)
        self.history.append(validation_result)
        
        self.try_update_best_val_metric(validation_result.val_metric)
        self.try_update_best_val_loss(validation_result.val_loss)
        self.try_early_stop()
        self.log(validation_result)
        
    def training_time(self):
        return time.time() - self.start_time

    def try_early_stop(self):
        self.steps_to_early_stop = self.recommender.config.early_stop_epochs - min(self.steps_loss_not_improved, self.steps_metric_not_improved)
        if self.steps_to_early_stop <= 0:
            print(f"early stopped at epoch {self.current_epoch}")
            self.early_stop_flag = True

        if self.recommender.config.training_time_limit is not None and self.training_time() > self.recommender.config.training_time_limit:
            print(f"time limit stop triggered at epoch {self.current_epoch}")
            self.early_stop_flag = True

    def try_update_best_val_loss(self, val_loss):
        self.steps_loss_not_improved += 1
        if (val_loss < self.best_val_loss):
            self.best_val_loss = val_loss
            self.steps_loss_not_improved = 0

    def try_update_best_val_metric(self, val_metric):
        self.steps_metric_not_improved += 1
        if (self.recommender.config.val_metric.less_is_better and val_metric < self.best_metric_val) or\
                            (not self.recommender.config.val_metric.less_is_better and val_metric > self.best_metric_val):
            self.steps_metric_not_improved = 0
            self.best_metric_val = val_metric
            self.best_epoch = self.current_epoch
            self.best_weights = self.recommender.model.get_weights()

    def log(self, validation_result: ValidationResult):
        config = self.recommender.config
        print(f"\tval_{config.val_metric.name}: {validation_result.val_metric:.5f}")
        print(f"\tbest_{config.val_metric.name}: {self.best_metric_val:.5f}")
        print(f"\ttrain_{config.val_metric.name}: {validation_result.train_metric:.5f}")
        print(f"\ttrain_loss: {validation_result.train_loss}")
        print(f"\tval_loss: {validation_result.val_loss}")
        print(f"\tbest_val_loss: {self.best_val_loss}")
        print(f"\tsteps_metric_not_improved: {self.steps_metric_not_improved}")
        print(f"\tsteps_loss_not_improved: {self.steps_loss_not_improved}")
        print(f"\tsteps_to_stop: {self.steps_to_early_stop}")
        print(f"\ttotal_training_time: {self.training_time()}")
        with self.tensorboard_writer.as_default(step=(self.current_epoch + 1)*config.max_batches_per_epoch*config.batch_size):
            tf.summary.scalar(f"{config.val_metric.name}/val", validation_result.val_metric)
            tf.summary.scalar(f"{config.val_metric.name}/train", validation_result.train_metric)
            tf.summary.scalar(f"{config.val_metric.name}/train_val_diff", validation_result.train_metric - validation_result.val_metric)
            tf.summary.scalar(f"{config.val_metric.name}/best_val", self.best_metric_val)
            tf.summary.scalar(f"{config.val_metric.name}/steps_metric_not_improved", self.steps_metric_not_improved)
            tf.summary.scalar(f"loss/train", validation_result.train_loss)
            tf.summary.scalar(f"loss/val", validation_result.val_loss)
            tf.summary.scalar(f"loss/train_val_diff", (validation_result.train_loss - validation_result.val_loss))
            tf.summary.scalar(f"loss/evaluations_without_improvement", self.steps_loss_not_improved)
            tf.summary.scalar(f"steps_to_early_stop", self.steps_to_early_stop)
            for metric in config.extra_val_metrics:
                tf.summary.scalar(f"{metric.get_name()}/train", validation_result.extra_train_metrics[metric.get_name()])
                tf.summary.scalar(f"{metric.get_name()}/val", validation_result.extra_val_metrics[metric.get_name()])
                tf.summary.scalar(f"{metric.get_name()}/train_val_diff", validation_result.extra_train_metrics[metric.get_name()]
                                                                    - validation_result.extra_val_metrics[metric.get_name()])
            self.recommender.model.log()
        
    def leave_one_out(self, users):
        seen_result = []
        result = []
        for user_id in users:
            user_actions = self.recommender.user_actions[self.recommender.users.get_id(user_id)]
            seen_items = {self.recommender.items.reverse_id(action[1]) for action in user_actions[:-1]}
            seen_result.append(seen_items)
            last_action = user_actions[-1] 
            user_result = Action(user_id=user_id, item_id=self.recommender.items.reverse_id(last_action[1]), timestamp=last_action[0])
            result.append([user_result])
        return seen_result, result

    def train_val_split(self):
        val_user_ids = [self.recommender.users.get_id(val_user) for val_user in self.recommender.val_users]
        train_user_ids = list(range(self.recommender.users.size()))
        val_users = self.recommender.user_actions_by_id_list(val_user_ids)
        train_users = self.recommender.user_actions_by_id_list(train_user_ids, val_user_ids)
        return train_users, val_users


    def train_epoch(self, generator):
        if self.recommender.config.use_keras_training:
            return self.train_keras(generator)
        else:
            return self.train_eager(generator)

    def train_keras(self, generator):
        if not self.model_is_compiled:
            self.recommender.model.compile(self.recommender.config.optimizer, self.recommender.config.loss)
            self.model_is_compiled = True
        summary =  self.recommender.model.fit(generator, steps_per_epoch=self.recommender.config.max_batches_per_epoch)
        return summary.history['loss'][0]
        
        
    def train_eager(self, generator):
        pbar = tqdm(generator, ascii=True, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', position=0, leave=True, ncols=70)
        variables = self.recommender.model.variables
        loss_sum = 0
        num_batches = 0
        for X, y_true in pbar:
            num_batches += 1
            with tf.GradientTape() as tape:
                tape.watch(variables)
                y_pred = self.recommender.model(X, training=True)
                loss_val = tf.reduce_mean(self.config.loss(y_true, y_pred))
                pass
            grad = tape.gradient(loss_val, variables)
            self.recommender.config.optimizer.apply_gradients(zip(grad, variables))
            loss_sum += loss_val
            pbar.set_description(f"loss: {loss_sum/num_batches:.5f}")
        pbar.close()
        train_loss = loss_sum/num_batches
        return train_loss

    def validate(self):
        self.val_generator.reset()
        print("validating..")
        pbar = tqdm(self.val_generator, ascii=True, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', position=0, leave=True, ncols=70)
        val_loss_sum = 0 
        num_val_batches = 0
        for X, y_true in pbar:
            num_val_batches += 1
            y_pred = self.recommender.model(X, training=True)
            loss_val = tf.reduce_mean(self.recommender.config.loss(y_true, y_pred))
            val_loss_sum += loss_val
        val_loss = val_loss_sum / num_val_batches 

        val_metric, extra_val_metrics = self.get_val_metrics(self.val_recommendation_requets, self.val_seen, self.val_ground_truth, callbacks=True)
        train_metric, extra_train_metrics = self.get_val_metrics(self.train_sample_recommendation_requests, self.train_sample_seen, self.train_sample_ground_truth)
        return ValidationResult(val_loss, val_metric, extra_val_metrics, train_metric, extra_train_metrics)

    def get_val_metrics(self, recommendation_requests, seen_items, ground_truth, callbacks=False):
        extra_recs = 0
        if self.recommender.flags.get('filter_seen', False):
            extra_recs += self.recommender.config.sequence_length        

        recs = self.recommender.recommend_batch(recommendation_requests, self.recommender.config.val_rec_limit + extra_recs, is_val=True)
        metric_sum = 0.0
        extra_metric_sums = defaultdict(lambda: 0.0)
        callback_recs, callback_truth = [], [] 
        for rec, seen, truth in zip(recs, seen_items, ground_truth):
            if self.recommender.flags.get('filter_seen', False):
                filtered_rec = [recommended_item for recommended_item in rec if recommended_item[0] not in seen]
                callback_recs.append(filtered_rec)
                callback_truth.append(truth)
                metric_sum += self.recommender.config.val_metric(filtered_rec, truth) 
                for extra_metric in self.recommender.config.extra_val_metrics:
                    extra_metric_sums[extra_metric.get_name()] += extra_metric(filtered_rec, truth)
            else:
                callback_recs.append(rec)
                callback_truth.append(truth)
                metric_sum += self.recommender.config.val_metric(rec, truth) 
        if callbacks:
            for callback in self.recommender.config.val_callbacks:
                callback(callback_recs, callback_truth)
                
        val_metric = metric_sum / len(recs)
        extra_metrics = {}
        for extra_metric in self.recommender.config.extra_val_metrics:
            extra_metrics[extra_metric.get_name()] = extra_metric_sums[extra_metric.get_name()] / len(recs)
        return val_metric, extra_metrics



