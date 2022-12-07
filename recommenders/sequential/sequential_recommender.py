import gc
from pathlib import Path
import random
import tempfile
import time
from collections import defaultdict
import joblib
import tensorflow as tf
import dill

from tqdm.auto import tqdm
from aprec.api.action import Action
from aprec.api.items_ranking_request import ItemsRankingRequest
from aprec.recommenders.sequential.models.sequential_recsys_model import SequentialDataParameters, get_sequential_model
from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig

from aprec.utils.item_id import ItemId
from aprec.recommenders.recommender import Recommender
from aprec.recommenders.sequential.data_generator.data_generator import DataGenerator, DataGeneratorAsyncFactory
import tensorflow as tf
import faiss

class SequentialRecommender(Recommender):
    def __init__(self, config: SequentialRecommenderConfig):
        super().__init__()
        self.config = config
        self.users = ItemId()
        self.items = ItemId()
        self.user_actions = defaultdict(list)
        self.metadata = {}
        #we use following two dicts for sampled metrics
        self.item_ranking_requrests = {}
        self.item_ranking_results = {}
        self.model_is_compiled = False

    def add_action(self, action):
        user_id_internal = self.users.get_id(action.user_id)
        action_id_internal = self.items.get_id(action.item_id)
        self.user_actions[user_id_internal].append((action.timestamp, action_id_internal))

    # exclude last action for val_users
    def user_actions_by_id_list(self, id_list, val_user_ids=None):
        val_users = set()
        if val_user_ids is not None:
            val_users = set(val_user_ids)
        result = []
        for user_id in id_list:
            if user_id not in val_users:
                result.append(self.user_actions[user_id])
            else:
                result.append(self.user_actions[user_id][:-1])
        return result

    def sort_actions(self):
        for user_id in self.user_actions:
            self.user_actions[user_id].sort()
            
    def leave_one_out(self, users):
        seen_result = []
        result = []
        for user_id in users:
            user_actions = self.user_actions[self.users.get_id(user_id)]
            seen_items = {self.items.reverse_id(action[1]) for action in user_actions[:-1]}
            seen_result.append(seen_items)
            last_action = user_actions[-1] 
            user_result = Action(user_id=user_id, item_id=self.items.reverse_id(last_action[1]), timestamp=last_action[0])
            result.append([user_result])
        return seen_result, result

    def rebuild_model(self):
        tensorboard_dir = self.get_tensorboard_dir()
        print(f"writing tensorboard logs to {tensorboard_dir}")
        tensorboard_writer = tf.summary.create_file_writer(tensorboard_dir)
        self.sort_actions()
        self.pass_parameters()
        train_users, val_users = self.train_val_split()
        
        self.val_recommendation_requets = [(user_id, None) for user_id in self.val_users]
        self.val_seen, self.val_ground_truth = self.leave_one_out(self.val_users)

        train_users_pool = list(self.users.straight.keys() - set(self.val_users))
        train_users_sample = random.choices(train_users_pool, k=len(self.val_users)) #these users will be used for calculating metric on train
        self.train_sample_recommendation_requests = [(user_id, None) for user_id in train_users_sample]
        self.train_sample_seen, self.train_sample_ground_truth = self.leave_one_out(train_users_sample)

        targets_builder = self.config.targets_builder()
        targets_builder.set_n_items(self.items.size())
        targets_builder.set_train_sequences(train_users)
        print("train_users: {}, val_users:{}, items:{}".format(len(train_users), len(self.val_users), self.items.size()))
        self.model = self.get_model()
        if hasattr(self.model, 'fit_biases'):
            self.model.fit_biases(train_users)
        
        if self.config.val_metric.less_is_better:
            best_metric_val = float('inf')
        else:
            best_metric_val = float('-inf')

        best_val_loss = float('inf')

        steps_metric_not_improved = 0
        steps_loss_not_improved = 0
        best_epoch = -1
        best_weights = self.model.get_weights()
        val_metric_history = []
        start_time = time.time()
        val_generator = DataGenerator(self.config, val_users, 
                                      self.items.size(),
                                      targets_builder=targets_builder,
                                      shuffle_data=False)

        data_generator_async_factory = DataGeneratorAsyncFactory(self.config,
                                      train_users,
                                      self.items.size(),
                                      targets_builder=targets_builder, 
                                      shuffle_data=True)

        early_stop_flag = False
        for epoch in range(self.config.train_epochs):
            if early_stop_flag:
                break
            generator = data_generator_async_factory.next_generator() 
            print(f"epoch: {epoch}")
            train_loss = self.train_epoch(generator)
            train_metric, extra_train_metrics, val_loss ,val_metric, extra_val_metrics = self.validate(val_generator)

            total_trainig_time = time.time() - start_time
            val_metric_history.append((total_trainig_time, val_metric))

            steps_metric_not_improved += 1
            if (self.config.val_metric.less_is_better and val_metric < best_metric_val) or\
                        (not self.config.val_metric.less_is_better and val_metric > best_metric_val):
                steps_metric_not_improved = 0
                best_metric_val = val_metric
                best_epoch = epoch
                best_weights = self.model.get_weights()
            steps_loss_not_improved += 1
            if (val_loss < best_val_loss):
                best_val_loss = val_loss
                steps_loss_not_improved = 0
            steps_to_early_stop = self.config.early_stop_epochs - min(steps_loss_not_improved, steps_metric_not_improved)
            print(f"\tval_{self.config.val_metric.name}: {val_metric:.5f}")
            print(f"\tbest_{self.config.val_metric.name}: {best_metric_val:.5f}")
            print(f"\ttrain_{self.config.val_metric.name}: {train_metric:.5f}")
            print(f"\ttrain_loss: {train_loss}")
            print(f"\tval_loss: {val_loss}")
            print(f"\tbest_val_loss: {best_val_loss}")
            print(f"\tsteps_metric_not_improved: {steps_metric_not_improved}")
            print(f"\tsteps_loss_not_improved: {steps_loss_not_improved}")
            print(f"\tsteps_to_stop: {steps_to_early_stop}")
            print(f"\ttotal_training_time: {total_trainig_time}")
            with tensorboard_writer.as_default(step=(epoch + 1)*self.config.max_batches_per_epoch*self.config.batch_size):
                tf.summary.scalar(f"{self.config.val_metric.name}/val", val_metric)
                tf.summary.scalar(f"{self.config.val_metric.name}/train", train_metric)
                tf.summary.scalar(f"{self.config.val_metric.name}/train_val_diff", train_metric - val_metric)
                tf.summary.scalar(f"{self.config.val_metric.name}/best_val", best_metric_val)
                tf.summary.scalar(f"{self.config.val_metric.name}/steps_metric_not_improved", steps_metric_not_improved)
                tf.summary.scalar(f"loss/train", train_loss)
                tf.summary.scalar(f"loss/val", val_loss)
                tf.summary.scalar(f"loss/train_val_diff", train_loss - val_loss)
                tf.summary.scalar(f"loss/evaluations_without_improvement", steps_loss_not_improved)
                tf.summary.scalar(f"steps_to_early_stop", steps_to_early_stop)
                for metric in self.config.extra_val_metrics:
                    tf.summary.scalar(f"{metric.get_name()}/train", extra_train_metrics[metric.name])
                    tf.summary.scalar(f"{metric.get_name()}/val", extra_val_metrics[metric.name])
                    tf.summary.scalar(f"{metric.get_name()}/train_val_diff", extra_train_metrics[metric.name] - extra_val_metrics[metric.name])
                if hasattr(self.model, 'log'):
                    self.model.log()

            if steps_to_early_stop <= 0:
                print(f"early stopped at epoch {epoch}")
                early_stop_flag = True
            if self.config.training_time_limit is not None and total_trainig_time > self.config.training_time_limit:
                print(f"time limit stop triggered at epoch {epoch}")
                early_stop_flag = True
            generator.cleanup()

        data_generator_async_factory.close()
        self.model.set_weights(best_weights)
        print(f"taken best model from epoch{best_epoch}. best_val_{self.config.val_metric.name}: {best_metric_val}")
        if self.config.use_ann_for_inference:
            self.build_ann_index()
        gc.collect()
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
    
    def build_ann_index(self):
        embedding_matrix = self.model.get_embedding_matrix().numpy()
        self.index = faiss.IndexFlatIP(embedding_matrix.shape[-1])
        self.index.add(embedding_matrix)
        pass
         
    def pass_parameters(self):
        self.config.loss.set_num_items(self.items.size())
        self.config.train_history_vectorizer.set_sequence_len(self.config.sequence_length)
        self.config.train_history_vectorizer.set_padding_value(self.items.size())
        self.config.pred_history_vectorizer.set_sequence_len(self.config.sequence_length)
        self.config.pred_history_vectorizer.set_padding_value(self.items.size())
        
    def add_test_items_ranking_request(self, request: ItemsRankingRequest):
        self.item_ranking_requrests[request.user_id] = request 

    def train_epoch(self, generator):
        if self.config.use_keras_training:
            return self.train_keras(generator)
        else:
            return self.train_eager(generator)

    def train_keras(self, generator):
        if not self.model_is_compiled:
            self.model.compile(self.config.optimizer, self.config.loss)
            self.model_is_compiled = True
        summary =  self.model.fit(generator, steps_per_epoch=self.config.max_batches_per_epoch)
        return summary.history['loss'][0]
        
        
    def train_eager(self, generator):
        pbar = tqdm(generator, ascii=True, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', position=0, leave=True, ncols=70)
        variables = self.model.variables
        loss_sum = 0
        num_batches = 0
        for X, y_true in pbar:
            num_batches += 1
            with tf.GradientTape() as tape:
                tape.watch(variables)
                y_pred = self.model(X, training=True)
                loss_val = tf.reduce_mean(self.config.loss(y_true, y_pred))
                pass
            grad = tape.gradient(loss_val, variables)
            self.config.optimizer.apply_gradients(zip(grad, variables))
            loss_sum += loss_val
            pbar.set_description(f"loss: {loss_sum/num_batches:.5f}")
        pbar.close()
        train_loss = loss_sum/num_batches
        return train_loss

    def validate(self, val_generator):
        val_generator.reset()
        print("validating..")
        pbar = tqdm(val_generator, ascii=True, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', position=0, leave=True, ncols=70)
        val_loss_sum = 0 
        num_val_batches = 0
        for X, y_true in pbar:
            num_val_batches += 1
            y_pred = self.model(X, training=True)
            loss_val = tf.reduce_mean(self.config.loss(y_true, y_pred))
            val_loss_sum += loss_val
        val_loss = val_loss_sum / num_val_batches 

        val_metric, extra_val_metrics = self.get_metric(self.val_recommendation_requets, self.val_seen, self.val_ground_truth, callbacks=True)
        train_metric, extra_train_metrics = self.get_metric(self.train_sample_recommendation_requests, self.train_sample_seen, self.train_sample_ground_truth)

        return train_metric, extra_train_metrics, val_loss, val_metric, extra_val_metrics

    def get_metric(self, recommendation_requests, seen_items, ground_truth, callbacks=False):
        extra_recs = 0
        if self.flags.get('filter_seen', False):
            extra_recs += self.config.sequence_length        

        recs = self.recommend_batch(recommendation_requests, self.config.val_rec_limit + extra_recs, is_val=True)
        metric_sum = 0.0
        extra_metric_sums = defaultdict(lambda: 0.0)
        callback_recs, callback_truth = [], [] 
        for rec, seen, truth in zip(recs, seen_items, ground_truth):
            if self.flags.get('filter_seen', False):
                filtered_rec = [recommended_item for recommended_item in rec if recommended_item[0] not in seen]
                callback_recs.append(filtered_rec)
                callback_truth.append(truth)
                metric_sum += self.config.val_metric(filtered_rec, truth) 
                for extra_metric in self.config.extra_val_metrics:
                    extra_metric_sums[extra_metric.get_name()] += extra_metric(filtered_rec, truth)
            else:
                callback_recs.append(rec)
                callback_truth.append(truth)
                metric_sum += self.config.val_metric(rec, truth) 

        for callback in self.config.val_callbacks:
            callback(callback_recs, callback_truth)
            
        val_metric = metric_sum / len(recs)
        extra_metrics = {}
        for extra_metric in self.config.extra_val_metrics:
            extra_metrics[extra_metric.get_name()] = extra_metric_sums[extra_metric.get_name()] / len(recs)
        return val_metric, extra_metrics

    def train_val_split(self):
        val_user_ids = [self.users.get_id(val_user) for val_user in self.val_users]
        train_user_ids = list(range(self.users.size()))
        val_users = self.user_actions_by_id_list(val_user_ids)
        train_users = self.user_actions_by_id_list(train_user_ids, val_user_ids)
        return train_users, val_users

    def get_model(self):
        data_params = SequentialDataParameters(num_items=self.items.size(),
                                               num_users=self.users.size(),
                                               batch_size=self.config.batch_size, 
                                               sequence_length=self.config.sequence_length)
        return get_sequential_model(self.config.model_config, data_params)

    def recommend(self, user_id, limit, features=None):
        if self.config.use_ann_for_inference:
            model_inputs = self.get_model_inputs(user_id) 
            user_emb = self.model.get_sequence_embeddings([model_inputs]).numpy()
            scores, items = self.index.search(user_emb, limit)
            result = [(self.items.reverse_id(items[0][i]), scores[0][i]) for i in range(len(items[0]))]
        else:    
            scores = self.get_all_item_scores(user_id)
            if user_id in self.item_ranking_requrests:
                self.process_item_ranking_request(user_id, scores)
            best_ids = tf.nn.top_k(scores, limit).indices.numpy()
            result = [(self.items.reverse_id(id), scores[id]) for id in best_ids]
        return result

    def get_item_rankings(self):
        for user_id in self.items_ranking_requests:
            self.process_item_ranking_request(user_id)
        return self.item_ranking_results

    def process_item_ranking_request(self,  user_id, scores=None):
        if (user_id not in self.item_ranking_requrests) or  (user_id in self.item_ranking_results):
            return
        if scores is None:
            scores = self.get_all_item_scores(user_id)
        request = self.item_ranking_requrests[user_id]
        user_result = []
        for item_id in request.item_ids:
            if (self.items.has_item(item_id)) and (self.items.get_id(item_id) < len(scores)):
                user_result.append((item_id, float(scores[self.items.get_id(item_id)])))
            else:
                user_result.append((item_id, float("-inf")))
        user_result.sort(key = lambda x: -x[1])
        self.item_ranking_results[user_id] = user_result
    
    def get_model_inputs(self, user_id, is_val=False):
        if not is_val:
            actions = self.user_actions[self.users.get_id(user_id)]
        else:
            actions = self.user_actions[self.users.get_id(user_id)][:-1]
        items_list = [action[1] for action in actions]
        model_actions = [(0, action) for action in items_list]
        session = self.config.pred_history_vectorizer(model_actions)
        session = session.reshape(1, self.config.sequence_length)
        model_inputs = [session]
        return model_inputs
    
    def recommend_multiple(self, recommendation_requets, limit, is_val=False):
        user_ids = [user_id for user_id, features in recommendation_requets]
        model_inputs = list(map(lambda id: self.get_model_inputs(id, is_val)[0], user_ids))
        model_inputs = tf.concat(model_inputs, 0)
        result = []
        if is_val or not(self.config.use_ann_for_inference):
            scoring_func = self.get_scoring_func()
            predictions = scoring_func([model_inputs])
            list(map(self.process_item_ranking_request, user_ids, predictions))
            best_predictions = tf.math.top_k(predictions, k=limit)
            ind = best_predictions.indices.numpy()
            vals = best_predictions.values.numpy()
        else:
            embs =  self.model.get_sequence_embeddings([model_inputs]).numpy()
            vals, ind = self.index.search(embs, limit)
        for i in range(len(user_ids)):
            result.append(list(zip(self.decode_item_ids(ind[i]), vals[i])))
        return result
    
    def get_tensorboard_dir(self):
        if self.tensorboard_dir is not None:
            return self.tensorboard_dir
        else:
            return tempfile.mkdtemp()

    def decode_item_ids(self, ids):
        result = []
        for id in ids:
            result.append(self.items.reverse_id(int(id)))
        return result

    def recommend_batch(self, recommendation_requests, limit, is_val=False):
        results = []
        start = 0
        end = min(start + self.config.eval_batch_size, len(recommendation_requests))
        print("generating recommendation in batches...")
        pbar = tqdm(total = len(recommendation_requests), ascii=True, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',  position=0, leave=True, ncols=70)
        while (start < end):
            req = recommendation_requests[start:end]
            results += self.recommend_multiple(req, limit, is_val)
            pbar.update(end - start)
            start = end  
            end = min(start + self.config.eval_batch_size, len(recommendation_requests))
        return results

    def get_scoring_func(self):
        if hasattr(self.model, 'score_all_items'):
            return self.model.score_all_items
        else: 
            return self.model
    
    def get_all_item_scores(self, user_id):
        model_inputs = self.get_model_inputs(user_id) 
        scoring_func = self.get_scoring_func()
        return scoring_func(model_inputs)[0].numpy()
    