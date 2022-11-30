import gc
import random
import tempfile
import time
from collections import defaultdict
import tensorflow as tf

from tqdm.auto import tqdm
from aprec.api.action import Action
from aprec.api.items_ranking_request import ItemsRankingRequest
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.recommenders.dnn_sequential_recommender.history_vectorizers.default_history_vectorizer import DefaultHistoryVectrizer
from aprec.recommenders.dnn_sequential_recommender.history_vectorizers.history_vectorizer import HistoryVectorizer
from aprec.recommenders.dnn_sequential_recommender.target_builders.full_matrix_targets_builder import FullMatrixTargetsBuilder
from aprec.recommenders.dnn_sequential_recommender.target_builders.target_builders import TargetBuilder
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.last_item_splitter import SequenceContinuation
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.random_fraction_splitter import RandomFractionSplitter
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.targetsplitter import TargetSplitter

from aprec.utils.item_id import ItemId
from aprec.recommenders.metrics.ndcg import KerasNDCG
from aprec.recommenders.recommender import Recommender
from aprec.recommenders.dnn_sequential_recommender.data_generator.data_generator import DataGenerator, DataGeneratorAsyncFactory
from aprec.recommenders.dnn_sequential_recommender.models.sequential_recsys_model import SequentialRecsysModel
from aprec.losses.loss import Loss
from aprec.losses.bce import BCELoss
import tensorflow.keras.backend as K
import tensorflow as tf
import faiss

from tensorflow.keras.optimizers import Adam

class DNNSequentialRecommender(Recommender):
    def __init__(self, model_arch: SequentialRecsysModel, loss: Loss = BCELoss(),
                 train_epochs=300, optimizer=Adam(),
                 sequence_splitter:TargetSplitter = RandomFractionSplitter, 
                 targets_builder: TargetBuilder = FullMatrixTargetsBuilder,
                 batch_size=1000, early_stop_epochs=100, target_decay=1.0,
                 training_time_limit=None,
                 train_history_vectorizer:HistoryVectorizer = DefaultHistoryVectrizer(), 
                 pred_history_vectorizer:HistoryVectorizer = DefaultHistoryVectrizer(),
                 data_generator_processes = 8, 
                 data_generator_queue_size = 16,
                 max_batches_per_epoch=10,
                 eval_batch_size = 1024, 
                 val_rec_limit=40,
                 val_metric = NDCG(40),
                 use_ann_for_inference = False):
        super().__init__()
        self.model_arch = model_arch
        self.users = ItemId()
        self.items = ItemId()
        self.user_actions = defaultdict(list)
        self.model = None
        self.user_vectors = None
        self.matrix = None
        self.mean_user = None
        self.train_epochs = train_epochs
        self.loss = loss
        self.early_stop_epochs = early_stop_epochs
        self.optimizer = optimizer
        self.metadata = {}
        self.batch_size = batch_size
        self.target_decay = target_decay
        self.val_users = None
        self.training_time_limit = training_time_limit
        self.users_with_actions = set()
        self.sequence_splitter = sequence_splitter
        self.targets_builder = targets_builder()
        self.train_history_vectorizer = train_history_vectorizer
        self.pred_history_vectorizer = pred_history_vectorizer
        self.data_generator_processes = data_generator_processes
        self.data_generator_queue_size = data_generator_queue_size
        self.max_batches_per_epoch = max_batches_per_epoch
        self.eval_batch_size=eval_batch_size

        #we use following two dicts for sampled metrics
        self.item_ranking_requrests = {}
        self.item_ranking_results = {}
        self.val_rec_limit = val_rec_limit
        self.val_metric = val_metric
        self.use_ann_for_inference = use_ann_for_inference

    def get_metadata(self):
        return self.metadata

    def set_loss(self, loss):
        self.loss = loss

    def name(self):
        return self.model

    def add_action(self, action):
        user_id_internal = self.users.get_id(action.user_id)
        action_id_internal = self.items.get_id(action.item_id)
        self.users_with_actions.add(user_id_internal)
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
        self.tensorboard_writer = tf.summary.create_file_writer(tensorboard_dir)
        self.sort_actions()
        self.pass_parameters()
        train_users, val_users = self.train_val_split()
        self.val_recommendation_requets = [(user_id, None) for user_id in self.val_users]
        self.val_seen, self.val_ground_truth = self.leave_one_out(self.val_users)

        train_users_pool = list(self.users.straight.keys() - set(self.val_users))
        train_users_sample = random.choices(train_users_pool, k=len(self.val_users)) #these users will be used for calculating metric on train
        self.train_sample_recommendation_requests = [(user_id, None) for user_id in train_users_sample]
        self.train_sample_seen, self.train_sample_ground_truth = self.leave_one_out(train_users_sample)

        self.targets_builder.set_n_items(self.items.size())
        self.targets_builder.set_train_sequences(train_users)
        print("train_users: {}, val_users:{}, items:{}".format(len(train_users), len(self.val_users), self.items.size()))
        self.model = self.get_model()
        if self.val_metric.less_is_better:
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
        val_generator = DataGenerator(val_users, 
                                      self.model_arch.max_history_length,
                                      self.items.size(),
                                      self.train_history_vectorizer,
                                      batch_size=self.batch_size,
                                      sequence_splitter=self.sequence_splitter, 
                                      targets_builder=self.targets_builder,
                                      shuffle_data=False)

        data_generator_async_factory = DataGeneratorAsyncFactory(self.data_generator_processes, self.data_generator_queue_size,
                                      train_users,
                                      self.model_arch.max_history_length,
                                      self.items.size(),
                                      self.train_history_vectorizer,
                                      batch_size=self.batch_size,
                                      sequence_splitter=self.sequence_splitter,
                                      targets_builder=self.targets_builder, 
                                      shuffle_data=True,
                                      max_batches_per_epoch = self.max_batches_per_epoch
                                      )


        for epoch in range(self.train_epochs):
            generator = data_generator_async_factory.next_generator() 
            print(f"epoch: {epoch}")
            train_loss, train_metric, val_loss ,val_metric = self.train_epoch(generator, val_generator)

            total_trainig_time = time.time() - start_time
            val_metric_history.append((total_trainig_time, val_metric))

            steps_metric_not_improved += 1
            if (self.val_metric.less_is_better and val_metric < best_metric_val) or\
                        (not self.val_metric.less_is_better and val_metric > best_metric_val):
                steps_metric_not_improved = 0
                best_metric_val = val_metric
                best_epoch = epoch
                best_weights = self.model.get_weights()
            steps_loss_not_improved += 1
            if (val_loss < best_val_loss):
                best_val_loss = val_loss
                steps_loss_not_improved = 0
            steps_to_early_stop = self.early_stop_epochs - min(steps_loss_not_improved, steps_metric_not_improved)
            print(f"\tval_{self.val_metric.name}: {val_metric:.5f}")
            print(f"\tbest_{self.val_metric.name}: {best_metric_val:.5f}")
            print(f"\ttrain_{self.val_metric.name}: {train_metric:.5f}")
            print(f"\ttrain_loss: {train_loss}")
            print(f"\tval_loss: {val_loss}")
            print(f"\tbest_val_loss: {best_val_loss}")
            print(f"\tsteps_metric_not_improved: {steps_metric_not_improved}")
            print(f"\tsteps_loss_not_improved: {steps_loss_not_improved}")
            print(f"\tsteps_to_stop: {steps_to_early_stop}")
            print(f"\ttotal_training_time: {total_trainig_time}")
            with self.tensorboard_writer.as_default(step=(epoch + 1)*self.max_batches_per_epoch*self.batch_size):
                tf.summary.scalar(f"val_{self.val_metric.name}", val_metric)
                tf.summary.scalar(f"train_{self.val_metric.name}", train_metric)
                tf.summary.scalar(f"{self.val_metric.name}_train_val_diff", train_metric - val_metric)
                tf.summary.scalar(f"best_{self.val_metric.name}", best_metric_val)
                tf.summary.scalar(f"tevaluations_without_improvement", steps_metric_not_improved)
                tf.summary.scalar(f"train_loss", train_loss)
                tf.summary.scalar(f"val_loss", val_loss)
                tf.summary.scalar(f"train_val_loss_diff", train_loss - val_loss)
                tf.summary.scalar(f"steps_metric_not_improved", steps_metric_not_improved)
                tf.summary.scalar(f"steps_loss_not_improved", steps_loss_not_improved)
                tf.summary.scalar(f"steps_to_early_stop", steps_to_early_stop)
                
            if steps_to_early_stop <= 0:
                print(f"early stopped at epoch {epoch}")
                break
            if self.training_time_limit is not None and total_trainig_time > self.training_time_limit:
                print(f"time limit stop triggered at epoch {epoch}")
                break
            generator.cleanup()

        data_generator_async_factory.close()
        self.model.set_weights(best_weights)
        self.metadata = {"epochs_trained": best_epoch + 1, f"best_val_{self.val_metric.name}": best_metric_val,
                         f"val_{self.val_metric.name}_history": val_metric_history}
        print(self.get_metadata())
        print(f"taken best model from epoch{best_epoch}. best_val_{self.val_metric.name}: {best_metric_val}")
        if self.use_ann_for_inference:
            self.build_ann_index()
        gc.collect()
        K.clear_session()
        tf.compat.v1.reset_default_graph()
    
    def build_ann_index(self):
        embedding_matrix = self.model.get_embedding_matrix().numpy()
        self.index = faiss.IndexFlatIP(embedding_matrix.shape[-1])
        self.index.add(embedding_matrix)
        pass
         
    def pass_parameters(self):
        self.loss.set_num_items(self.items.size())
        self.loss.set_batch_size(self.batch_size)
        self.train_history_vectorizer.set_sequence_len(self.model_arch.max_history_length)
        self.train_history_vectorizer.set_padding_value(self.items.size())
        self.pred_history_vectorizer.set_sequence_len(self.model_arch.max_history_length)
        self.pred_history_vectorizer.set_padding_value(self.items.size())
        
    def add_test_items_ranking_request(self, request: ItemsRankingRequest):
        self.item_ranking_requrests[request.user_id] = request 

    def train_epoch(self, generator, val_generator):
        val_generator.reset()
        pbar = tqdm(generator, ascii=True, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', position=0, leave=True, ncols=70)
        variables = self.model.variables
        loss_sum = 0
        num_batches = 0
        for X, y_true in pbar:
            num_batches += 1
            with tf.GradientTape() as tape:
                y_pred = self.model(X, training=True)
                loss_val = tf.reduce_mean(self.loss(y_true, y_pred))
            grad = tape.gradient(loss_val, variables)
            self.optimizer.apply_gradients(zip(grad, variables))
            loss_sum += loss_val
            pbar.set_description(f"loss: {loss_sum/num_batches:.5f}")
        pbar.close()
        train_loss = loss_sum/num_batches

        print("validating..")
        pbar = tqdm(val_generator, ascii=True, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', position=0, leave=True, ncols=70)
        val_loss_sum = 0 
        num_val_batches = 0
        for X, y_true in pbar:
            num_val_batches += 1
            y_pred = self.model(X, training=True)
            loss_val = tf.reduce_mean(self.loss(y_true, y_pred))
            val_loss_sum += loss_val
        val_loss = val_loss_sum / num_val_batches 

        val_metric = self.get_metric(self.val_recommendation_requets, self.val_seen, self.val_ground_truth)
        train_metric = self.get_metric(self.train_sample_recommendation_requests, self.train_sample_seen, self.train_sample_ground_truth)

        return train_loss, train_metric, val_loss, val_metric 

    def get_metric(self, recommendation_requests, seen_items, ground_truth):
        extra_recs = 0
        if self.flags.get('filter_seen', False):
            extra_recs += self.model_arch.max_history_length
        
        recs = self.recommend_batch(recommendation_requests, self.val_rec_limit + extra_recs, is_val=True)
        metric_sum = 0.0
        for rec, seen, truth in zip(recs, seen_items, ground_truth):
            if self.flags.get('filter_seen', False):
                filtered_rec = [recommended_item for recommended_item in rec if recommended_item[0] not in seen]
                metric_sum += self.val_metric(filtered_rec, truth) 
            else:
                metric_sum += self.val_metric(rec, truth) 
                
        val_metric = metric_sum / len(recs)
        return val_metric

    def train_val_split(self):
        all_user_ids = self.users_with_actions
        val_user_ids = [self.users.get_id(val_user) for val_user in self.val_users]
        train_user_ids = list(all_user_ids)
        val_users = self.user_actions_by_id_list(val_user_ids)
        train_users = self.user_actions_by_id_list(train_user_ids, val_user_ids)
        return train_users, val_users

    def get_model(self):
        self.model_arch.set_common_params(num_items=self.items.size(),
                                          num_users=self.users.size(),
                                          batch_size=self.batch_size)
        model = self.model_arch.get_model()
        return model

    def recommend(self, user_id, limit, features=None):
        if self.use_ann_for_inference:
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
        session = self.pred_history_vectorizer(model_actions)
        session = session.reshape(1, self.model_arch.max_history_length)
        model_inputs = [session]
        return model_inputs
    
    def recommend_multiple(self, recommendation_requets, limit, is_val=False):
        user_ids = [user_id for user_id, features in recommendation_requets]
        model_inputs = list(map(lambda id: self.get_model_inputs(id, is_val)[0], user_ids))
        model_inputs = tf.concat(model_inputs, 0)
        result = []
        if is_val or not(self.use_ann_for_inference):
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
        end = min(start + self.eval_batch_size, len(recommendation_requests))
        print("generating recommendation in batches...")
        pbar = tqdm(total = len(recommendation_requests), ascii=True, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',  position=0, leave=True, ncols=70)
        while (start < end):
            req = recommendation_requests[start:end]
            results += self.recommend_multiple(req, limit, is_val)
            pbar.update(end - start)
            start = end  
            end = min(start + self.eval_batch_size, len(recommendation_requests))
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
