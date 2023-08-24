
from multiprocessing.context import SpawnContext, SpawnProcess
from typing import Any

import numpy as np

from aprec.utils.item_id import ItemId


class RecommenderProcess(object):
    def __init__(self,in_queue, out_queue, users: ItemId, items:ItemId,
                 user_actions, model_checkpoint, model_config,
                 filter_seen, gen_limit,
                pred_history_vectorizer) -> None:
        self.users = users
        self.items = items
        self.user_actions = user_actions
        self.model_checkpoint = model_checkpoint
        self.model_config = model_config
        self.filter_seen = filter_seen
        self.gen_limit = gen_limit
        self.pred_history_vectorizer = pred_history_vectorizer
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.model=None

    def get_pred_sequence(self, internal_user_id):
        actions = self.user_actions[internal_user_id]
        sep_item_id = self.items.get_id('<SEP>')
        sequence = [action[1] for action in actions]
        sequence.append(sep_item_id)
        return sequence

    def ensure_model(self):
        from aprec.recommenders.sequential.models.generative.gpt_rec_rl import RLGPT2RecModel
        if self.model is None: 
            self.model = RLGPT2RecModel.from_config(self.model_config)
            self.model.load_weights(self.model_checkpoint + "/model.h5")
            self.model.fit_biases(None)

    def recommend(self, user_id) -> Any:
        from aprec.recommenders.rl_generative.generator import static_generate
        self.ensure_model()
        internal_user_id = self.users.get_id(user_id)
        seq = self.get_pred_sequence(internal_user_id)
        sep_item_id = self.items.get_id('<SEP>')
        user_recs = static_generate(seq, self.filter_seen, sep_item_id, greedy=True,
                                              train=False, items=self.items, gen_limit=self.gen_limit,
                                              pred_history_vectorizer=self.pred_history_vectorizer,
                                              model=self.model)[0]
        recs = [] 
        for i in range(len(user_recs)):
            score = 1/np.log2(i+2)
            item_id = self.items.reverse_id(user_recs[i])
            recs.append((item_id, score))
        return recs

    def __call__(self):
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        while True:
            user_id, features = self.in_queue.get()
            recs = self.recommend(user_id)
            self.out_queue.put((user_id, recs))

class MultiprocessRecommendationProcessor(object):
    def __init__(self, recommendation_requets, n_processes, users, items,
                 user_actions, model_checkpoint, model_config,
                 filter_seen, gen_limit,
                pred_history_vectorizer) -> None:
        ctx = SpawnContext()
        self.result_queue = ctx.Queue()
        self.task_queue = ctx.Queue()
        self.processes = []
        for task in recommendation_requets:
            self.task_queue.put(task)
        self.n_processes = n_processes
        self.processess = []
        self.users = users 
        self.items = items
        self.user_actions = user_actions
        self.model_checkpoint = model_checkpoint
        self.model_config = model_config
        self.filter_seen = filter_seen
        self.gen_limit = gen_limit
        self.pred_history_vectorizer = pred_history_vectorizer
        

    def __enter__(self):
        for i in range(self.n_processes):
            recommender = RecommenderProcess(self.task_queue, self.result_queue, self.users, self.items,
                 self.user_actions, self.model_checkpoint, self.model_config,
                 self.filter_seen, self.gen_limit,
                self.pred_history_vectorizer)
            recommender_process = SpawnProcess(target=recommender)
            recommender_process.daemon = True
            recommender_process.start()
            self.processes.append(recommender_process)
        return self

    def next_recommendation(self):
        user_id, recs = self.result_queue.get()
        return user_id, recs

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for process in self.processes:
            process.terminate()
            process.join() 
