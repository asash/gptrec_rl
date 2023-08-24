from pickle import UnpicklingError
import time

import numpy as np
import tensorflow as tf
from aprec.api.action import Action
from aprec.recommenders.recommender import Recommender
import gzip
import dill
from pathlib import Path

from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender

#use PreTrained SequentialRecommender as a teacher for gptrec
class TeacherRecommender(Recommender):
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.sequential_recommender:SequentialRecommender = None
        self.flags = {}
        self.out_dir = None

    def ensure_model(self):
        if self.sequential_recommender is not None:
            return
        try:
            self.sequential_recommender:SequentialRecommender = dill.load(gzip.open(self.checkpoint, 'rb')).recommender
        except UnpicklingError as e: #TODO this is a hack. The models store link to the experiment file, so we need to create file.
            msg = str(e)
            if "No such file or directory" in msg:
                filename = Path(msg.split(':')[1].strip().strip("'"))
                filename.parent.mkdir(parents=True, exist_ok=True)
                filename.touch()
                self.sequential_recommender = dill.load(gzip.open(self.checkpoint, 'rb')).recommender
            else:
                raise e
        pass
    

    def add_action(self, action: Action):
        pass

    def rebuild_model(self):
        pass

    def recommend_by_items_multiple(self, user_actions, limit):
        self.ensure_model()
        model_inputs = []
        pred_history_vectorizer = self.sequential_recommender.config.pred_history_vectorizer
        for actions in user_actions:
            actions_internal = [(0, self.sequential_recommender.items.get_id(action)) for action in actions]
            hist = pred_history_vectorizer(actions_internal)
            hist = hist.reshape(1, self.sequential_recommender.config.sequence_length)
            model_inputs.append(hist)
        scoring_func = self.sequential_recommender.get_scoring_func()
        model_inputs = [tf.concat(model_inputs, axis=0)]
        predictions = scoring_func([model_inputs])
        best_predictions = tf.math.top_k(predictions, k=limit)
        ind = best_predictions.indices.numpy()
        vals = best_predictions.values.numpy()    
        result = []
        for i in range(len(user_actions)):
            result.append(list(zip(self.sequential_recommender.decode_item_ids(ind[i]), vals[i])))
        return result
        
    def recommend_by_items(self, actions, limit, filter_seen=True):
        self.ensure_model()
        pred_history_vectorizer = self.sequential_recommender.config.pred_history_vectorizer
        actions_internal = [(0, self.sequential_recommender.items.get_id(action)) for action in actions]
        hist = pred_history_vectorizer(actions_internal)
        hist = hist.reshape(1, self.sequential_recommender.config.sequence_length)
        model_inputs = [hist]
        scoring_func = self.sequential_recommender.get_scoring_func()
        scores = scoring_func(model_inputs).numpy()[0]
        if filter_seen:
            for action in actions:
                scores[self.sequential_recommender.items.get_id(action)] = -np.inf
        best_ids = tf.nn.top_k(scores, limit).indices.numpy()
        result = [(self.sequential_recommender.items.reverse_id(id), scores[id]) for id in best_ids]
        return result

if __name__ == "__main__":
    checkpoint="/home/aprec/Projects/aprec/evaluation/results/checkpoints_for_rl/steam_baselines/bert4rec.dill.gz"
    pretrain_recommender=TeacherRecommender(checkpoint)  
    pretrain_recommender.ensure_model()

    batch = []
    for i in range(128):
        batch.append([pretrain_recommender.sequential_recommender.items.reverse_id(id) for timestamp, id in pretrain_recommender.sequential_recommender.user_actions[i]])
    time_batch_start = time.time()
    recs = pretrain_recommender.recommend_by_items_multiple(batch, 10)
    time_batch_end = time.time()
    recs_one_by_one = [pretrain_recommender.recommend_by_items(actions, 10, filter_seen=False) for actions in batch]
    time_one_by_one_end = time.time()
    print("one_by_one_time", time_one_by_one_end - time_batch_end)
    print("batch_itme", time_batch_end - time_batch_start)
    pass
