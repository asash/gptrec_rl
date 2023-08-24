from pickle import UnpicklingError

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
    def __init__(self, checkoint):
        try:
            self.bert4rec:SequentialRecommender = dill.load(gzip.open(checkoint, 'rb')).recommender
        except UnpicklingError as e: #TODO this is a hack. The models store link to the experiment file, so we need to create file.
            msg = str(e)
            if "No such file or directory" in msg:
                filename = Path(msg.split(':')[1].strip().strip("'"))
                filename.parent.mkdir(parents=True, exist_ok=True)
                filename.touch()
                self.bert4rec = dill.load(gzip.open(checkoint, 'rb')).recommender
            else:
                raise e
        pass

    def add_action(self, action: Action):
        pass

    def rebuild_model(self):
        pass

    def recommend_by_items(self, actions, limit, filter_seen=True):
        pred_history_vectorizer = self.bert4rec.config.pred_history_vectorizer
        actions_internal = [(0, self.bert4rec.items.get_id(action)) for action in actions]
        hist = pred_history_vectorizer(actions_internal)
        hist = hist.reshape(1, self.bert4rec.config.sequence_length)
        model_inputs = [hist]
        scoring_func = self.bert4rec.get_scoring_func()
        scores = scoring_func(model_inputs).numpy()[0]
        if filter_seen:
            for action in actions:
                scores[self.bert4rec.items.get_id(action)] = -np.inf
        best_ids = tf.nn.top_k(scores, limit).indices.numpy()
        result = [(self.bert4rec.items.reverse_id(id), scores[id]) for id in best_ids]
        return result
        