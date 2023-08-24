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
        