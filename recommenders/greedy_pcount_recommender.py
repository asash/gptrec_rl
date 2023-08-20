from collections import Counter, defaultdict
from functools import lru_cache
from aprec.api.action import Action
from aprec.recommenders.recommender import Recommender
import gzip
import numpy as np
import dill 


class PCountRecommender(Recommender):
    def __init__(self, checkpoint, actions, pcount_cutoff=1000, pcount_lambda=0.5):
        self.recommender = dill.load(gzip.open(checkpoint, 'rb'))
        self.pcount_cutoff = pcount_cutoff
        self.pcount_lambda = pcount_lambda
        actions_counter = Counter() 
        for action in actions:
            actions_counter[action.item_id] += 1
        self.probs = {}
        actions_counter = Counter() 
        for action in actions:
            actions_counter[action.item_id] += 1
        self.probs = {}
        for item_id in actions_counter:
            self.probs[item_id] = actions_counter[item_id]/len(actions)
        self.out_dir = None
       
    
    def add_action(self, action: Action):
        pass #we load pre-trained recommender
    
    def rebuild_model(self):
        pass #we load pre-trained recommender
    
    def recommend(self, user_id, limit: int, features=None):
        recommendations = self.recommender.recommend(user_id, self.pcount_cutoff)
        if len(recommendations) < limit:
            raise Exception("Not enough recommendations for user " + str(user_id))
        
        for i in range(len(recommendations)):
            recommendations[i][1] = self.pcount_lambda * recommendations[i][1] - (1-self.pcount_lambda)*self.probs[recommendations[i][0]]

        recommendations.sort(key=lambda x: -x[1])
        return recommendations
   