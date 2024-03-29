from argparse import Action
from collections import Counter, defaultdict

import tqdm
from aprec.recommenders.recommender import Recommender
from aprec.utils.item_id import ItemId
import numpy as np


class SmartMC(Recommender):
    def __init__(self, cache_items=1000, order=200, discount = 0.8):
        super().__init__()
        self.user_actions = defaultdict(list)
        self.cache_items = cache_items
        self.transitions_counter = 0
        self.order = order
        self.item_id = ItemId()
        self.discount = discount

    def add_action(self, action: Action):
        internal_item_id = self.item_id.get_id(action.item_id) 
        self.user_actions[action.user_id].append(internal_item_id)

    def rebuild_model(self):
        self.src_count = np.zeros(self.item_id.size())
        self.total_count = 0
        self.src_dst_dist_count = np.zeros([self.item_id.size(), self.item_id.size(), self.order])
        for userid, session in tqdm.tqdm(self.user_actions.items(), ascii=True, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', position=0, leave=True, ncols=70):
            for start_idx in range(len(session)):
                end_idx = min(start_idx + self.order + 1, len(session))
                src_action = session[start_idx]
                self.src_count[src_action] += 1
                self.total_count += 1
                for dst_idx in range(start_idx +1, end_idx):
                    dst_action = session[dst_idx]                    
                    dist = dst_idx - start_idx -1 
                    self.src_dst_dist_count[src_action][dst_action][dist] += 1
        pass
        

    def recommend(self, user_id, limit: int, features=None):
        if user_id not in self.user_actions:
            return []
        session = self.user_actions[user_id]
        return self.session_recommend(session, limit) 

    def session_recommend(self, session, limit, filter_seen=True):
        result = np.zeros(self.item_id.size(), dtype=np.float32)
        for dist in range(min(self.order, len(session))):
            src = session[-dist-1]
            src_counter = self.src_count[src]
            counters = self.src_dst_dist_count[src, :, dist]
            eps = 0.1
            discount = self.discount ** dist
            log_probs = np.log((counters + eps)/src_counter)
            try:
                result += discount*log_probs
            except ValueError as ve:
                pass
            pass
        if filter_seen:
            for item in session:
                result[item] = -np.inf
        recs = np.argsort(result)[::-1][:limit]
        final_result = []
        for internal_id in recs:
            final_result.append((self.item_id.reverse_id(internal_id), result[internal_id]))
        return final_result
 



    #items list is the sequence of user-item interactions
    def recommend_by_items(self, items_list, limit: int, filter_seen=True):
        session = []
        for item in items_list:
            session.append(self.item_id.get_id(item))
        return self.session_recommend(session, limit, filter_seen)
            
