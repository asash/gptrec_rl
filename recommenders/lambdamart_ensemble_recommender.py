from typing import List
from aprec import recommenders
from aprec.api.action import Action
from aprec.api.item import Item
from aprec.api.user import User
from aprec.recommenders.recommender import Recommender
import numpy as np
from lightgbm import LGBMRanker

class LambdaMARTEnsembleRecommender(Recommender):
    def __init__(self, candidates_selection_recommender: Recommender, 
                                other_recommenders: List[Recommender], 
                                n_ensemble_users = 1000,
                                candidates_limit = 1000, 
                                ):
        super().__init__()
        self.candidates_selection_recommender = candidates_selection_recommender
        self.other_recommenders = other_recommenders
        self.n_ensemble_users = n_ensemble_users
        self.user_actions = {}
        self.candidates_limit = candidates_limit
    
    def add_item(self, item: Item):
        self.candidates_selection_recommender.add_item(item)
        for other_recommender in self.other_recommenders:
            self.other_recommenders[other_recommender].add_item(item)

    def add_user(self, user: User):
        self.candidates_selection_recommender.add_user(user)
        for other_recommender in self.other_recommenders:
            self.other_recommenders[other_recommender].add_user(user)

    def add_action(self, action: Action):
        if action.user_id not in self.user_actions:
            self.user_actions[action.user_id] = [action]
        else:
            self.user_actions[action.user_id].append(action)

    def rebuild_model(self):
        all_users = list(self.user_actions.keys() - set(self.val_users))
        ensemble_users = set(np.random.choice(all_users, self.n_ensemble_users))
        for user in all_users:
            if user not in ensemble_users:
                all_actions = self.user_actions[user]
            else:
                all_actions = self.user_actions[user][:-1]

            for action in all_actions:
                self.candidates_selection_recommender.add_action(action)
                for recommender in self.other_recommenders:
                    self.other_recommenders[recommender].add_action(action)

        print("rebuilding candidates selection recommender...")
        self.candidates_selection_recommender.rebuild_model()

        for other_recommender in self.other_recommenders:
            print(f"rebuilding recommender {other_recommender}")
            self.other_recommenders[other_recommender].rebuild_model()


        samples = []
        target = []
        group = []
        for user_id in ensemble_users:
            candidates = self.build_candidates(user_id)
            target_id = self.user_actions[user_id][-1].item_id
            for candidate in candidates:
                samples.append(candidates[candidate])
                target.append(int(candidate == target_id))
            group.append(len(candidates))
        self.ranker = LGBMRanker()
        self.ranker.fit(samples, target, group=group)

    def recommend(self, user_id, limit: int, features=None):
        candidates = self.build_candidates(user_id)
        items = []
        features = []
        for candidate in candidates:
            items.append(candidate)
            features.append(candidates[candidate])
        scores = self.ranker.predict(features) 
        recs = list(zip(items, scores))
        return sorted(recs, key=lambda x: -x[1])[:limit]



    
    def build_candidates(self, user_id):
        candidates = self.candidates_selection_recommender.recommend(user_id, limit=self.candidates_limit)
        candidate_features = {}
        for idx, candidate in enumerate(candidates):
            user_session_len = len(self.user_actions.get(user_id, ()))
            candidate_features[candidate[0]] = [user_session_len, idx, candidate[1]]

        cnt = 1
        for recommender in self.other_recommenders:
            cnt += 1
            recs = self.other_recommenders[recommender].recommend(user_id, limit=self.candidates_limit)
            for idx, candidate in enumerate(recs):
                if candidate[0] in candidate_features:
                    candidate_features[candidate[0]] += [idx, candidate[1]]
            for candidate in candidate_features:
                if len(candidate_features[candidate]) < 2*cnt:
                    candidate_features[candidate] += [self.candidates_limit, -1000]
        return candidate_features
    
    def set_val_users(self, val_users):
        self.val_users = val_users
        self.candidates_selection_recommender.set_val_users(val_users=val_users)
        for recommender in self.other_recommenders:
            self.other_recommenders[recommender].set_val_users(val_users)