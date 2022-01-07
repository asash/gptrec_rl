from typing import List

import lightgbm
from aprec import recommenders
from aprec.api.action import Action
from aprec.api.item import Item
from aprec.api.user import User
from aprec.recommenders.recommender import Recommender
import numpy as np
from lightgbm import LGBMRanker, Dataset
from lightgbm.callback import early_stopping, log_evaluation

class LambdaMARTEnsembleRecommender(Recommender):
    def __init__(self, candidates_selection_recommender: Recommender, 
                                other_recommenders: List[Recommender], 
                                n_ensemble_users = 1000,
                                n_ensemble_val_users = 256,
                                candidates_limit = 1000, 
                                ndcg_at = 40,
                                num_leaves = 15,
                                max_trees = 20000,
                                early_stopping=1000,
                                lambda_l2=0.0,
                                booster='gbdt'
                                ):
        super().__init__()
        self.candidates_selection_recommender = candidates_selection_recommender
        self.other_recommenders = other_recommenders
        self.n_ensemble_users = n_ensemble_users
        self.user_actions = {}
        self.candidates_limit = candidates_limit
        self.max_trees = max_trees 
        self.early_stopping=early_stopping
        self.booster = booster
        self.ndcg_at = ndcg_at
        self.n_ensemble_val_users = n_ensemble_val_users
        self.num_leaves = num_leaves
    
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
        all_users = list(self.user_actions.keys())
        ensemble_users_selection = list(self.user_actions.keys() - set(self.val_users))
        ensemble_users = set(np.random.choice(ensemble_users_selection, self.n_ensemble_users))
        ensemble_val_users_selection = list(self.user_actions.keys() - set(self.val_users) - ensemble_users)
        ensemble_val_users = set(np.random.choice(ensemble_val_users_selection, self.n_ensemble_val_users))
        
        selected_users = ensemble_users.union(ensemble_val_users)

        for user in all_users:
            if user not in selected_users:
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


        train_dataset = self.get_data(ensemble_users)
        val_dataset = self.get_data(ensemble_val_users)

        self.ranker = lightgbm.train(
            params={
             'objective': 'lambdarank',
             'eval_at': self.ndcg_at,
             'boosting': self.booster, 
             'num_leaves': self.num_leaves, 
            },
            train_set=train_dataset, 
            valid_sets=[val_dataset], 
            num_boost_round=self.max_trees,
            early_stopping_rounds=self.early_stopping
        )
    
    def get_metadata(self):
        feature_importance =  list(zip(self.ranker.feature_name(), self.ranker.feature_importance()))
        result = {}
        result['feature_importance'] = []
        for feature, score in sorted(feature_importance, key=lambda x: -x[1]):
            result['feature_importance'].append((feature, int(score)))
        return result


    def get_data(self, users):
        features_list = ['sessions_len', 'candidate_recommender_idx', 'candidate_recommender_score']
        for recommender in self.other_recommenders:
            features_list += (f'is_present_in_{recommender}', f'{recommender}_idx', f'{recommender}_score')

        samples = []
        target = []
        group = []
        for user_id in users:
            candidates = self.build_candidates(user_id)
            target_id = self.user_actions[user_id][-1].item_id
            for candidate in candidates:
                samples.append(candidates[candidate])
                target.append(int(candidate == target_id))
            group.append(len(candidates))
        return Dataset(np.array(samples),label=target, group=group, feature_name=features_list, free_raw_data=False).construct()

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
            recommender_processed_candidates = set()
            for idx, candidate in enumerate(recs):
                if candidate[0] in candidate_features:
                    candidate_features[candidate[0]] += [1, idx, candidate[1]]
                    recommender_processed_candidates.add(candidate[0])
            for candidate in candidate_features:
                if candidate not in recommender_processed_candidates:
                    candidate_features[candidate] += [0, self.candidates_limit, -1000]
                    recommender_processed_candidates.add(candidate)
        return candidate_features
    
    def set_val_users(self, val_users):
        self.val_users = val_users
        self.candidates_selection_recommender.set_val_users(val_users=val_users)
        for recommender in self.other_recommenders:
            self.other_recommenders[recommender].set_val_users(val_users)