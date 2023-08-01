from random import Random

import numpy as np
from aprec.recommenders.recommender import Recommender
from aprec.recommenders.sequential.target_builders.target_builders import TargetBuilder
from aprec.utils.item_id import ItemId


class PreTrainTargetsBuilder(TargetBuilder):
    def __init__(self, pre_train_recommender: Recommender, 
                       sequence_vectorizer,
                       item_ids: ItemId, 
                       random_seed=31337, 
                       ignore_value=-100, 
                       generation_limit=10,
                ):
        self.random = Random()
        self.random.seed(random_seed) 
        self.targets = []
        self.ignore_value = ignore_value
        self.positions = []
        self.item_ids = item_ids
        self.pre_train_recommender = pre_train_recommender
        self.generation_limit = generation_limit
        self.sequence_vectorizer = sequence_vectorizer


    def build(self, user_targets):
        self.positions = []
        self.targets = []   
        
        for sequence in user_targets:
            sequence_len = len(sequence)
            sequence_ids =  [item_id for _, item_id in sequence]
            decoded_ids = [self.item_ids.reverse_id(item_id) for item_id in sequence_ids]
            sep_item_id = self.item_ids.get_id("<SEP>")
            vectorized_sequence = self.sequence_vectorizer(sequence)
            recommendations = self.pre_train_recommender.recommend_by_items(decoded_ids, self.generation_limit) 
            encoded_recommendations = [self.item_ids.get_id(rec[0]) for rec in recommendations]
            vectorized_sequence = np.concatenate((vectorized_sequence, [sep_item_id], encoded_recommendations))
            position_ids = np.concatenate([np.arange(self.sequence_len, -1, -1), np.arange(self.sequence_len+1, self.sequence_len + 1 + len(recommendations))])
            self.positions.append(position_ids)
            self.targets.append(vectorized_sequence)
            pass
        self.positions = np.array(self.positions)
        self.targets = np.array(self.targets)



    def get_targets(self, start, end):
        return [self.targets[start:end], self.positions[start:end]], self.targets[start:end]

