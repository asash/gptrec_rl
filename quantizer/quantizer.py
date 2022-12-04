from collections import defaultdict
from typing import List
import numpy as np
from sklearn.decomposition import TruncatedSVD
from aprec.api.action import Action
from scipy.sparse import csr_matrix

from aprec.utils.item_id import ItemId

class Quantizer(object):
    def __init__(self, num_components=128, quants_per_component=10, item_ids = ItemId(), user_ids=ItemId()):
        self.num_components = num_components
        self.quants_per_component = quants_per_component
        self.item_ids = ItemId()
        self.user_ids = ItemId()
        self.percentiles = np.linspace(0, 100, quants_per_component+1)


    def fit(self, actions: List[Action]):
        rows = []
        cols = []
        vals = []
        for action in actions:
            cols.append(self.user_ids.get_id(action.user_id))
            rows.append(self.item_ids.get_id(action.item_id))
            vals.append(1)
        matrix = csr_matrix((vals, (rows, cols)))
        svd = TruncatedSVD(self.num_components)
        item_embeddings = svd.fit_transform(matrix) 
        percentile_borders = np.transpose(np.percentile(item_embeddings, self.percentiles, 0))
        self.item_representations = []

        #TODO: can this be vectorized? 
        for item_id in range(len(item_embeddings)):
            item_embedding = item_embeddings[item_id]
            item_repr = []
            for latent_dimension in range(self.num_components):
                dim_pct = get_percentile(item_embedding[latent_dimension], percentile_borders[latent_dimension])
                val = latent_dimension * self.quants_per_component + dim_pct
                item_repr.append(val)
            self.item_representations.append(item_repr)
        self.item_representations = np.array(self.item_representations)
    
    def set_train_sequences(train_sequences):
        pass
    
    def sim(self, item_id, k=10):
        internal_id = self.item_ids.get_id(item_id)
        item_repr = self.item_representations[internal_id]
        scores = np.sum((item_repr == self.item_representations), axis=-1)
        best_items = scores.argsort()[::-1]
        result = []
        for item in best_items[:k]:
            result.append((self.item_ids.reverse_id(item), scores[item]))
        return result
            


    
def get_percentile(val, borders):
    for i in range(len(borders) - 1):
        if val >= borders[i] and val < borders[i+1]:
            return i 
    return len(borders) -2
            