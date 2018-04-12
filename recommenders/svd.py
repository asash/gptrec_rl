from aprec.utils.item_id import ItemId
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import numpy as np

class SvdRecommender():
    def __init__ (self, num_latent_components, random_seed=None):
        self.latent_components = num_latent_components
        self.users = ItemId()
        self.items = ItemId()
        self.rows = []
        self.cols = []
        self.vals = []
        self.model = None 
        self.user_vectors = None
        self.mean_user = None
        self.random_seed = random_seed
        
    def name(self):
        return "Svd@{}".format(self.latent_components)

    def add_action(self, action):
        row = self.users.get_id(action.user_id)
        col = self.items.get_id(action.item_id)
        self.rows.append(row)
        self.cols.append(col)
        self.vals.append(1.0)

    def rebuild_model(self):
        matrix = csr_matrix((self.vals, (self.rows, self.cols)))
        self.model = TruncatedSVD(n_components=self.latent_components, random_state=self.random_seed)
        self.user_vectors = self.model.fit_transform(matrix)
        self.mean_user = np.mean(self.user_vectors, axis=0)

    def get_next_items(self, user_id, limit):
        user_vec = self.mean_user
        if self.users.has_item(user_id):
            user_vec = self.user_vectors[self.users.get_id(user_id)]
            print(user_vec) 
        restored = self.model.inverse_transform([user_vec])[0]
        with_index = list(zip(restored, range(len(restored))))
        score_idx = sorted(with_index, reverse=True)[:limit]
        result = [(self.items.reverse_id(id), score) for (score, id) in score_idx]
        return result

    def get_similar_items(self, item_id, limit):
        raise(NotImplementedError)

    def to_str(self):
        raise(NotImplementedError)

    def from_str(self):
        raise(NotImplementedError)
