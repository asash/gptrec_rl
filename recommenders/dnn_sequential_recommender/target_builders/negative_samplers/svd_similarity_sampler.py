import logging
from random import Random
import numpy as np
from scipy.sparse import csr_matrix

from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from aprec.recommenders.dnn_sequential_recommender.target_builders.negative_samplers.negatives_sampler import NegativesSampler


class SVDSimilaritySampler(NegativesSampler):
    def __init__(self, sample_size=400, seed=31337, ann_sampling_factor=1, num_svd_components=64):
        self.sample_size = sample_size
        self.random = Random()
        self.random.seed(seed) 
        self.num_svd_components=num_svd_components
        self.random_seed = seed
        self.ann_sampling_factor = ann_sampling_factor
        self.values = [0] * self.sample_size 

    def set_train_sequences(self, train_sequences):
        logging.warning("building svd similarty sampler...")
        rows = []
        cols = []
        vals = []
        max_item = 0
        for i in range(len(train_sequences)):
            for item in train_sequences[i]:
                rows.append(item[1])
                cols.append(i)
                max_item = max(item[1], max_item)
                vals.append(1.0)

        matrix = csr_matrix((vals, (rows, cols)))
        svd = TruncatedSVD(n_components=self.num_svd_components, random_state=self.random_seed)
        embeddings = svd.fit_transform(matrix)
        ann = NearestNeighbors(n_neighbors = self.sample_size * self.ann_sampling_factor + 1) 
        ann.fit(embeddings)
        self.samples = []
        self.probs = []
        logging.warning("computing knn index in svd similarity sampler...")
        neighbors = ann.kneighbors(embeddings, return_distance=True)
        self.items = neighbors[1][:,1:]
        distances = neighbors[0][:,1:]
        sims = 1/distances
        self.probs = np.transpose(sims.T / np.sum(sims, axis=1))
        logging.warning("svd similarity index built.\n")

    def sample_negatives(self, positive):
        sample = self.random.choices(self.items[positive], weights=self.probs[positive], k=self.sample_size) 
        return sample, self.values