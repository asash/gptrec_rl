from random import Random
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD

import numpy as np
from aprec.recommenders.dnn_sequential_recommender.target_builders.target_builders import TargetBuilder

class ItemsMaskingWithNegativesTargetsBuilder(TargetBuilder):
    def __init__(self, negatives_sampler, 
                       random_seed=31337, 
                       relative_positions_encoding = True, 
                       ignore_value=-100): #-100 is used by default in hugginface's BERT implementation
        self.random = Random()
        self.random.seed(random_seed) 
        self.ignore_value = ignore_value
        self.relative_positions_encoding = relative_positions_encoding
        self.positions = []
        self.negatives_sampler = negatives_sampler

    def set_n_items(self, n):
        super().set_n_items(n)
        self.negatives_sampler.set_n_items(n)
    
    def set_train_sequences(self, train_sequences):
        self.negatives_sampler.set_train_sequences(train_sequences)

    def build(self, user_targets):
        label_seqs = []
        positions = []
        targets = []
        for seq_len, user in user_targets:
            user_positions = []
            user_negatives = [self.negatives_sampler.default_vector()] * self.sequence_len
            user_positives = [self.n_items + 2] * self.sequence_len
            user_targets = [[self.ignore_value] * (len(user_negatives[0]) +1)] * self.sequence_len
            target_vector = [1] + [0] * len(user_negatives[0])

            if self.relative_positions_encoding:
                split_pos = self.random.randint(self.sequence_len - seq_len, self.sequence_len - 1)
            else:
                split_pos = self.sequence_len - 1

            for i in range(self.sequence_len):
                user_positions.append(self.sequence_len - split_pos  + i) 

            positions.append(user_positions)
            for pos in user:
                user_positives[pos[0]] = pos[1][1]
                user_negatives[pos[0]] = self.negatives_sampler.sample_negatives(pos[1][1])
                user_targets[pos[0]] = target_vector
            user_negatives = np.array(user_negatives)
            user_targets = np.array(user_targets)
            user_positives = np.transpose([user_positives])
            user_result = np.concatenate([user_positives, user_negatives], axis=1)
            targets.append(user_targets)
            label_seqs.append(user_result)
        self.positions = np.array(positions)
        self.label_seqs = np.array(label_seqs)
        self.targets = np.array(targets)

    def get_targets(self, start, end):
        return [self.label_seqs[start:end], self.positions[start:end]], self.targets[start:end]


class NegativesSampler(object):
    def default_vector(self):
        return self.def_vector

    def set_n_items(self, n):
        self.n_items = n
        self.def_vector = [n+2] * self.sample_size 

   
    def sample_negatives(self, positive):
        raise NotImplementedError()

    def get_sample_size(self):
        return self.sample_size

    def set_train_sequences(self, train_sequences):
        pass


class SVDSimilaritySampler(NegativesSampler):
    def __init__(self, sample_size, seed=31337, ann_sampling_factor=2, num_svd_components=64):
        self.sample_size = sample_size
        self.random = Random()
        self.random.seed(seed) 
        self.num_svd_components=num_svd_components
        self.random_seed = seed
        self.ann_sampling_factor = ann_sampling_factor

    def set_train_sequences(self, train_sequences):
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
        neighbors = ann.kneighbors(embeddings, return_distance=True)
        self.items = neighbors[1][:,1:]
        distances = neighbors[0][:,1:]
        sims = 1/distances
        self.probs = np.transpose(sims.T / np.sum(sims, axis=1))
        pass

    def sample_negatives(self, positive):
        sample = self.random.choices(self.items[positive], weights=self.probs[positive], k=self.sample_size) 
        return sample
    
 


class RandomNegativesSampler(NegativesSampler):
    def __init__(self, sample_size, seed=31337):
        self.sample_size = sample_size
        self.random = Random()
        self.random.seed(seed) 
    


    def sample_negatives(self, positive):
        result = []
        while len(result) < self.sample_size:
            sample = self.random.randint(0, self.n_items - 1)
            if sample != positive:
                result.append(sample)
        return result

  