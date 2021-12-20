from aprec.datasets.movielens100k import get_movielens100k_actions
from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.svd import SvdRecommender
from aprec.recommenders.lightfm import LightFMRecommender
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.recommenders.matrix_factorization import MatrixFactorizationRecommender
from aprec.recommenders.random_recommender import RandomRecommender
from aprec.recommenders.salrec.salrec_recommender import SalrecRecommender
from aprec.evaluation.metrics.precision import Precision
from aprec.evaluation.metrics.recall import Recall
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.hit import HIT
from tensorflow.keras.optimizers import Adam
from aprec.evaluation.split_actions import random_split
import numpy as np


DATASET = get_movielens100k_actions(min_rating=1.0)

USERS_FRACTIONS = [1]

def top_recommender():
    return FilterSeenRecommender(TopRecommender())

def svd_recommender(k):
    return FilterSeenRecommender(SvdRecommender(k))

def mf_recommender(embedding_size, num_epochs, loss, batch_size, regularization, learning_rate):
    return FilterSeenRecommender(MatrixFactorizationRecommender(embedding_size,
                     num_epochs, loss, batch_size, regularization, learning_rate))

def lightfm_recommender(k, loss):
    return FilterSeenRecommender(LightFMRecommender(k, loss))

def salrec(loss, learning_rate, regularization, max_history_len, batch_size, num_blocks, 
                                             num_target_predictions, embedding_size, bottleneck_size, num_bottlenecks,early_stop,
                                             activation_override=None):
    activation = 'sigmoid' if loss == 'binary_crossentropy' else 'linear'
    if activation_override is not None:
        activation = activation_override
    return FilterSeenRecommender(SalrecRecommender(train_epochs=10000, loss=loss,
                                                   optimizer=Adam(learning_rate=learning_rate), early_stop_epochs=early_stop,
                                                   batch_size=batch_size, sigma=1.0, ndcg_at=40,
                                                   max_history_len=max_history_len,
                                                   output_layer_activation=activation,
                                                   num_blocks=num_blocks,
                                                   num_target_predictions=num_target_predictions,
                                                   target_decay=1.0,
                                                   regularization=regularization,
                                                   embedding_size=embedding_size, 
                                                   bottleneck_size=bottleneck_size,
                                                   num_bottlenecks=num_bottlenecks,
                                                   positional = False
                                                   ))


RECOMMENDERS = {
    "top_recommender": top_recommender,
    "random_recommender": RandomRecommender,
    "svd_recommender_32": lambda: svd_recommender(32),
    "lightfm_recommender_30_bpr": lambda: lightfm_recommender(30, 'bpr'),
    "lightfm_recommender_30_warp": lambda: lightfm_recommender(30, 'warp'),
}

for i in range(10000):
    all_losses = ['binary_crossentropy', 'xendcg', 'lambdarank', 'bpr', 'climf', 'mse']
    loss = all_losses[i % len(all_losses)]
    regularization = float(np.random.choice([0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2, 4, 8]))
    embedding_size = int(np.random.choice([16, 32, 64, 128, 256, 512, 1024]))
    batch_size = int(np.random.choice([32, 64]))
    bottleneck_size=int(np.random.choice([16, 32, 64, 128, 256, 512, 1024]))
    early_stop=int(np.random.choice([16, 32, 64, 128, 256]))
    max_history_len=int(np.random.choice([8, 16, 32, 64, 128, 256, 512]))
    num_bottlenecks=int(np.random.choice([0, 1, 2, 3, 4, 5]))
    num_blocks=int(np.random.choice([0, 1, 2, 3, 4, 5]))
    learning_rate = float(np.random.choice([0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]))
    num_target_predictions=np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    recommender = lambda loss=loss, learning_rate=learning_rate, regularization=regularization, max_history_len=max_history_len,\
                       batch_size=batch_size, num_blocks=num_blocks, \
                       num_target_predictions=num_target_predictions, \
                       embedding_size=embedding_size, \
                       bottleneck_size=bottleneck_size, \
                       num_bottlenecks=num_bottlenecks, \
                       early_stop=early_stop: \
                        salrec(loss, learning_rate, regularization, max_history_len, batch_size, num_blocks,  num_target_predictions, embedding_size, bottleneck_size, num_bottlenecks, early_stop)
    name = f"salrec_l:{loss}_lr:{learning_rate}_reg:{regularization}_maxHist:{max_history_len}_"\
                        f"bs:{batch_size}_nBlock:{num_blocks}_nTargets:{num_target_predictions}_"\
                        f"es:{embedding_size}_bottleneck:{bottleneck_size}_nBottlenecks:{num_bottlenecks}_earlySt:{early_stop}"
    RECOMMENDERS[name] = recommender


TEST_FRACTION = 0.25
MAX_TEST_USERS=943
N_VAL_USERS=64

METRICS = [NDCG(40), Precision(5), Recall(5), HIT(10), MRR(), MAP(10)]


RECOMMENDATIONS_LIMIT = 100
SPLIT_STRATEGY = lambda actions: random_split(actions),

