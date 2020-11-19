from aprec.datasets.movielens import get_movielens_actions, filter_popular_items
from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.mlp import GreedyMLP
from aprec.recommenders.mlp_historical import GreedyMLPHistorical
from aprec.recommenders.gru_recommender import GRURecommender
from aprec.recommenders.mlp_historical_embedding import GreedyMLPHistoricalEmbedding
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.evaluation.metrics.precision import Precision
from aprec.evaluation.metrics.recall import Recall
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.sps import SPS
from aprec.evaluation.metrics.average_popularity_rank import AveragePopularityRank
from aprec.recommenders.svd import SvdRecommender
from aprec.recommenders.lightfm import LightFMRecommender
from tensorflow.keras.optimizers import Adam

DATASET = get_movielens_actions(min_rating=0.0)
USERS_FRACTION = 1.0 
MAX_TEST_ACTIONS_PER_USER = 5000

def top_recommender():
    return FilterSeenRecommender(TopRecommender())

def mlp():
    return FilterSeenRecommender(GreedyMLP(train_epochs=300))

def mlp_historical_embedding_lr():
    loss = 'lambdarank' 
    return FilterSeenRecommender(GreedyMLPHistoricalEmbedding(train_epochs=10000, loss=loss,
                                                              optimizer=Adam(), early_stop_epochs=100,
                                                              batch_size=1000, sigma=1.0, ndcg_at=40, test_actions_per_user=5000,
                                                              output_layer_activation='linear'))

def mlp_historical_embedding():
    loss = 'binary_crossentropy' 
    return FilterSeenRecommender(GreedyMLPHistoricalEmbedding(train_epochs=10000, early_stop_epochs=100,
                                                              test_actions_per_user=5,
                                                              ndcg_at=40, 
                                                              batch_size=1000,
                                                              loss=loss, optimizer=Adam()))

def mlp_historical():
    return FilterSeenRecommender(GreedyMLPHistorical(train_epochs=250))

def lstm():
    return FilterSeenRecommender(GRURecommender(train_epochs=250))

def svd_recommender(k):
    return FilterSeenRecommender(SvdRecommender(k))

def lightfm_recommender(k, loss):
    return FilterSeenRecommender(LightFMRecommender(k, loss))

RECOMMENDERS = {
    "top_recommender": top_recommender,
    "lightfm_30_WARP": lambda: lightfm_recommender(30, 'warp'),
    "lightfm_30_BPR": lambda: lightfm_recommender(30, 'bpr'),
    "svd_recommender_30": lambda: svd_recommender(30),
    "GreedyMLPHistoricalEmbeddingLambdarank": mlp_historical_embedding_lr,
    "GreedyMLPHistoricalEmbedding": mlp_historical_embedding,
}

FRACTIONS_TO_SPLIT = (0.85, )

dataset_for_metric = get_movielens_actions(min_rating=1.0)
METRICS = [Precision(5), NDCG(40), Recall(5), SPS(10), AveragePopularityRank(5, dataset_for_metric)]


