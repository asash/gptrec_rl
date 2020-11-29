from aprec.datasets.movielens import get_movielens_actions, filter_popular_items
from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.mlp_historical_embedding import GreedyMLPHistoricalEmbedding
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.evaluation.metrics.precision import Precision
from aprec.evaluation.metrics.recall import Recall
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.sps import SPS
from aprec.evaluation.metrics.average_popularity_rank import AveragePopularityRank
from aprec.evaluation.metrics.pairwise_cos_sim import PairwiseCosSim
from aprec.recommenders.svd import SvdRecommender
from aprec.recommenders.lightfm import LightFMRecommender
from tensorflow.keras.optimizers import Adam

DATASET = [action for action in get_movielens_actions(min_rating=0.0)]
USERS_FRACTIONS = [0.15]


def top_recommender():
    return FilterSeenRecommender(TopRecommender())

def mlp_historical_embedding(loss):
    activation = 'linear' if loss == 'lambdarank' else 'sigmoid'
    return FilterSeenRecommender(GreedyMLPHistoricalEmbedding(train_epochs=10000, loss=loss,
                                                              optimizer=Adam(), early_stop_epochs=100,
                                                              batch_size=200, sigma=1.0, ndcg_at=40,
                                                              n_val_users=200,
                                                              bottleneck_size=64,
                                                              output_layer_activation=activation))
def lightfm_recommender(k, loss):
    return FilterSeenRecommender(LightFMRecommender(k, loss))

def svd_recommender(k):
    return FilterSeenRecommender(SvdRecommender(num_latent_components=k))

RECOMMENDERS = {
    "top_recommender": top_recommender,
    "lightfm_30_WARP": lambda: lightfm_recommender(30, 'warp'),
    "lightfm_30_BPR": lambda: lightfm_recommender(30, 'bpr'),
    "svd_recommender_30": lambda: svd_recommender(30),
    "APREC-GMLPHE-Lambdarank": lambda: mlp_historical_embedding('lambdarank'),
    "APREC-GMLPHE-BCE": lambda: mlp_historical_embedding('binary_crossentropy'),
}

FRACTION_TO_SPLIT = 0.85

METRICS = [Precision(5), NDCG(40), Recall(5), SPS(10), AveragePopularityRank(10, DATASET), PairwiseCosSim(DATASET, 10)]


