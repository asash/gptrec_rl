from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.svd import SvdRecommender
from aprec.evaluation.metrics.precision import Precision
from aprec.evaluation.metrics.recall import Recall
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.average_popularity_rank import AveragePopularityRank
from aprec.evaluation.metrics.pairwise_cos_sim import PairwiseCosSim
from aprec.recommenders.booking_recommender.booking_recommender import BookingRecommender
from tensorflow.keras.optimizers import Adam
from aprec.evaluation.metrics.sps import SPS
from aprec.datasets.booking import get_booking_dataset


DATASET = get_booking_dataset('./booking_data/booking_train_set.csv')

def top_recommender():
    return TopRecommender()

def svd_recommender(k):
    return SvdRecommender(k)

def mlp_historical_embedding(loss, activation_override=None):
    activation = 'linear' if loss == 'lambdarank' else 'sigmoid'
    if activation_override is not None:
        activation = activation_override
    return BookingRecommender(train_epochs=10000, loss=loss,
                                        optimizer=Adam(), early_stop_epochs=100,
                                        batch_size=250, sigma=1.0, ndcg_at=40,
                                        n_val_users=2000,
                                        bottleneck_size=64,
                                        max_history_len=180,
                                        output_layer_activation=activation)

RECOMMENDERS = {
    "top_recommender": top_recommender,
    "svd_recommender": lambda: svd_recommender(30),
    "APREC-GMLPHE-Lambdarank": lambda: mlp_historical_embedding('lambdarank', 'linear'),

}



SPLIT_STRATEGY = "LEAVE_ONE_OUT"

USERS_FRACTIONS = [1.0]

dataset_for_metric = [action for action in get_booking_dataset('./booking_data/booking_train_set.csv')]
METRICS = [Precision(4), NDCG(4), NDCG(40), Recall(5), SPS(10), MRR(), MAP(10), AveragePopularityRank(5, dataset_for_metric),
           PairwiseCosSim(dataset_for_metric, 10)]
del(dataset_for_metric)
