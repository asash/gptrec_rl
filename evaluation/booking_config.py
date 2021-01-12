from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.conditional_top_recommender import ConditionalTopRecommender
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.recommenders.item_item import ItemItemRecommender
from aprec.recommenders.transition_chain_recommender import TransitionsChainRecommender
from aprec.recommenders.svd import SvdRecommender
from aprec.evaluation.metrics.precision import Precision
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.recommenders.booking_recommender.booking_recommender import BookingRecommender
from tensorflow.keras.optimizers import Adam
from aprec.evaluation.metrics.sps import SPS
from aprec.datasets.booking import get_booking_dataset


DATASET = get_booking_dataset('./booking_data/booking_train_set.csv')

def top_recommender():
    return TopRecommender()

def conditional_top_recommender():
    return ConditionalTopRecommender('hotel_country')

def filter_seen_recommender(recommender):
    return FilterSeenRecommender(recommender)

def svd_recommender(k):
    return SvdRecommender(k)

def item_item_recommender():
    return ItemItemRecommender()

def mlp_historical_embedding(loss, activation_override=None, bottleneck_size=64):
    activation = 'linear' if loss == 'lambdarank' else 'sigmoid'
    if activation_override is not None:
        activation = activation_override
    return BookingRecommender(train_epochs=10000, loss=loss,
                                        optimizer=Adam(), early_stop_epochs=20,
                                        batch_size=250, sigma=1.0, ndcg_at=40,
                                        n_val_users=2000,
                                        bottleneck_size=bottleneck_size,
                                        max_history_len=50,
                                        output_layer_activation=activation)

RECOMMENDERS = {
    "top_recommender": top_recommender,
    "conditional_top_recommender": conditional_top_recommender,
    "svd_recommender": lambda: svd_recommender(30),
    "item_temem_recommender": item_item_recommender,
    "transitions_chain_recommender": TransitionsChainRecommender,
    "APREC-GMLPHE-Lambdarank-128": lambda: mlp_historical_embedding('lambdarank', 'linear', 128),
    "APREC-GMLPHE-Lambdarank-256": lambda: mlp_historical_embedding('lambdarank', 'linear', 256),
    "APREC-GMLPHE-Lambdarank-512": lambda: mlp_historical_embedding('lambdarank', 'linear', 512),
    "APREC-GMLPHE-Lambdarank-1024": lambda: mlp_historical_embedding('lambdarank', 'linear', 1024),
}

SPLIT_STRATEGY = "LEAVE_ONE_OUT"

USERS_FRACTIONS = [1.0]

dataset_for_metric = [action for action in get_booking_dataset('./booking_data/booking_train_set.csv')]
METRICS = [Precision(4), SPS(4), NDCG(4), NDCG(40)]
del(dataset_for_metric)
