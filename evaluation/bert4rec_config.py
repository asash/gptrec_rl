from aprec.datasets.movielens import get_movielens_actions
from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.svd import SvdRecommender
from aprec.recommenders.lightfm import LightFMRecommender
from aprec.recommenders.constant_recommender import ConstantRecommender
from aprec.recommenders.mlp_historical_embedding import GreedyMLPHistoricalEmbedding
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.recommenders.vanilla_bert4rec import VanillaBERT4Rec
from aprec.evaluation.metrics.precision import Precision
from aprec.evaluation.metrics.recall import Recall
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from tensorflow.keras.optimizers import Adam
from aprec.evaluation.metrics.average_popularity_rank import AveragePopularityRank
from aprec.evaluation.metrics.pairwise_cos_sim import PairwiseCosSim
from aprec.evaluation.metrics.sps import SPS

DATASET = get_movielens_actions(min_rating=1.0)

USERS_FRACTIONS = [.1]

def top_recommender():
    return FilterSeenRecommender(TopRecommender())

def svd_recommender(k):
    return FilterSeenRecommender(SvdRecommender(k))

def lightfm_recommender(k, loss):
    return FilterSeenRecommender(LightFMRecommender(k, loss))

def vanilla_bert4rec():
    max_seq_length = 200
    masked_lm_prob = 0.2
    max_predictions_per_seq = 40
    mask_prob = 1.0
    dupe_factor = 10
    pool_size = 10
    prop_sliding_window = 0.5
    num_warmup_steps = 100
    num_train_steps = 400000
    batch_size = 256
    learning_rate = 1e-4
    random_seed = 31337

    bert_config = {
        "attention_probs_dropout_prob": 0.2,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.2,
        "hidden_size": 64,
        "initializer_range": 0.02,
        "intermediate_size": 256,
        "max_position_embeddings": 200,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "type_vocab_size": 2,
    }
    recommender = VanillaBERT4Rec(max_seq_length=max_seq_length,
                              masked_lm_prob=masked_lm_prob,
                              max_predictions_per_seq=max_predictions_per_seq,
                              mask_prob=mask_prob,
                              dupe_factor=dupe_factor,
                              pool_size=pool_size,
                              prop_sliding_window=prop_sliding_window,
                              random_seed=random_seed,
                              bert_config=bert_config,
                              num_warmup_steps=num_warmup_steps,
                              num_train_steps=num_train_steps,
                              batch_size=batch_size,
                              learning_rate=learning_rate)
    return FilterSeenRecommender(recommender)


def mlp_historical_embedding(loss, activation_override=None):
    activation = 'linear' if loss == 'lambdarank' else 'sigmoid'
    if activation_override is not None:
        activation = activation_override
    return FilterSeenRecommender(GreedyMLPHistoricalEmbedding(train_epochs=10000, loss=loss,
                                                              optimizer=Adam(), early_stop_epochs=100,
                                                              batch_size=150, sigma=1.0, ndcg_at=40,
                                                              n_val_users=600,
                                                              bottleneck_size=64,
                                                              output_layer_activation=activation))

def constant_recommender():
    return ConstantRecommender([('457', 0.45),
                               ('380', 0.414),
                               ('110', 0.413),
                               ('292', 0.365),
                               ('296', 0.323),
                               ('595', 0.313), 
                               ('588', 0.312), 
                               ('592', 0.293),
                               ('440', 0.286),
                               ('357', 0.286),
                               ('434', 0.280),
                               ('593', 0.280),
                               ('733', 0.276),
                               ('553', 0.257),
                               ('253', 0.257)])

RECOMMENDERS = {
   # "APREC-GMLPHE-Lambdarank": lambda: mlp_historical_embedding('lambdarank'),
   # "APREC-GMLPHE-BCE": lambda: mlp_historical_embedding('binary_crossentropy'),
    "vanilla_bert4rec": vanilla_bert4rec,
    "top_recommender": top_recommender,
    "svd_recommender_30": lambda: svd_recommender(30),
    "lightfm_recommender_30_warp": lambda: lightfm_recommender(30, 'warp'),
    "constant_recommender": constant_recommender,
}

FRACTION_TO_SPLIT = 0.85

dataset_for_metric = [action for action in get_movielens_actions(min_rating=1.0)]
METRICS = [Precision(5), NDCG(40), Recall(5), SPS(10), MRR(), MAP(10), AveragePopularityRank(10, dataset_for_metric),
           PairwiseCosSim(dataset_for_metric, 10)]
del(dataset_for_metric)


SPLIT_STRATEGY = "LEAVE_ONE_OUT"
