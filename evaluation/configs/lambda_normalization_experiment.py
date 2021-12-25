import random

from aprec.losses.lambda_gamma_rank import LambdaGammaRankLoss
from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.svd import SvdRecommender
from aprec.recommenders.salrec.salrec_recommender import SalrecRecommender
from aprec.recommenders.lightfm import LightFMRecommender
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.recommenders.vanilla_bert4rec import VanillaBERT4Rec
from aprec.evaluation.metrics.precision import Precision
from aprec.evaluation.metrics.recall import Recall
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from tensorflow.keras.optimizers import Adam
from aprec.evaluation.metrics.hit import HIT
from aprec.evaluation.split_actions import LeaveOneOut

DATASET = "ml-20m"


USERS_FRACTIONS = [1.]

def top_recommender():
    return FilterSeenRecommender(TopRecommender())

def svd_recommender(k):
    return FilterSeenRecommender(SvdRecommender(k))

def lightfm_recommender(k, loss):
    return FilterSeenRecommender(LightFMRecommender(k, loss))

def vanilla_bert4rec(num_steps):
    recommender = VanillaBERT4Rec(num_train_steps=num_steps)
    return FilterSeenRecommender(recommender)

def salrec(loss, num_blocks, learning_rate,
                session_len, activation_override=None,
                log_lambdas=False
           ):
    activation = 'linear' if loss == 'lambdarank' else 'sigmoid'
    if activation_override is not None:
        activation = activation_override
    return FilterSeenRecommender(SalrecRecommender(train_epochs=10000, loss=loss,
                                                   optimizer=Adam(learning_rate), 
                                                   early_stop_epochs=300,
                                                   batch_size=128,
                                                   max_history_len=session_len,
                                                   output_layer_activation=activation,
                                                   training_time_limit = 3600,
                                                   num_blocks=num_blocks,
                                                   num_target_predictions=5,
                                                   eval_ndcg_at=40,
                                                   target_decay=0.8, 
                                                   log_lambdas_len=log_lambdas
                                                   ))

recommenders_raw = {
    "Transformer-Lambdarank-blocks:3-lr:0.001-ndcg:50-session_len:100-lambda_norm:True-truncate:2500-bce_weight:0.975":
        lambda: salrec(LambdaGammaRankLoss(pred_truncate_at=2500, bce_grad_weight=0.975, lambda_normalization=True), 3, 0.001, 100, True),

    "Transformer-Lambdarank-blocks:3-lr:0.001-ndcg:50-session_len:100-lambda_norm:False-truncate:2500-bce_weight:0.975":
        lambda: salrec(LambdaGammaRankLoss(pred_truncate_at=2500, bce_grad_weight=0.975, lambda_normalization=False), 3, 0.001, 100, True),

}

all_recommenders = list(recommenders_raw.keys())
random.shuffle(all_recommenders)


RECOMMENDERS = {}
for model in all_recommenders:
    RECOMMENDERS[model] = recommenders_raw[model]

print(f"evaluating {len(RECOMMENDERS)} models")

N_VAL_USERS=1024
MAX_TEST_USERS=65536

METRICS = [NDCG(10), NDCG(2), NDCG(5), NDCG(20), NDCG(40), Precision(10), Recall(10), HIT(1), HIT(10), MRR(), MAP(10)]


SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
