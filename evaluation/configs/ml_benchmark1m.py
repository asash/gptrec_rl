import random

from aprec.datasets.bert4rec_datasets import get_bert4rec_dataset
from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.svd import SvdRecommender
from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from tensorflow.keras.optimizers import Adam
from aprec.evaluation.metrics.hit import HIT
from aprec.losses.lambda_gamma_rank import LambdaGammaRankLoss
from aprec.losses.bce import BCELoss
from aprec.recommenders.dnn_sequential_recommender.models.caser import Caser
from aprec.recommenders.dnn_sequential_recommender.models.gru4rec import GRU4Rec
from aprec.evaluation.split_actions import LeaveOneOut
from aprec.recommenders.lightfm import LightFMRecommender

DATASET = get_bert4rec_dataset("ml-1m")

USERS_FRACTIONS = [1.]

def top_recommender():
    return FilterSeenRecommender(TopRecommender())

def svd_recommender(k):
    return FilterSeenRecommender(SvdRecommender(k))


def lightfm_recommender(k, loss):
    return FilterSeenRecommender(LightFMRecommender(k, loss, num_threads=32))

def dnn(model_arch, loss,learning_rate=0.001, last_only=False):
    return FilterSeenRecommender(DNNSequentialRecommender(train_epochs=10000, loss=loss,
                                                          model_arch=model_arch,
                                                          optimizer=Adam(learning_rate),
                                                          early_stop_epochs=100,
                                                          batch_size=128,
                                                          training_time_limit = 3600,
                                                          eval_ndcg_at=40,
                                                          target_decay=1.0,
                                                          train_on_last_item_only=last_only
                                                          ))

recommenders_raw = {
    "CASER-nouid-BCE": lambda: dnn(Caser(requires_user_id=False), BCELoss()),
    "CASER-nouid-Lambdarank-Vanilla": lambda: dnn(Caser(requires_user_id=False), LambdaGammaRankLoss()),
    "CASER-nouid-Lambdarank-Truncated:2500": lambda: dnn(Caser(requires_user_id=False), LambdaGammaRankLoss(pred_truncate_at=2500)),
    "CASER-nouid-Lambdarank-Truncated:2500-bce-weight:0.975": lambda: dnn(Caser(requires_user_id=False),
                                                     LambdaGammaRankLoss(pred_truncate_at=2500, bce_grad_weight=0.975)),

    "CASER-lastonly-nouid-BCE": lambda: dnn(Caser(requires_user_id=False), BCELoss(), last_only=True),
    "CASER-lastonly-nouid-Lambdarank-Vanilla": lambda: dnn(Caser(requires_user_id=False), LambdaGammaRankLoss(), last_only=True),
    "CASER-lastonly-nouid-Lambdarank-Truncated:2500": lambda: dnn(Caser(requires_user_id=False),
                                                         LambdaGammaRankLoss(pred_truncate_at=2500), last_only=True),
    "CASER-lastonly-nouid-Lambdarank-Truncated:2500-bce-weight:0.975": lambda: dnn(Caser(requires_user_id=False),
                                                                          LambdaGammaRankLoss(pred_truncate_at=2500,
                                                                                              bce_grad_weight=0.975), last_only=True),

    "CASER-lastonly-BCE": lambda: dnn(Caser(), BCELoss(), last_only=True),
    "CASER-lastonly-Lambdarank-Vanilla": lambda: dnn(Caser(), LambdaGammaRankLoss(),
                                                           last_only=True),
    "CASER-lastonly-Lambdarank-Truncated:2500": lambda: dnn(Caser(),
                                                                  LambdaGammaRankLoss(pred_truncate_at=2500),
                                                                  last_only=True),
    "CASER-lastonly-Lambdarank-Truncated:2500-bce-weight:0.975": lambda: dnn(Caser(),
                                                                                   LambdaGammaRankLoss(
                                                                                       pred_truncate_at=2500,
                                                                                       bce_grad_weight=0.975),
                                                                                   last_only=True),


    "GRU4Rec-lastonly-BCE": lambda: dnn(GRU4Rec(), BCELoss(), last_only=True),
    "GRU4Rec-lastonly-Lambdarank-Vanilla": lambda: dnn(GRU4Rec(), LambdaGammaRankLoss(), last_only=True),
    "GRU4Rec-lastonly-Lambdarank-Truncated:2500": lambda: dnn(GRU4Rec(),
                                                            LambdaGammaRankLoss(pred_truncate_at=2500),
                                                            last_only=True),
    "GRU4Rec-lastonly-Lambdarank-Truncated:2500-bce-weight:0.975": lambda: dnn(GRU4Rec(),
                                                                             LambdaGammaRankLoss(
                                                                                 pred_truncate_at=2500,
                                                                                 bce_grad_weight=0.975),
                                                                             last_only=True),


    "GRU4Rec-BCE": lambda: dnn(GRU4Rec(), BCELoss()),
    "GRU4Rec-Lambdarank-Vanilla": lambda: dnn(GRU4Rec(), LambdaGammaRankLoss()),
    "GRU4Rec-Lambdarank-Truncated:2500": lambda: dnn(GRU4Rec(), LambdaGammaRankLoss(pred_truncate_at=2500)),
    "GRU4Rec-Lambdarank-Truncated:2500-bce-weight:0.975": lambda: dnn(GRU4Rec(),
                                                                               LambdaGammaRankLoss(
                                                                                   pred_truncate_at=2500,
                                                                                   bce_grad_weight=0.975))
}

all_recommenders = list(recommenders_raw.keys())
random.shuffle(all_recommenders)


RECOMMENDERS = {
        "top_recommender": top_recommender,
        "svd_recommender": lambda: svd_recommender(128),
        "lightfm_recommender": lambda: lightfm_recommender(128, 'bpr')

}
for model in all_recommenders:
    RECOMMENDERS[model] = recommenders_raw[model]

print(f"evaluating {len(RECOMMENDERS)} models")

N_VAL_USERS=256
MAX_TEST_USERS=6040

SAMPLED_METRICS_ON=101
METRICS = [HIT(1), HIT(5), HIT(10), NDCG(5), NDCG(10), MRR()]

SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
