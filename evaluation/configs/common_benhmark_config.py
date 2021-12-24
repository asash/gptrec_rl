from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.svd import SvdRecommender
from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
from aprec.recommenders.lightfm import LightFMRecommender
from aprec.recommenders.vanilla_bert4rec import VanillaBERT4Rec
from aprec.losses.bce import BCELoss
from aprec.recommenders.dnn_sequential_recommender.models.sasrec import SASRec

from aprec.losses.lambda_gamma_rank import LambdaGammaRankLoss
from aprec.recommenders.dnn_sequential_recommender.models.caser import Caser
from aprec.recommenders.dnn_sequential_recommender.models.gru4rec import GRU4Rec


from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.hit import HIT

from tensorflow.keras.optimizers import Adam

from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender

USERS_FRACTIONS = [1.]

def top_recommender():
    return TopRecommender()


def svd_recommender(k):
    return SvdRecommender(k)


def lightfm_recommender(k, loss):
    return LightFMRecommender(k, loss, num_threads=32)


def dnn(model_arch, loss, learning_rate=0.001, last_only=False):
    return DNNSequentialRecommender(train_epochs=10000, loss=loss,
                                                          model_arch=model_arch,
                                                          optimizer=Adam(learning_rate),
                                                          early_stop_epochs=100,
                                                          batch_size=128,
                                                          training_time_limit=600,
                                                          eval_ndcg_at=40,
                                                          target_decay=1.0,
                                                          train_on_last_item_only=last_only
                                                          )


def vanilla_bert4rec(time_limit):
    recommender = VanillaBERT4Rec(training_time_limit=time_limit, num_train_steps=10000000)
    return recommender


recommenders = {
    "SASRec-Lambdarank-Truncated:2500-bce-weight:0.975": lambda: dnn(
        SASRec(), LambdaGammaRankLoss(pred_truncate_at=2500, bce_grad_weight=0.975)),
    "SASRec-BCE": lambda: dnn(
        SASRec(), BCELoss()),
    "SASRec-lastonly-BCE": lambda: dnn(
        SASRec(), BCELoss(), last_only=True),
    "SASRec-Lambdarank-Truncated:2500": lambda: dnn(SASRec(),
                                                    LambdaGammaRankLoss(pred_truncate_at=2500)),

    "top_recommender": top_recommender,
    "svd_recommender": lambda: svd_recommender(128),
    "lightfm_recommender": lambda: lightfm_recommender(128, 'bpr'),

    "CASER-nouid-Lambdarank-Truncated:2500": lambda: dnn(Caser(requires_user_id=False),
                                                         LambdaGammaRankLoss(pred_truncate_at=2500)),
    "CASER-nouid-Lambdarank-Truncated:2500-bce-weight:0.975": lambda: dnn(
        Caser(requires_user_id=False), LambdaGammaRankLoss(pred_truncate_at=2500, bce_grad_weight=0.975)),

    "GRU4Rec-Lambdarank-Truncated:2500": lambda: dnn(GRU4Rec(), LambdaGammaRankLoss(pred_truncate_at=2500)),
    "GRU4Rec-Lambdarank-Truncated:2500-bce-weight:0.975": lambda: dnn(GRU4Rec(), LambdaGammaRankLoss(
        pred_truncate_at=2500,
        bce_grad_weight=0.975)),

}

METRICS = [HIT(1), HIT(5), HIT(10), NDCG(5), NDCG(10), MRR()]
SAMPLED_METRICS_ON=101

def get_recommenders(filter_seen: bool):
    result = {}
    for recommender_name in recommenders:
        if filter_seen:
            result[recommender_name] = lambda: FilterSeenRecommender(recommenders[recommender_name]())
        else:
            result[recommender_name] = recommenders[recommender_name]
    return result
