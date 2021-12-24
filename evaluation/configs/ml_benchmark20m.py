from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.svd import SvdRecommender
from aprec.recommenders.lightfm import LightFMRecommender
from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.evaluation.metrics.precision import Precision
from aprec.evaluation.metrics.recall import Recall
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from tensorflow.keras.optimizers import Adam
from aprec.evaluation.metrics.hit import HIT
from aprec.losses.lambda_gamma_rank import LambdaGammaRankLoss
from aprec.losses.bce import BCELoss
from aprec.recommenders.dnn_sequential_recommender.models.gru4rec import GRU4Rec
from aprec.evaluation.split_actions import LeaveOneOut
from aprec.recommenders.dnn_sequential_recommender.models.caser import Caser
from aprec.recommenders.vanilla_bert4rec import VanillaBERT4Rec

DATASET = "ml-20m"


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

def vanilla_bert4rec(time_limit):
    recommender = VanillaBERT4Rec(training_time_limit=time_limit, num_train_steps=10000000)
    return FilterSeenRecommender(recommender)

recommenders_raw = {
    "BERT4rec-1h": lambda: vanilla_bert4rec(3600),
    "svd_recommender": lambda: svd_recommender(128), 
    "top_recommender": top_recommender, 
    "lightfm_recommender": lambda: lightfm_recommender(128, 'bpr'), 


    "CASER-nouid-lastonly-BCE": lambda: dnn(Caser(requires_user_id=False), BCELoss(), last_only=True),
    "CASER-nouid-BCE": lambda: dnn(Caser(requires_user_id=False), BCELoss()),
    "CASER-nouid-Lambdarank-Truncated:2500": lambda: dnn(Caser(requires_user_id=False), LambdaGammaRankLoss(pred_truncate_at=2500)),

    "CASER-nouid-Lambdarank-Truncated:2500-bce-weight:0.975": lambda: dnn(
        Caser(requires_user_id=False), LambdaGammaRankLoss(pred_truncate_at=2500, bce_grad_weight=0.975)),


    "GRU4Rec-BCE": lambda: dnn(GRU4Rec(), BCELoss()),
    "GRU4Rec-lastonly-BCE": lambda: dnn(GRU4Rec(), BCELoss(), last_only=True),
    "GRU4Rec-Lambdarank-Truncated:2500": lambda: dnn(GRU4Rec(), LambdaGammaRankLoss(pred_truncate_at=2500)),
    "GRU4Rec-Lambdarank-Truncated:2500-bce-weight:0.975": lambda: dnn(GRU4Rec(), LambdaGammaRankLoss(
                                                                                        pred_truncate_at=2500,
                                                                                        bce_grad_weight=0.975)),
    "BERT4rec-8h": lambda: vanilla_bert4rec(4*3600)
}


all_recommenders = list(recommenders_raw.keys())


RECOMMENDERS = {
    }
for model in all_recommenders:
    RECOMMENDERS[model] = recommenders_raw[model]

print(f"evaluating {len(RECOMMENDERS)} models")

N_VAL_USERS=1024
MAX_TEST_USERS=4096

METRICS = [NDCG(10), NDCG(2), NDCG(5), NDCG(20), NDCG(40), Precision(10), Recall(10), HIT(1), HIT(10), MRR(), MAP(10)]


SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
