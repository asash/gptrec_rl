import numpy as np
from aprec.evaluation.configs.top_recommender_config import TARGET_ITEMS_SAMPLER
from aprec.evaluation.samplers.pop_sampler import PopTargetItemsSampler

from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.svd import SvdRecommender
from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
from aprec.recommenders.lightfm import LightFMRecommender
from aprec.recommenders.vanilla_bert4rec import VanillaBERT4Rec
from aprec.losses.bce import BCELoss
from aprec.recommenders.dnn_sequential_recommender.models.sasrec import SASRec
from aprec.recommenders.dnn_sequential_recommender.models.gru4rec import GRU4Rec
from aprec.recommenders.dnn_sequential_recommender.models.caser import Caser

from aprec.losses.lambda_gamma_rank import LambdaGammaRankLoss


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


def dnn(model_arch, loss, learning_rate=0.001, last_only=False, training_time_limit=3600):
    return DNNSequentialRecommender(train_epochs=10000, loss=loss,
                                                          model_arch=model_arch,
                                                          optimizer=Adam(learning_rate),
                                                          early_stop_epochs=100,
                                                          batch_size=128,
                                                          training_time_limit=training_time_limit,
                                                          eval_ndcg_at=40,
                                                          target_decay=1.0,
                                                          train_on_last_item_only=last_only
                                                          )


def vanilla_bert4rec(time_limit):
    recommender = VanillaBERT4Rec(training_time_limit=time_limit, num_train_steps=10000000)
    return recommender


recommenders = {
    "top": top_recommender,
    "lightfm-bpr": lambda: lightfm_recommender(128, 'bpr'),
    "SASRec-BCE-TimeLimit:1h-lastonly:True": lambda: dnn(
            SASRec(), BCELoss(), last_only=True),
    "SASRec-Lambdarank-Full-TimeLimit:1h-lastonly:True": lambda: dnn(
        SASRec(), LambdaGammaRankLoss(), last_only=True),
    "SASRec-Lambdarank-Truncated:2500-TimeLimit:1h-lastonly:True": lambda: dnn(
        SASRec(), LambdaGammaRankLoss(pred_truncate_at=2500), last_only=True),
    "SASRec-Lambdarank-Truncated:2500-bce_weight:0.975-TimeLimit:1h-lastonly:True": lambda: dnn(
        SASRec(), LambdaGammaRankLoss(pred_truncate_at=2500, bce_grad_weight=0.975), last_only=True),

    "GRU4Rec-BCE-TimeLimit:1h-lastonly:True": lambda: dnn(
        GRU4Rec(), BCELoss(), last_only=True),
    "GRU4Rec-Lambdarank-Full-TimeLimit:1h-lastonly:True": lambda: dnn(
        GRU4Rec(), LambdaGammaRankLoss(), last_only=True),
    "GRU4Rec-Lambdarank-Truncated:2500-TimeLimit:1h-lastonly:True": lambda: dnn(
        GRU4Rec(), LambdaGammaRankLoss(pred_truncate_at=2500), last_only=True),
    "GRU4Rec-Lambdarank-Truncated:2500-bce_weight:0.975-TimeLimit:1h-lastonly:True": lambda: dnn(
        GRU4Rec(), LambdaGammaRankLoss(pred_truncate_at=2500, bce_grad_weight=0.975), last_only=True),

    "Caser-BCE-TimeLimit:1h-lastonly:True": lambda: dnn(
        Caser(), BCELoss(), last_only=True),
    "Caser-Lambdarank-Full-TimeLimit:1h-lastonly:True": lambda: dnn(
        Caser(), LambdaGammaRankLoss(), last_only=True),
    "Caser-Lambdarank-Truncated:2500-TimeLimit:1h-lastonly:True": lambda: dnn(
        Caser(), LambdaGammaRankLoss(pred_truncate_at=2500), last_only=True),
    "Caser-Lambdarank-Truncated:2500-bce_weight:0.975-TimeLimit:1h-lastonly:True": lambda: dnn(
        Caser(), LambdaGammaRankLoss(pred_truncate_at=2500, bce_grad_weight=0.975), last_only=True),
    "bert4rec-1h": lambda: vanilla_bert4rec(3600),

    "svd": lambda: svd_recommender(128),
}
for i in range(0):
    dropout_rate = float(np.random.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    num_blocks = int(np.random.choice([1, 2, 3, 4, 5]))
    num_heads = int(np.random.choice([1, 2, 4, 8, 16, 32]))
    emb_size = int(np.random.choice([32, 64, 128, 256, 512]))
    seq_len = int(np.random.choice([4, 6, 8, 16, 32, 64]))


    model = SASRec(num_blocks=num_blocks,
                   num_heads=num_heads,
                   embedding_size=emb_size,
                   dropout_rate=dropout_rate,
                   max_history_len=seq_len)

    model_name = f"SasRec-blocks:{num_blocks}" \
                 f"-heads:{num_heads}-seq_len:{seq_len}-dropout:{dropout_rate}" \
                 f"-dim:{emb_size}"

    bce_weight = np.random.choice([0.0, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9995, 0.9999])
    truncation = int(np.random.choice([100, 200, 400, 800, 1500, 2500, 5000, 10000]))
    loss = LambdaGammaRankLoss(pred_truncate_at=truncation, bce_grad_weight=bce_weight)
    loss_name=f"Lambdarank-trunc:{truncation}-bce_weight:{bce_weight}"

    training_time_minutes = int(np.random.choice([5, 10, 15]))
    last_only = bool(np.random.choice([True, False]))
    training_properties=f"TimeLimit:{training_time_minutes}m-lastonly:{last_only}`"
    recommender_name = "-".join([model_name, loss_name, training_properties])

    recommenders[recommender_name] = lambda model=model, loss=loss, \
                                            last_only=last_only, \
                                            training_time_minutes=training_time_minutes: dnn(model_arch=model,
                                                                     loss=loss, last_only=last_only,
                                                                     training_time_limit=training_time_minutes*60)


METRICS = [HIT(1), HIT(5), HIT(10), NDCG(5), NDCG(10), MRR()]
TARGET_ITEMS_SAMPLER = PopTargetItemsSampler(101)

def get_recommenders(filter_seen: bool):
    result = {}
    for recommender_name in recommenders:
        if filter_seen:
            result[recommender_name] =\
                lambda recommender_name=recommender_name: FilterSeenRecommender(recommenders[recommender_name]())
        else:
            result[recommender_name] = recommenders[recommender_name]
    return result
