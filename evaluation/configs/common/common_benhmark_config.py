import numpy as np
from aprec.evaluation.samplers.random_sampler import RandomTargetItemSampler
from aprec.losses.bpr import BPRLoss
from aprec.losses.top1 import TOP1Loss
from aprec.recommenders.dnn_sequential_recommender.models.sasrec.sasrec import SASRec
from aprec.recommenders.salrec.salrec_recommender import SalrecRecommender

from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.svd import SvdRecommender
from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
from aprec.recommenders.lightfm import LightFMRecommender
from aprec.recommenders.vanilla_bert4rec import VanillaBERT4Rec
from aprec.losses.bce import BCELoss
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
                                                          eval_ndcg_at=10,
                                                          target_decay=1.0,
                                                          train_on_last_item_only=last_only
                                                          )

def salrec(loss, training_time_limit=3600, target_decay=1.0, seq_len = 100):
    return SalrecRecommender(loss=loss, target_decay=target_decay, max_history_len=seq_len, training_time_limit=training_time_limit,
                             train_epochs=10000, early_stop_epochs=1000, num_blocks=3, num_bottlenecks=1)


def vanilla_bert4rec(time_limit):
    recommender = VanillaBERT4Rec(training_time_limit=time_limit, num_train_steps=10000000)
    return recommender


recommenders = {
    "SASRec-noreuse-Lambdarank-Truncated:4000-bce_weight:0.975-TimeLimit:1h-lastonly:False": lambda: dnn(
            SASRec(reuse_item_embeddings=False, max_history_len=100),
            LambdaGammaRankLoss(pred_truncate_at=4000, bce_grad_weight=0.975), last_only=False),

    "SASRec-encodeoutput-Lambdarank-Truncated:4000-bce_weight:0.975-TimeLimit:1h-lastonly:False": lambda: dnn(
            SASRec(encode_output_embeddings=True, max_history_len=100),
            LambdaGammaRankLoss(pred_truncate_at=4000, bce_grad_weight=0.975), last_only=False),

    "SASRec-TOP1max-TimeLimit:1h-lastonly:True": lambda: dnn(
            SASRec(max_history_len=100),
            TOP1Loss(softmax_weighted=True), last_only=True),

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


METRICS = [HIT(1), HIT(5), HIT(10), NDCG(5), NDCG(10), MRR(), HIT(4), NDCG(40)]
TARGET_ITEMS_SAMPLER = RandomTargetItemSampler(101)

def get_recommenders(filter_seen: bool):
    result = {}
    for recommender_name in recommenders:
        if filter_seen:
            result[recommender_name] =\
                lambda recommender_name=recommender_name: FilterSeenRecommender(recommenders[recommender_name]())
        else:
            result[recommender_name] = recommenders[recommender_name]
    return result
