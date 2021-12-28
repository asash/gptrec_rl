import numpy as np
from aprec.evaluation.samplers.random_sampler import RandomTargetItemSampler
from aprec.losses.bpr import BPRLoss
from aprec.losses.top1 import TOP1Loss
from aprec.recommenders.dnn_sequential_recommender.models.sasrec.sasrec import SASRec
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.biased_percentage_splitter import BiasedPercentageSplitter
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.last_item_splitter import LastItemSplitter
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


def dnn(model_arch, loss, sequence_splitter, learning_rate=0.001, training_time_limit=3600):
    return DNNSequentialRecommender(train_epochs=10000, loss=loss,
                                                          model_arch=model_arch,
                                                          optimizer=Adam(learning_rate),
                                                          early_stop_epochs=100,
                                                          batch_size=128,
                                                          training_time_limit=training_time_limit,
                                                          eval_ndcg_at=10,
                                                          target_decay=1.0,
                                                          sequence_splitter=sequence_splitter
                                                          )

def salrec(loss, training_time_limit=3600, target_decay=1.0, seq_len = 100):
    return SalrecRecommender(loss=loss, target_decay=target_decay, max_history_len=seq_len, training_time_limit=training_time_limit,
                             train_epochs=10000, early_stop_epochs=1000, num_blocks=3, num_bottlenecks=1)


def vanilla_bert4rec(time_limit):
    recommender = VanillaBERT4Rec(training_time_limit=time_limit, num_train_steps=10000000)
    return recommender


recommenders = {
    "top": top_recommender, 
    "svd": lambda: svd_recommender(128), 
    "mf-bpr": lambda: lightfm_recommender(128, 'bpr'),
    "GRU4Rec-Lambdarank-Truncated:4000-bce_weight:0.975-TimeLimit:1h-splitbias:0.8-splitpct:0.2": lambda: dnn(
            GRU4Rec(),
            LambdaGammaRankLoss(pred_truncate_at=4000, bce_grad_weight=0.975),
            BiasedPercentageSplitter(0.2, 0.8)),

    "GRU4Rec-Lambdarank-Truncated:4000-TimeLimit:1h-splitbias:0.8-splitpct:0.2": lambda: dnn(
            GRU4Rec(),
            LambdaGammaRankLoss(pred_truncate_at=4000),
            BiasedPercentageSplitter(0.2, 0.8)),

    "GRU4Rec-Lambdarank-TimeLimit:1h-splitbias:0.8-splitpct:0.2": lambda: dnn(
            GRU4Rec(),
            LambdaGammaRankLoss(),
            BiasedPercentageSplitter(0.2, 0.8)),

    "GRU4Rec-BCE-TimeLimit:1h-splitbias:0.8-splitpct:0.2": lambda: dnn(
            GRU4Rec(),
            BCELoss(),
            BiasedPercentageSplitter(0.2, 0.8)),
            
    "GRU4Rec-BPR-TimeLimit:1h-splitbias:0.8-splitpct:0.2": lambda: dnn(
            GRU4Rec(),
            BPRLoss(pred_truncate=4000),
            BiasedPercentageSplitter(0.2, 0.8)),

    "GRU4Rec-BCE-TimeLimit:1h-lastitem": lambda: dnn(
            GRU4Rec(),
            BCELoss(),
            LastItemSplitter()),

    "GRU4Rec-Top1-TimeLimit:1h-lastitem": lambda: dnn(
            GRU4Rec(),
            TOP1Loss(),
            LastItemSplitter()),

    "GRU4Rec-BPR-TimeLimit:1h-lastitem": lambda: dnn(
            GRU4Rec(),
            BPRLoss(pred_truncate=4000, max_positives=1),
            LastItemSplitter()),

    "GRU4Rec-Top1-max-TimeLimit:1h-lastitem": lambda: dnn(
            GRU4Rec(),
            TOP1Loss(softmax_weighted=True),
            LastItemSplitter()),

    "GRU4Rec-BPR-max-TimeLimit:1h-lastitem": lambda: dnn(
            GRU4Rec(),
            BPRLoss(softmax_weighted=True, pred_truncate=4000, max_positives=1),
            LastItemSplitter()),           

    "GRU4Rec-Lambdarank-Truncated:4000-bce_weight:0.975-TimeLimit:1h-lastitem": lambda: dnn(
            GRU4Rec(),
            LambdaGammaRankLoss(pred_truncate_at=4000, bce_grad_weight=0.975),
            LastItemSplitter()),

    "GRU4Rec-Lambdarank-Truncated:4000-TimeLimit:1h-lastitem": lambda: dnn(
            GRU4Rec(),
            LambdaGammaRankLoss(pred_truncate_at=4000),
            LastItemSplitter()),

    "GRU4Rec-Lambdarank-TimeLimit:1h-splitbias:0.8-lastitem": lambda: dnn(
            GRU4Rec(),
            LambdaGammaRankLoss(),
            LastItemSplitter()),



    "Caser-Lambdarank-Truncated:4000-bce_weight:0.975-TimeLimit:1h-splitbias:0.8-splitpct:0.2": lambda: dnn(
            Caser(),
            LambdaGammaRankLoss(pred_truncate_at=4000, bce_grad_weight=0.975),
            BiasedPercentageSplitter(0.2, 0.8)),

    "Caser-Lambdarank-Truncated:4000-TimeLimit:1h-splitbias:0.8-splitpct:0.2": lambda: dnn(
            Caser(),
            LambdaGammaRankLoss(pred_truncate_at=4000),
            BiasedPercentageSplitter(0.2, 0.8)),

    "Caser-Lambdarank-TimeLimit:1h-splitbias:0.8-splitpct:0.2": lambda: dnn(
            Caser(),
            LambdaGammaRankLoss(),
            BiasedPercentageSplitter(0.2, 0.8)),

    "Caser-BCE-TimeLimit:1h-splitbias:0.8-splitpct:0.2": lambda: dnn(
            Caser(),
            BCELoss(),
            BiasedPercentageSplitter(0.2, 0.8)),
            
    "Caser-BPR-TimeLimit:1h-splitbias:0.8-splitpct:0.2": lambda: dnn(
            Caser(),
            BPRLoss(pred_truncate=4000),
            BiasedPercentageSplitter(0.2, 0.8)),

    "Caser-BCE-TimeLimit:1h-lastitem": lambda: dnn(
            Caser(),
            BCELoss(),
            LastItemSplitter()),

    "Caser-Top1-TimeLimit:1h-lastitem": lambda: dnn(
            Caser(),
            TOP1Loss(),
            LastItemSplitter()),

    "Caser-BPR-TimeLimit:1h-lastitem": lambda: dnn(
            Caser(),
            BPRLoss(pred_truncate=4000, max_positives=1),
            LastItemSplitter()),

    "Caser-Top1-max-TimeLimit:1h-lastitem": lambda: dnn(
            Caser(),
            TOP1Loss(softmax_weighted=True),
            LastItemSplitter()),

    "Caser-BPR-max-TimeLimit:1h-lastitem": lambda: dnn(
            Caser(),
            BPRLoss(softmax_weighted=True, pred_truncate=4000, max_positives=1),
            LastItemSplitter()),           

    "Caser-Lambdarank-Truncated:4000-bce_weight:0.975-TimeLimit:1h-lastitem": lambda: dnn(
            Caser(),
            LambdaGammaRankLoss(pred_truncate_at=4000, bce_grad_weight=0.975),
            LastItemSplitter()),

    "Caser-Lambdarank-Truncated:4000-TimeLimit:1h-lastitem": lambda: dnn(
            Caser(),
            LambdaGammaRankLoss(pred_truncate_at=4000),
            LastItemSplitter()),

    "Caser-Lambdarank-TimeLimit:1h-splitbias:0.8-lastitem": lambda: dnn(
            Caser(),
            LambdaGammaRankLoss(),
            LastItemSplitter()),

    "SASRec-Lambdarank-Truncated:4000-bce_weight:0.975-TimeLimit:1h-splitbias:0.8-splitpct:0.2": lambda: dnn(
            SASRec(),
            LambdaGammaRankLoss(pred_truncate_at=4000, bce_grad_weight=0.975),
            BiasedPercentageSplitter(0.2, 0.8)),

    "SASRec-Lambdarank-Truncated:4000-TimeLimit:1h-splitbias:0.8-splitpct:0.2": lambda: dnn(
            SASRec(),
            LambdaGammaRankLoss(pred_truncate_at=4000),
            BiasedPercentageSplitter(0.2, 0.8)),

    "SASRec-Lambdarank-TimeLimit:1h-splitbias:0.8-splitpct:0.2": lambda: dnn(
            SASRec(),
            LambdaGammaRankLoss(),
            BiasedPercentageSplitter(0.2, 0.8)),

    "SASRec-BCE-TimeLimit:1h-splitbias:0.8-splitpct:0.2": lambda: dnn(
            SASRec(),
            BCELoss(),
            BiasedPercentageSplitter(0.2, 0.8)),
            
    "SASRec-BPR-TimeLimit:1h-splitbias:0.8-splitpct:0.2": lambda: dnn(
            SASRec(),
            BPRLoss(pred_truncate=4000),
            BiasedPercentageSplitter(0.2, 0.8)),

    "SASRec-BCE-TimeLimit:1h-lastitem": lambda: dnn(
            SASRec(),
            BCELoss(),
            LastItemSplitter()),

    "SASRec-Top1-TimeLimit:1h-lastitem": lambda: dnn(
            SASRec(),
            TOP1Loss(),
            LastItemSplitter()),

    "SASRec-BPR-TimeLimit:1h-lastitem": lambda: dnn(
            SASRec(),
            BPRLoss(pred_truncate=4000, max_positives=1),
            LastItemSplitter()),

    "SASRec-Top1-max-TimeLimit:1h-lastitem": lambda: dnn(
            SASRec(),
            TOP1Loss(softmax_weighted=True),
            LastItemSplitter()),

    "SASRec-BPR-max-TimeLimit:1h-lastitem": lambda: dnn(
            SASRec(),
            BPRLoss(softmax_weighted=True, pred_truncate=4000, max_positives=1),
            LastItemSplitter()),           

    "SASRec-Lambdarank-Truncated:4000-bce_weight:0.975-TimeLimit:1h-lastitem": lambda: dnn(
            SASRec(),
            LambdaGammaRankLoss(pred_truncate_at=4000, bce_grad_weight=0.975),
            LastItemSplitter()),

    "SASRec-Lambdarank-Truncated:4000-TimeLimit:1h-lastitem": lambda: dnn(
            SASRec(),
            LambdaGammaRankLoss(pred_truncate_at=4000),
            LastItemSplitter()),

    "SASRec-Lambdarank-TimeLimit:1h-splitbias:0.8-lastitem": lambda: dnn(
            SASRec(),
            LambdaGammaRankLoss(),
            LastItemSplitter()),
    "ber4rec-1h": lambda: vanilla_bert4rec(3600),
    "ber4rec-16h": lambda: vanilla_bert4rec(3600*16)
}
for i in range(0):
    loss_type = np.random.choice(["top1max", 'bce', 'lambdarank'])

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
