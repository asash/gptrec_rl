from aprec.evaluation.split_actions import LeaveOneOut
from aprec.recommenders.dnn_sequential_recommender.history_vectorizers.add_mask_history_vectorizer import AddMaskHistoryVectorizer
from aprec.recommenders.dnn_sequential_recommender.history_vectorizers.default_history_vectorizer import DefaultHistoryVectrizer

from aprec.recommenders.dnn_sequential_recommender.models.sasrec.sasrec import SASRec
from aprec.recommenders.dnn_sequential_recommender.target_builders.full_matrix_targets_builder import FullMatrixTargetsBuilder
from aprec.recommenders.dnn_sequential_recommender.target_builders.negative_per_positive_target import NegativePerPositiveTargetBuilder
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.last_item_splitter import SequenceContinuation
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.shifted_sequence_splitter import ShiftedSequenceSplitter
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.recency_sequence_sampling import RecencySequenceSampling
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.recency_sequence_sampling import exponential_importance
from aprec.recommenders.vanilla_bert4rec import VanillaBERT4Rec
from aprec.losses.bce import BCELoss
from aprec.losses.lambda_gamma_rank import LambdaGammaRankLoss



from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.hit import HIT


from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender

USERS_FRACTIONS = [1.0]

def top_recommender():
     from aprec.recommenders.top_recommender import TopRecommender
     return TopRecommender()


def svd_recommender(k):
    from aprec.recommenders.svd import SvdRecommender
    return SvdRecommender(k)


def lightfm_recommender(k, loss):
    from aprec.recommenders.lightfm import LightFMRecommender
    return LightFMRecommender(k, loss, num_threads=32)



def vanilla_bert4rec(time_limit):
    recommender = VanillaBERT4Rec(training_time_limit=time_limit, num_train_steps=10000000)
    return recommender

HISTORY_LEN=50

def dnn(model_arch, loss, sequence_splitter, 
                val_sequence_splitter, 
                target_builder,
                training_time_limit=3600,  
                max_epochs=10000, 
                metric = None, 
                pred_history_vectorizer = DefaultHistoryVectrizer()):
    from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender

    from tensorflow.keras.optimizers import Adam
    from aprec.recommenders.metrics.ndcg import KerasNDCG
    if metric is None:
        metric=KerasNDCG(40)
    optimizer=Adam(beta_2=0.98)
    return DNNSequentialRecommender(train_epochs=max_epochs, loss=loss,
                                                          model_arch=model_arch,
                                                          optimizer=optimizer,
                                                          early_stop_epochs=max_epochs,
                                                          batch_size=256,
                                                          max_batches_per_epoch=48,
                                                          training_time_limit=training_time_limit,
                                                          sequence_splitter=sequence_splitter, 
                                                          targets_builder=target_builder, 
                                                          val_sequence_splitter = val_sequence_splitter,
                                                          metric=metric,
                                                          pred_history_vectorizer=pred_history_vectorizer,
                                                          debug=True)

def sasrec_rss(recency_importance, add_cls=False, pos_smoothing=0,
               pos_embedding='default', pos_embeddding_comb='add', 
               causal_attention = True
               ):
        target_splitter = lambda: RecencySequenceSampling(0.2, exponential_importance(recency_importance), add_cls=add_cls)
        val_splitter = lambda: SequenceContinuation(add_cls=add_cls)
        pred_history_vectorizer = AddMaskHistoryVectorizer() if add_cls else DefaultHistoryVectrizer()
        return dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False, num_heads=1, 
                   pos_embedding=pos_embedding,
                   pos_emb_comb=pos_embeddding_comb,
                   pos_smoothing=pos_smoothing, 
                   causal_attention=causal_attention,
                   embedding_size=64),
            LambdaGammaRankLoss(pred_truncate_at=4000),
            sequence_splitter=target_splitter,
            val_sequence_splitter=val_splitter,
            target_builder=FullMatrixTargetsBuilder, 
            pred_history_vectorizer=pred_history_vectorizer)

def vanilla_sasrec():
    model_arch = SASRec(max_history_len=HISTORY_LEN, vanilla=True, num_heads=1, embedding_size=64)

    return dnn(model_arch,  BCELoss(),
            ShiftedSequenceSplitter,
            target_builder=lambda: NegativePerPositiveTargetBuilder(HISTORY_LEN), 
            val_sequence_splitter=SequenceContinuation,
            metric=BCELoss())




recommenders = {
#    "Sasrec-rss-lambdarank-0.8-nocls-exp-mult-sm8": lambda: sasrec_rss(0.8, add_cls=False, pos_smoothing=8, pos_embedding='exp', pos_embeddding_comb='mult'),
    "Sasrec-rss-lambdarank-0.8-default-bidirectional": lambda: sasrec_rss(0.8, pos_embeddding_comb='add', pos_embedding='default', causal_attention=False),
    "Sasrec-rss-lambdarank-0.8-default-causal": lambda: sasrec_rss(0.8, pos_embeddding_comb='add', pos_embedding='default', causal_attention=True),
    "Sasrec-rss-vanilla": lambda: vanilla_sasrec(),
    # "Sasrec-rss-lambdarank-0.8-cls-exp-mult": lambda: sasrec_rss(0.8, add_cls=True, pos_embedding='exp', pos_embeddding_comb='mult'),
    #"Sasrec-rss-lambdarank-0.8-cls-sin-mult": lambda: sasrec_rss(0.8, add_cls=True, pos_embedding='sin', pos_embeddding_comb='mult'),
    #"Sasrec-rss-lambdarank-0.8-nocls-sin-mult": lambda: sasrec_rss(0.8, add_cls=False, pos_embedding='sin', pos_embeddding_comb='mult'),
    #"Sasrec-rss-lambdarank-0.8-cls-default-mult": lambda: sasrec_rss(0.8, add_cls=True, pos_embedding='default', pos_embeddding_comb='mult'),
    #"Sasrec-rss-lambdarank-0.8-nocls-default-mult": lambda: sasrec_rss(0.8, add_cls=False, pos_embedding='default', pos_embeddding_comb='mult'),
    #"Sasrec-rss-lambdarank-0.8-cls-exp-add": lambda: sasrec_rss(0.8, add_cls=True, pos_embedding='exp', pos_embeddding_comb='add'),
    #"Sasrec-rss-lambdarank-0.8-nocls-exp-add": lambda: sasrec_rss(0.8, add_cls=False, pos_embedding='exp', pos_embeddding_comb='add'),
    #"Sasrec-rss-lambdarank-0.8-cls-sin-add": lambda: sasrec_rss(0.8, add_cls=True, pos_embedding='sin', pos_embeddding_comb='add'),
    #"Sasrec-rss-lambdarank-0.8-nocls-sin-add": lambda: sasrec_rss(0.8, add_cls=False, pos_embedding='sin', pos_embeddding_comb='add'),
    #"Sasrec-rss-lambdarank-0.8-cls-default-add": lambda: sasrec_rss(0.8, add_cls=True, pos_embedding='default', pos_embeddding_comb='add'),
    #"Sasrec-rss-lambdarank-0.8-nocls-default-add": lambda: sasrec_rss(0.8, add_cls=False, pos_embedding='default', pos_embeddding_comb='add'),
}

METRICS = [HIT(1), HIT(5), HIT(10), NDCG(5), NDCG(10), MRR(), HIT(4), NDCG(40), MAP(10)]

def get_recommenders(filter_seen: bool, filter_recommenders = set()):
    result = {}
    for recommender_name in recommenders:
        if recommender_name in filter_recommenders:
                continue
        if filter_seen:
            result[recommender_name] =\
                lambda recommender_name=recommender_name: FilterSeenRecommender(recommenders[recommender_name]())
        else:
            result[recommender_name] = recommenders[recommender_name]
    return result

DATASET = "yelp_warm5"
N_VAL_USERS=1024
MAX_TEST_USERS=138493
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
RECOMMENDERS = get_recommenders(filter_seen=True)