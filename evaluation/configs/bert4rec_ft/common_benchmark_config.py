import random
from aprec.evaluation.metrics.highest_score import HighestScore
from aprec.evaluation.metrics.model_confidence import Confidence
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.hit import HIT
from aprec.losses.bce import BCELoss
from aprec.losses.lambda_gamma_rank import LambdaGammaRankLoss
from aprec.losses.softmax_crossentropy import SoftmaxCrossEntropy
from aprec.recommenders.dnn_sequential_recommender.target_builders.negative_samplers import SVDSimilaritySampler
from aprec.recommenders.dnn_sequential_recommender.target_builders.negative_samplers.popularity_based_sampler import PopularityBasedSampler
from aprec.recommenders.dnn_sequential_recommender.target_builders.negative_samplers.random_negatives_sampler import RandomNegativesSampler
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.shifted_sequence_splitter import ShiftedSequenceSplitter
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.recommenders.lightfm import LightFMRecommender
from aprec.recommenders.top_recommender import TopRecommender

USERS_FRACTIONS = [1.0]


def bert4rec_ft(negatives_sampler, loss, use_ann=False, batch_size=256):
        from aprec.recommenders.dnn_sequential_recommender.history_vectorizers.add_mask_history_vectorizer import AddMaskHistoryVectorizer
        from aprec.recommenders.dnn_sequential_recommender.targetsplitters.items_masking import ItemsMasking
        from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
        from aprec.recommenders.dnn_sequential_recommender.target_builders.items_masking_with_negatives import ItemsMaskingWithNegativesTargetsBuilder
        from aprec.recommenders.dnn_sequential_recommender.models.bert4recft.bert4recft import BERT4RecFT
        from aprec.losses.items_masking_loss_proxy import ItemsMaksingLossProxy
        sequence_len = 100
        model = BERT4RecFT(max_history_len=sequence_len)
        negatives_per_positive = negatives_sampler.get_sample_size()
        recommender = DNNSequentialRecommender(model, train_epochs=100000, early_stop_epochs=200,
                                               batch_size=batch_size,
                                               training_time_limit=3600000, 
                                               loss = ItemsMaksingLossProxy(loss, negatives_per_positive, sequence_len),
                                               sequence_splitter=lambda: ItemsMasking(), 
                                               targets_builder= lambda: ItemsMaskingWithNegativesTargetsBuilder(negatives_sampler=negatives_sampler),
                                               pred_history_vectorizer=AddMaskHistoryVectorizer(),
                                               max_batches_per_epoch=48,
                                               eval_batch_size=128,
                                               extra_val_metrics=[HIT(10), HighestScore(), Confidence('Softmax'), Confidence('Sigmoid')],
                                               use_ann_for_inference=use_ann)
        return recommender

def full_bert(loss, num_samples_normalization=False, batch_size=64):
        sequence_len = 100
        from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
        from aprec.losses.mean_ypred_ploss import MeanPredLoss
        from aprec.recommenders.dnn_sequential_recommender.targetsplitters.items_masking import ItemsMasking
        from aprec.recommenders.dnn_sequential_recommender.models.bert4recft.full_bert import FullBERT
        from aprec.recommenders.dnn_sequential_recommender.target_builders.items_masking_target_builder import ItemsMaskingTargetsBuilder
        from aprec.recommenders.dnn_sequential_recommender.history_vectorizers.add_mask_history_vectorizer import AddMaskHistoryVectorizer
        model = FullBERT(max_history_len=sequence_len, loss=loss, num_samples_normalization=num_samples_normalization)
        recommender = DNNSequentialRecommender(model, train_epochs=100000, early_stop_epochs=200,
                                               batch_size=batch_size,
                                               training_time_limit=3600000, 
                                               loss = MeanPredLoss(),
                                               sequence_splitter=lambda: ItemsMasking(), 
                                               targets_builder= lambda: ItemsMaskingTargetsBuilder(),
                                               pred_history_vectorizer=AddMaskHistoryVectorizer(),
                                               max_batches_per_epoch=192, 
                                               eval_batch_size=128, 
                                               use_ann_for_inference=False, 
                                               extra_val_metrics=[HIT(10), HighestScore(), Confidence('Softmax'), Confidence('Sigmoid')]
                                               )
        return recommender


recommenders = {
   #"BERT4RecFullLambdaGammmaRankNormalized": lambda: full_bert(LambdaGammaRankLoss(pred_truncate_at=1024, bce_grad_weight=0.5), batch_size=64, num_samples_normalization=True),
   #"BERT4RecFullLambdaRankNormalized": lambda: full_bert(LambdaGammaRankLoss(pred_truncate_at=1024), batch_size=64, num_samples_normalization=True),
   #"BERT4RecFullLambdaRank": lambda: full_bert(LambdaGammaRankLoss(pred_truncate_at=1024), batch_size=64),

   "BERT4RecFullSoftMaxCE": lambda: full_bert(SoftmaxCrossEntropy()),
   "BERT4RecFullSoftMaxCENormalized": lambda: full_bert(SoftmaxCrossEntropy(), num_samples_normalization=True),
   #"BERT4RecSampling400BCE": lambda: bert4rec_ft(RandomNegativesSampler(400), BCELoss()),
   #"BERT4RecFullBCENormalized": lambda: full_bert(BCELoss(), num_samples_normalization=True),
   #"BERT4RecFullBCE": lambda: full_bert(BCELoss()),
   #"BERT4RecSampling400LambdaGammaRankLoss": lambda: bert4rec_ft(RandomNegativesSampler(400), LambdaGammaRankLoss(), batch_size=64),
   #"BERT4RecSampling400SoftMaxCE": lambda: bert4rec_ft(RandomNegativesSampler(400), SoftmaxCrossEntropy()),
}

METRICS = [HIT(1), HIT(5), HIT(10), NDCG(5), NDCG(10), MRR(), HIT(4), NDCG(40), MAP(10)]
#TARGET_ITEMS_SAMPLER = PopTargetItemsSampler(101)

def get_recommenders(filter_seen: bool):
    result = {}
    all_recommenders = list(recommenders.keys())
    for recommender_name in all_recommenders:
        if filter_seen:
            result[recommender_name] =\
                lambda recommender_name=recommender_name: FilterSeenRecommender(recommenders[recommender_name]())
        else:
            result[recommender_name] = recommenders[recommender_name]
    return result

