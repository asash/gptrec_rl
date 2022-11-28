import random
from aprec.evaluation.samplers.pop_sampler import PopTargetItemsSampler
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.hit import HIT
from aprec.recommenders.dnn_sequential_recommender.target_builders.negative_samplers import SVDSimilaritySampler, PopularityBasedSampler, MixtureSampler, RandomNegativesSampler
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.shifted_sequence_splitter import ShiftedSequenceSplitter
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.recommenders.lightfm import LightFMRecommender
from aprec.recommenders.top_recommender import TopRecommender

USERS_FRACTIONS = [1.0]


def bert4rec_ft(negatives_sampler=SVDSimilaritySampler()):
        from aprec.recommenders.dnn_sequential_recommender.history_vectorizers.add_mask_history_vectorizer import AddMaskHistoryVectorizer
        from aprec.recommenders.dnn_sequential_recommender.targetsplitters.items_masking import ItemsMasking
        from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
        from aprec.recommenders.dnn_sequential_recommender.target_builders.items_masking_with_negatives import ItemsMaskingWithNegativesTargetsBuilder
        from aprec.recommenders.dnn_sequential_recommender.models.bert4recft.bert4recft import BERT4RecFT
        from aprec.losses.bce import BCELoss
        from aprec.losses.items_masking_loss_proxy import ItemsMaksingLossProxy
        sequence_len = 100
        model = BERT4RecFT(max_history_len=sequence_len)
        batch_size = 256 
        negatives_per_positive = negatives_sampler.get_sample_size()
        recommender = DNNSequentialRecommender(model, train_epochs=100000, early_stop_epochs=200,
                                               batch_size=batch_size,
                                               training_time_limit=3600000, 
                                               loss = ItemsMaksingLossProxy(BCELoss(), negatives_per_positive, sequence_len),
                                                sequence_splitter=lambda: ItemsMasking(), 
                                               targets_builder= lambda: ItemsMaskingWithNegativesTargetsBuilder(negatives_sampler=negatives_sampler),
                                               pred_history_vectorizer=AddMaskHistoryVectorizer(),
                                               max_batches_per_epoch=24
                                               )
        return recommender

def regular_bert4rec():
        sequence_len = 100
        from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
        from aprec.losses.mean_ypred_ploss import MeanPredLoss
        from aprec.recommenders.dnn_sequential_recommender.targetsplitters.items_masking import ItemsMasking
        from aprec.recommenders.dnn_sequential_recommender.models.bert4rec.bert4rec import BERT4Rec
        from aprec.recommenders.dnn_sequential_recommender.target_builders.items_masking_target_builder import ItemsMaskingTargetsBuilder
        from aprec.recommenders.dnn_sequential_recommender.history_vectorizers.add_mask_history_vectorizer import AddMaskHistoryVectorizer
        model = BERT4Rec( max_history_len=sequence_len)
        recommender = DNNSequentialRecommender(model, train_epochs=100000, early_stop_epochs=200,
                                               batch_size=64,
                                               training_time_limit=3600000, 
                                               loss = MeanPredLoss(),
                                               debug=False, sequence_splitter=lambda: ItemsMasking(), 
                                               targets_builder= lambda: ItemsMaskingTargetsBuilder(),
                                               val_sequence_splitter=lambda: ItemsMasking(force_last=True),
                                               metric=MeanPredLoss(), 
                                               pred_history_vectorizer=AddMaskHistoryVectorizer(),
                                               max_batches_per_epoch=100, 
                                               )
        return recommender

def vanilla_sasrec():
        sequence_len = 200
        from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
        from aprec.losses.bce import BCELoss
        from aprec.recommenders.dnn_sequential_recommender.models.sasrec.sasrec import SASRec
        from aprec.recommenders.dnn_sequential_recommender.target_builders.negative_per_positive_target import NegativePerPositiveTargetBuilder
        from tensorflow.keras.optimizers import Adam
        model = SASRec(max_history_len=sequence_len, vanilla=True)
        recommender = DNNSequentialRecommender(model, train_epochs=100000, early_stop_epochs=200,
                                               batch_size=64,
                                               training_time_limit=3600000, 
                                               optimizer=Adam(beta_2=0.98),
                                               loss = BCELoss(),
                                               debug=False, sequence_splitter=ShiftedSequenceSplitter, 
                                               targets_builder= lambda: NegativePerPositiveTargetBuilder(sequence_len),
                                               val_sequence_splitter=ShiftedSequenceSplitter,
                                               max_batches_per_epoch=100, 
                                               metric=BCELoss(), 
                                               )
        return recommender


def top_recommender():
    return TopRecommender()

def lightfm_recommender(k=256, loss='bpr'):
    return LightFMRecommender(k, loss, num_threads=32)


recommenders = {
  "BERT4RecScaleRandomPopularity": lambda: bert4rec_ft(MixtureSampler([RandomNegativesSampler(129), PopularityBasedSampler(129)])),
  "BERT4RecScaleRandomSVD": lambda: bert4rec_ft(MixtureSampler([RandomNegativesSampler(129), SVDSimilaritySampler(129)])),
  "BERT4RecScalePopularitySVD": lambda: bert4rec_ft(MixtureSampler([PopularityBasedSampler(129), SVDSimilaritySampler(129)])),
  "BERT4RecScaleRandomPopularitySVD": lambda: bert4rec_ft(MixtureSampler([RandomNegativesSampler(86), PopularityBasedSampler(86), SVDSimilaritySampler(86)])),
  "BERT4RecScaleRandomOnly": lambda: bert4rec_ft(RandomNegativesSampler(258)),
  "BERT4RecScalePopularityOnly": lambda: bert4rec_ft(PopularityBasedSampler(258)),
  "BERT4RecScaleSVDOnly": lambda: bert4rec_ft(SVDSimilaritySampler(258)),
}

METRICS = [HIT(1), HIT(5), HIT(10), NDCG(5), NDCG(10), MRR(), HIT(4), NDCG(40), MAP(10)]
TARGET_ITEMS_SAMPLER = PopTargetItemsSampler(101)

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

