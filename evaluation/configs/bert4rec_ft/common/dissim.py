import random
from aprec.evaluation.samplers.pop_sampler import PopTargetItemsSampler
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.hit import HIT
from aprec.losses.bce import BCELoss
from aprec.recommenders.sequential.target_builders.negative_samplers import RandomNegativesWithCosSimValues
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.recommenders.lightfm import LightFMRecommender
from aprec.recommenders.metrics.ndcg import KerasNDCG
from aprec.recommenders.top_recommender import TopRecommender

USERS_FRACTIONS = [1.0]


def bert4rec_ft(negatives_sampler, metric, loss, sequence_len=100, add_positive=True, activation='linear'):
        from aprec.recommenders.sequential.history_vectorizers.add_mask_history_vectorizer import AddMaskHistoryVectorizer
        from aprec.recommenders.sequential.targetsplitters.items_masking import ItemsMasking
        from aprec.recommenders.sequential.sequential_recommender import DNNSequentialRecommender
        from aprec.recommenders.sequential.target_builders.items_masking_with_negatives import ItemsMaskingWithNegativesTargetsBuilder
        from aprec.recommenders.sequential.models.bert4recft.bert4recft import BERT4RecFT
        from aprec.losses.items_masking_loss_proxy import ItemsMaksingLossProxy
        model = BERT4RecFT(max_history_len=sequence_len, output_layer_activation=activation)
        batch_size = 256 
        negatives_per_positive = negatives_sampler.get_sample_size()
        recommender = DNNSequentialRecommender(model, train_epochs=100000, early_stop_epochs=200,
                                               batch_size=batch_size,
                                               training_time_limit=3600000, 
                                               loss = ItemsMaksingLossProxy(loss, negatives_per_positive, sequence_len, add_positive=add_positive),
                                               sequence_splitter=lambda: ItemsMasking(), 
                                               targets_builder= lambda: ItemsMaskingWithNegativesTargetsBuilder(negatives_sampler=negatives_sampler, add_positive=add_positive),
                                               data_generator_processes=8,
                                               pred_history_vectorizer=AddMaskHistoryVectorizer(),
                                               max_batches_per_epoch=24
                                               )
        return recommender

def top_recommender():
    return TopRecommender()

def lightfm_recommender(k=256, loss='bpr'):
    return LightFMRecommender(k, loss, num_threads=32)

num_negatives = 258
metric = lambda: KerasNDCG(40)

recommenders = {
  #"BERT4RecScaleRandom": lambda: bert4rec_ft(RandomNegativesSampler(num_negatives), metric(), loss()),
  #"BERT4RecScaleDissim": lambda: bert4rec_ft(AffinityDissimilaritySampler(num_negatives, smoothing=1), metric(), loss()),
  #"BERT4RecScaleRandomCosSim": lambda: bert4rec_ft(RandomNegativesWithCosSimValues(num_negatives), metric(), loss()),
  "BERT4RecScaleRandomCosSimLinear": lambda: bert4rec_ft(RandomNegativesWithCosSimValues(num_negatives), metric(), BCELoss()),
  "BERT4RecScaleRandomCosSimNoPosLinear": lambda: bert4rec_ft(RandomNegativesWithCosSimValues(num_negatives), metric(), BCELoss(), add_positive=False),
  "BERT4RecScaleRandomCosSimSigmoid": lambda: bert4rec_ft(RandomNegativesWithCosSimValues(num_negatives), metric(), BCELoss(), activation='sigmoid'),
  "BERT4RecScaleRandomCosSimNoPosSigmoid": lambda: bert4rec_ft(RandomNegativesWithCosSimValues(num_negatives), metric(), BCELoss(), add_positive=False, activation='sigmoid'),
  "BERT4RecScaleRandomCosSimSoftmax": lambda: bert4rec_ft(RandomNegativesWithCosSimValues(num_negatives), metric(), BCELoss(), activation='softmax'),
  "BERT4RecScaleRandomCosSimNoPosSoftmax": lambda: bert4rec_ft(RandomNegativesWithCosSimValues(num_negatives), metric(), BCELoss(), add_positive=False, activation='softmax'),
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

