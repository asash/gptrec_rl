from aprec.evaluation.metrics.entropy import Entropy
from aprec.evaluation.metrics.highest_score import HighestScore
from aprec.evaluation.metrics.model_confidence import Confidence
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.hit import HIT
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.recommenders.lightfm import LightFMRecommender
from aprec.recommenders.top_recommender import TopRecommender

USERS_FRACTIONS = [1.0]

EXTRA_VAL_METRICS = [HIT(10), NDCG(10), HighestScore(), 
                     Confidence('Softmax'), Confidence('Sigmoid'), 
                     Entropy('Sigmoid', 2),  Entropy('Sigmoid', 5),  Entropy('Sigmoid', 10),
                     Entropy('Softmax', 2), Entropy('Softmax', 5), Entropy('Softmax', 10)]
SEQUENCE_LENGTH=200
EMBEDDING_SIZE=128
 

def sasrec_rss(recency_importance): 
        from aprec.recommenders.sequential.models.sasrec.sasrec import SASRecConfig

        from aprec.recommenders.sequential.target_builders.positives_only_targets_builder import PositvesOnlyTargetBuilder
        from aprec.recommenders.sequential.targetsplitters.recency_sequence_sampling import RecencySequenceSampling
        from aprec.recommenders.sequential.targetsplitters.recency_sequence_sampling import exponential_importance
        target_splitter = lambda: RecencySequenceSampling(0.2, exponential_importance(recency_importance))
        model_config = SASRecConfig(vanilla=False, embedding_size=EMBEDDING_SIZE)
        return sasrec_style_model(
            model_config,
            sequence_splitter=target_splitter,
            target_builder=PositvesOnlyTargetBuilder, 
            batch_size=1024)

def vanilla_sasrec():
    from aprec.recommenders.sequential.models.sasrec.sasrec import SASRecConfig
    from aprec.recommenders.sequential.target_builders.negative_per_positive_target import NegativePerPositiveTargetBuilder
    from aprec.recommenders.sequential.targetsplitters.shifted_sequence_splitter import ShiftedSequenceSplitter
    from aprec.recommenders.sequential.history_vectorizers.default_history_vectorizer import DefaultHistoryVectrizer
    model_config = SASRecConfig(vanilla=True, embedding_size=EMBEDDING_SIZE)
    return sasrec_style_model(model_config, 
            ShiftedSequenceSplitter,
            target_builder=lambda: NegativePerPositiveTargetBuilder(SEQUENCE_LENGTH),
            batch_size=1024)


def sasrec_style_model(model_config, sequence_splitter, 
                target_builder,
                max_epochs=10000, 
                batch_size=128,
                ):
    from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender
    from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig

    config = SequentialRecommenderConfig(model_config,                       
                                train_epochs=max_epochs,
                                early_stop_epochs=200,
                                batch_size=batch_size,
                                max_batches_per_epoch=256,
                                sequence_splitter=sequence_splitter, 
                                targets_builder=target_builder, 
                                use_keras_training=True,
                                extra_val_metrics=EXTRA_VAL_METRICS, 
                                sequence_length=SEQUENCE_LENGTH
                                )
    
    return SequentialRecommender(config)

def get_bert_style_model(model_config):
        from aprec.recommenders.sequential.history_vectorizers.add_mask_history_vectorizer import AddMaskHistoryVectorizer
        from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender
        from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig
        from aprec.recommenders.sequential.target_builders.items_masking_target_builder import ItemsMaskingTargetsBuilder
        from aprec.recommenders.sequential.targetsplitters.items_masking import ItemsMasking
        recommender_config = SequentialRecommenderConfig(model_config, 
                                               train_epochs=10000, early_stop_epochs=200,
                                               batch_size=128,
                                               sequence_splitter=ItemsMasking, 
                                               max_batches_per_epoch=256,
                                               targets_builder=ItemsMaskingTargetsBuilder,
                                               pred_history_vectorizer=AddMaskHistoryVectorizer(),
                                               use_keras_training=True,
                                               sequence_length=SEQUENCE_LENGTH,
                                               extra_val_metrics = EXTRA_VAL_METRICS)
        
        return SequentialRecommender(recommender_config)

def full_bert():
        from aprec.recommenders.sequential.models.bert4rec.full_bert import FullBERTConfig
        model_config =  FullBERTConfig(embedding_size=EMBEDDING_SIZE)
        return get_bert_style_model(model_config)

def sampling_bert():
        from aprec.recommenders.sequential.models.bert4rec.bert4recft import SampleBERTConfig
        model_config =  SampleBERTConfig(embedding_size=EMBEDDING_SIZE)
        return get_bert_style_model(model_config)

def popularity():
        return TopRecommender()

def mf_bpr():
        return LightFMRecommender(num_latent_components=EMBEDDING_SIZE, num_threads=32)


recommenders = {
        "popularity": popularity,
        "mf-bpr": mf_bpr,
        "BERT4rec": full_bert, 
        "BERT4rec-sampling": sampling_bert, 
        "SASRec": vanilla_sasrec,
        "SASRec-RSS": lambda: sasrec_rss(0.8)
}

METRICS = [HIT(1), HIT(5), HIT(10), NDCG(5), NDCG(10), MRR(), HIT(4), NDCG(40), MAP(10)]

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

