from aprec.evaluation.metrics.entropy import Entropy
from aprec.evaluation.metrics.highest_score import HighestScore
from aprec.evaluation.metrics.hit import HIT
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.model_confidence import Confidence
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.split_actions import LeaveOneOut
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender

def sasrec_rss(recency_importance, add_cls=False, pos_smoothing=0,
               pos_embedding='default', pos_embeddding_comb='add', 
               causal_attention = True):
        from aprec.recommenders.sequential.history_vectorizers.add_mask_history_vectorizer import AddMaskHistoryVectorizer
        from aprec.recommenders.sequential.history_vectorizers.default_history_vectorizer import DefaultHistoryVectrizer
        from aprec.recommenders.sequential.models.sasrec.sasrec import SASRecConfig

        from aprec.recommenders.sequential.target_builders.positives_only_targets_builder import PositvesOnlyTargetBuilder
        from aprec.recommenders.sequential.targetsplitters.recency_sequence_sampling import RecencySequenceSampling
        from aprec.recommenders.sequential.targetsplitters.recency_sequence_sampling import exponential_importance


        target_splitter = lambda: RecencySequenceSampling(0.2, exponential_importance(recency_importance), add_cls=add_cls)
        pred_history_vectorizer = AddMaskHistoryVectorizer() if add_cls else DefaultHistoryVectrizer()
        model_config = SASRecConfig(vanilla=False, num_heads=1, 
                   pos_embedding=pos_embedding,
                   pos_emb_comb=pos_embeddding_comb,
                   pos_smoothing=pos_smoothing, 
                   causal_attention=causal_attention) 

        return dnn(
            model_config,
            sequence_splitter=target_splitter,
            target_builder=PositvesOnlyTargetBuilder, 
            batch_size=1024,
            pred_history_vectorizer=pred_history_vectorizer)

def vanilla_sasrec():
    from aprec.recommenders.sequential.models.sasrec.sasrec import SASRecConfig
    from aprec.recommenders.sequential.target_builders.negative_per_positive_target import NegativePerPositiveTargetBuilder
    from aprec.recommenders.sequential.targetsplitters.shifted_sequence_splitter import ShiftedSequenceSplitter
    from aprec.recommenders.sequential.history_vectorizers.default_history_vectorizer import DefaultHistoryVectrizer

    sequence_length = 50
    model_config = SASRecConfig(vanilla=True )


    return dnn(model_config, 
            ShiftedSequenceSplitter,
            target_builder=lambda: NegativePerPositiveTargetBuilder(sequence_length),
            sequence_length=sequence_length, 
            batch_size=1024,
            pred_history_vectorizer= DefaultHistoryVectrizer())


def dnn(model_config, sequence_splitter, 
                target_builder,
                pred_history_vectorizer,
                training_time_limit=6*3600,  
                max_epochs=10000, 
                sequence_length=50, 
                batch_size=128,
                ):
    from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender
    from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig

    config = SequentialRecommenderConfig(model_config,                       
                                train_epochs=max_epochs,
                                early_stop_epochs=max_epochs,
                                batch_size=batch_size,
                                max_batches_per_epoch=256,
                                training_time_limit=training_time_limit,
                                sequence_splitter=sequence_splitter, 
                                targets_builder=target_builder, 
                                pred_history_vectorizer=pred_history_vectorizer,
                                sequence_length=sequence_length, 
                                use_keras_training=True,
                                extra_val_metrics=EXTRA_VAL_METRICS
                                )
    
    return SequentialRecommender(config)


USERS_FRACTIONS = [1.0]

EXTRA_VAL_METRICS = [HIT(10), NDCG(10), HighestScore(), 
                     Confidence('Softmax'), Confidence('Sigmoid'), 
                     Entropy('Sigmoid', 10),Entropy('Softmax', 10),
                     HIT(100), 
                     MRR(),
                     HIT(1)]

recommenders = {
    "Sasrec-rss": lambda: sasrec_rss(0.8),
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

DATASET = "movies_dtaset_with_budget_band"
N_VAL_USERS=1024
MAX_TEST_USERS=100000
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
RECOMMENDERS = get_recommenders(filter_seen=True)

if __name__ == "__main__":
    from aprec.tests.test_configs import TestConfigs
    TestConfigs().validate_config(__file__)