from aprec.evaluation.metrics.entropy import Entropy
from aprec.evaluation.metrics.highest_score import HighestScore
from aprec.evaluation.metrics.model_confidence import Confidence
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.hit import HIT
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender

USERS_FRACTIONS = [1.0]

EXTRA_VAL_METRICS = [NDCG(10), HighestScore(), NDCG(40), HIT(1), MRR(), 
                     Confidence('Softmax'),
                     Confidence('Sigmoid'),
                     Entropy('Softmax', 10), 
                     HIT(10)]

METRICS = [HIT(1), HIT(5), HIT(10), NDCG(5), NDCG(10), MRR(), HIT(4), NDCG(40), MAP(10)]
#TARGET_ITEMS_SAMPLER = PopTargetItemsWithReplacementSampler(101)

SEQUENCE_LENGTH=100

def gpt2rec():
        from aprec.recommenders.sequential.models.generative.gpt_rec import GPT2RecConfig
        from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig
        from aprec.recommenders.sequential.targetsplitters.id_splitter import IdSplitter
        from aprec.recommenders.sequential.target_builders.dummy_builder import DummyTargetBuilder
        from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender

        model_config = GPT2RecConfig(embedding_size=256)
        bs=16
        recommender_config = SequentialRecommenderConfig(model_config, train_epochs=20, early_stop_epochs=300,
                                               batch_size=bs,
                                               eval_batch_size=bs, 
                                               validation_batch_size=bs,
                                               training_time_limit=100000000,  
                                               sequence_splitter=IdSplitter, 
                                               targets_builder=DummyTargetBuilder,
                                               use_keras_training=False,
                                               sequence_length=SEQUENCE_LENGTH,
                                               max_batches_per_epoch=256,
                                               extra_val_metrics=EXTRA_VAL_METRICS
                                               )
        
        recommender = SequentialRecommender(recommender_config)
        return recommender

recommenders = {"gpt2rec": gpt2rec} 
 
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

