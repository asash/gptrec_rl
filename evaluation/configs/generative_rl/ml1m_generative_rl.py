import random

from aprec.evaluation.metrics.entropy import Entropy
from aprec.evaluation.metrics.highest_score import HighestScore
from aprec.evaluation.metrics.model_confidence import Confidence
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.hit import HIT
from aprec.evaluation.metrics.non_zero_scores import NonZeroScores
from aprec.evaluation.split_actions import LeaveOneOut
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender


USERS_FRACTIONS = [1.0]

EXTRA_VAL_METRICS = [NDCG(10), HighestScore(), NDCG(40), HIT(1), MRR(), 
                     Confidence('Softmax'),
                     Confidence('Sigmoid'),
                     Entropy('Softmax', 10), 
                     NonZeroScores(),
                     HIT(10)]

METRICS = [HIT(1), HIT(5), HIT(10), NDCG(5), NDCG(10), MRR(), HIT(4), NDCG(40), MAP(10)]
#TARGET_ITEMS_SAMPLER = PopTargetItemsWithReplacementSampler(101)

SEQUENCE_LENGTH=200


def generative_tuning_recommender():       
        from aprec.recommenders.first_order_mc import FirstOrderMarkovChainRecommender
        from aprec.recommenders.sequential.generative_tuning_recommender import GenerativeTuningRecommender
        from aprec.recommenders.sequential.models.generative.gpt_rec_rl import RLGPT2RecConfig
        from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig
        from aprec.recommenders.sequential.target_builders.dummy_builder import DummyTargetBuilder
        from aprec.recommenders.sequential.targetsplitters.id_splitter import IdSplitter
        model_config = RLGPT2RecConfig(transformer_blocks=3, embedding_size=256, tokenizer='id', tokens_per_item=1, values_per_dim=3500, attention_heads=4)
        pre_training_recommender = lambda: FilterSeenRecommender(FirstOrderMarkovChainRecommender())

        recommender_config = SequentialRecommenderConfig(model_config, train_epochs=100, early_stop_epochs=10,
                                               batch_size=128,
                                               training_time_limit=200000,  
                                               sequence_splitter=IdSplitter, 
                                               max_batches_per_epoch=100,
                                               targets_builder=DummyTargetBuilder,
                                               use_keras_training=True,
                                               sequence_length=SEQUENCE_LENGTH,
                                               validate_on_loss=True
                                               )
        recommender = GenerativeTuningRecommender(recommender_config, pre_training_recommender, validate_every_steps=20, max_tuning_steps=1000)
        return recommender
        

recommenders = {
    "generative_tuning_recommender": generative_tuning_recommender
} 

r_list = list(recommenders.items())
random.seed(31337)
random.shuffle(r_list)
recommenders=dict(r_list)

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

DATASET = "BERT4rec.ml-1m"
N_VAL_USERS=128
MAX_TEST_USERS=6040
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
RECOMMENDERS = get_recommenders(filter_seen=True)

if __name__ == "__main__":
    from aprec.tests.misc.test_configs import TestConfigs
    TestConfigs().validate_config(__file__)

