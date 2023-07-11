import random
from aprec.datasets.movielens1m import get_genre_dict

from aprec.evaluation.metrics.entropy import Entropy
from aprec.evaluation.metrics.highest_score import HighestScore
from aprec.evaluation.metrics.ild import ILD
from aprec.evaluation.metrics.model_confidence import Confidence
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.hit import HIT
from aprec.evaluation.metrics.non_zero_scores import NonZeroScores
from aprec.evaluation.split_actions import LeaveOneOut
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.recommenders.sequential.models.generative.reward_metrics.ild_reward import ILDReward
from aprec.recommenders.sequential.models.generative.reward_metrics.ndcg_reward import NDCGReward
from aprec.recommenders.sequential.models.generative.reward_metrics.weighted_sum_reward import WeightedSumReward
from aprec.recommenders.top_recommender import TopRecommender


USERS_FRACTIONS = [1.0]

EXTRA_VAL_METRICS = [NDCG(10), HighestScore(), NDCG(40), HIT(1), MRR(), 
                     Confidence('Softmax'),
                     Confidence('Sigmoid'),
                     Entropy('Softmax', 10), 
                     NonZeroScores(),
                     HIT(10)]

METRICS = [HIT(1), HIT(10), NDCG(10), ILD(get_genre_dict()) ]
#TARGET_ITEMS_SAMPLER = PopTargetItemsWithReplacementSampler(101)

SEQUENCE_LENGTH=200

def sasrec_style_model(model_config, sequence_splitter, 
                target_builder,
                max_epochs=10000, 
                batch_size=1024,
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
                                use_keras_training=False,
                                extra_val_metrics=EXTRA_VAL_METRICS, 
                                sequence_length=SEQUENCE_LENGTH
                                )
    
    return SequentialRecommender(config)

def gsasrec(num_samples=256, t=0.75, max_epochs=10000):
    from aprec.recommenders.sequential.models.sasrec.sasrec import SASRecConfig
    from aprec.recommenders.sequential.targetsplitters.shifted_sequence_splitter import ShiftedSequenceSplitter
    from aprec.recommenders.sequential.target_builders.positives_sequence_target_builder import PositivesSequenceTargetBuilder
    model_config = SASRecConfig(vanilla=True, embedding_size=256, loss='bce', vanilla_num_negatives=num_samples, 
                                vanilla_bce_t=t)
    return sasrec_style_model(model_config, 
            ShiftedSequenceSplitter,
            target_builder=lambda: PositivesSequenceTargetBuilder(SEQUENCE_LENGTH),
            batch_size=64, 
            max_epochs=max_epochs)


def generative_tuning_recommender():       
        from aprec.recommenders.sequential.generative_tuning_recommender import GenerativeTuningRecommender
        from aprec.recommenders.sequential.models.generative.gpt_rec_rl import RLGPT2RecConfig
        from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig
        from aprec.recommenders.sequential.target_builders.dummy_builder import DummyTargetBuilder
        from aprec.recommenders.sequential.targetsplitters.id_splitter import IdSplitter
        from aprec.recommenders.sequential.models.sasrec.sasrec import SASRecConfig
        sasrec_config = SASRecConfig(vanilla=True, embedding_size=256, loss='bce', vanilla_num_negatives=256, 
                                vanilla_bce_t=0.75)

        model_config = RLGPT2RecConfig(transformer_blocks=3, embedding_size=256, tokenizer='id', tokens_per_item=1, values_per_dim=3500, attention_heads=4)
        #pre_training_recommender = lambda: FilterSeenRecommender(gsasrec(256, 0.75,1))
        pre_training_recommender = lambda: FilterSeenRecommender(TopRecommender())

        recommender_config = SequentialRecommenderConfig(model_config, train_epochs=1, early_stop_epochs=200,
                                               batch_size=128,
                                               training_time_limit=200000,  
                                               sequence_splitter=IdSplitter, 
                                               max_batches_per_epoch=100,
                                               targets_builder=DummyTargetBuilder,
                                               use_keras_training=False,
                                               sequence_length=SEQUENCE_LENGTH,
                                               validate_on_loss=True
                                               )
        recommender = GenerativeTuningRecommender(recommender_config, pre_training_recommender,
                                                  validate_every_steps=2, max_tuning_steps=4, 
                                                  tuning_batch_size=16, 
                                                  clip_eps=0.1,
                                                  reward_metric=WeightedSumReward([NDCGReward(10), ILDReward(get_genre_dict())], [1, 0.05]),
                                                  tradeoff_monitoring_rewards=[(NDCGReward(10), ILDReward(get_genre_dict()))],
                                                  )
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

DATASET = "ml1m_items_with5users"
N_VAL_USERS=128
MAX_TEST_USERS=6040
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
RECOMMENDERS = get_recommenders(filter_seen=True)

if __name__ == "__main__":
    from aprec.tests.misc.test_configs import TestConfigs
    TestConfigs().validate_config(__file__)

