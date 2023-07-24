import random
from aprec.datasets.movielens1m import get_genre_dict

from aprec.evaluation.metrics.ild import ILD
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.hit import HIT
from aprec.evaluation.split_actions import LeaveOneOut
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.recommenders.first_order_mc import FirstOrderMarkovChainRecommender
from aprec.recommenders.fmc_plus import SmartMC
from aprec.recommenders.lightfm import LightFMRecommender
from aprec.recommenders.random_recommender import RandomRecommender
from aprec.recommenders.sequential.models.generative.reward_metrics.ild_reward import ILDReward
from aprec.recommenders.sequential.models.generative.reward_metrics.ndcg_reward import NDCGReward
from aprec.recommenders.sequential.models.generative.reward_metrics.weighted_sum_reward import WeightedSumReward
from aprec.recommenders.top_recommender import TopRecommender


USERS_FRACTIONS = [1.0]


METRICS = [HIT(1), HIT(10), NDCG(10), ILD(get_genre_dict()) ]
#TARGET_ITEMS_SAMPLER = PopTargetItemsWithReplacementSampler(101)

SEQUENCE_LENGTH=200

def generative_tuning_recommender(ild_lambda, pretrain_recommender=SmartMC(order=50, discount=0.6), max_pretrain_epochs=10000):       
        from aprec.recommenders.sequential.generative_tuning_recommender import GenerativeTuningRecommender
        from aprec.recommenders.sequential.models.generative.gpt_rec_rl import RLGPT2RecConfig
        from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig
        from aprec.recommenders.sequential.target_builders.dummy_builder import DummyTargetBuilder
        from aprec.recommenders.sequential.targetsplitters.id_splitter import IdSplitter


        model_config = RLGPT2RecConfig(transformer_blocks=3, embedding_size=256, tokenizer='id', tokens_per_item=1, values_per_dim=3500, attention_heads=4)
        pre_training_recommender = lambda: FilterSeenRecommender(pretrain_recommender)

        recommender_config = SequentialRecommenderConfig(model_config, train_epochs=max_pretrain_epochs, early_stop_epochs=200,
                                               batch_size=128,
                                               training_time_limit=200000,  
                                               sequence_splitter=IdSplitter, 
                                               max_batches_per_epoch=100,
                                               targets_builder=DummyTargetBuilder,
                                               use_keras_training=True,
                                               sequence_length=SEQUENCE_LENGTH,
                                               validate_on_loss=True
                                               )
        recommender = GenerativeTuningRecommender(recommender_config, pre_training_recommender,
                                                  validate_every_steps=80, max_tuning_steps=16000, 
                                                  tuning_batch_size=16, 
                                                  clip_eps=0.1,
                                                  reward_metric=WeightedSumReward([NDCGReward(10), ILDReward(get_genre_dict())], [1, ild_lambda]),
                                                  tradeoff_monitoring_rewards=[(NDCGReward(10), ILDReward(get_genre_dict()))],
                                                  gae_gamma=0.1, 
                                                  gae_lambda=0.1
                                                  )
        return recommender
        

recommenders = {
    "generative_tuning_recommender_lambda:0": lambda: generative_tuning_recommender(ild_lambda=0),
    "generative_tuning_recommender_lambda:0.01": lambda: generative_tuning_recommender(ild_lambda=0.01),
    "generative_tuning_recommender_lambda:0.05": lambda: generative_tuning_recommender(ild_lambda=0.05),
    "generative_tuning_recommender_lambda:0.2": lambda: generative_tuning_recommender(ild_lambda=0.2),
    "generative_tuning_recommender_lambda:1": lambda: generative_tuning_recommender(ild_lambda=1), 
    "generative_tuning_recommender_lambda:0_init:mf": lambda: generative_tuning_recommender(ild_lambda=0, pretrain_recommender=LightFMRecommender(256)), 
    "generative_tuning_recommender_lambda:0_init:top": lambda: generative_tuning_recommender(ild_lambda=0, pretrain_recommender=TopRecommender()), 
    "generative_tuning_recommender_lambda:0_init:first_order_mc": lambda: generative_tuning_recommender(ild_lambda=0, pretrain_recommender=FirstOrderMarkovChainRecommender()), 
    "generative_tuning_recommender_lambda:0_init:random_recommender": lambda: generative_tuning_recommender(ild_lambda=0, pretrain_recommender=RandomRecommender()), 
    "generative_tuning_recommender_lambda:0_init:no_pretrain": lambda: generative_tuning_recommender(ild_lambda=0, pretrain_recommender=RandomRecommender(), max_pretrain_epochs=0), 
} 




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

N_VAL_USERS=512
DATASET = "steam_deduped_1000items_warm_users"
MAX_TEST_USERS=6040
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
RECOMMENDERS = get_recommenders(filter_seen=True)

if __name__ == "__main__":
    from aprec.tests.misc.test_configs import TestConfigs
    TestConfigs().validate_config(__file__)

