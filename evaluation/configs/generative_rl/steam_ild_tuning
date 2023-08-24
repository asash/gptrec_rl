import json
from aprec.datasets.steam import get_genres_steam_deduped_1000items_warm_users

from aprec.evaluation.metrics.ild import ILD
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.hit import HIT
from aprec.evaluation.metrics.pcount import PCOUNT
from aprec.evaluation.split_actions import LeaveOneOut
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender


from aprec.datasets.datasets_register import DatasetsRegister
from aprec.recommenders.sequential.models.generative.reward_metrics.ild_reward import ILDReward
from aprec.recommenders.sequential.models.generative.reward_metrics.ndcg_reward import NDCGReward
from aprec.recommenders.sequential.models.generative.reward_metrics.pcount_reward import PCountReward
from aprec.recommenders.sequential.models.generative.reward_metrics.weighted_sum_reward import WeightedSumReward
from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.lightfm import LightFMRecommender


USERS_FRACTIONS = [1.0]
RECOMMENDATIONS_LIMIT=500

genre_func = get_genres_steam_deduped_1000items_warm_users
DATASET = "steam_deduped_1000items_warm_users_noties"


METRICS = [HIT(1), HIT(10), NDCG(10), ILD(genre_func()), PCOUNT(10, DatasetsRegister()[DATASET]()) ]
#TARGET_ITEMS_SAMPLER = PopTargetItemsWithReplacementSampler(101)

SEQUENCE_LENGTH=200
#CHECKPOINT="./results/BERT4rec.ml-1m/ml1_generative_long_tuning_2023_08_13T07_36_23/checkpoints/checkpoint_step_18820"
CHECKPOINT="/home/aprec/Projects/aprec/evaluation/results/checkpoints_for_rl/steam_pretrained_from_bert4rec_100ksteps"
ILD_REWARD = ILDReward(genre_func())

def generative_tuning_recommender(ild_lambda=0.5, checkpoint_dir=CHECKPOINT, gae_gamma=0.99, gae_lambda=0.9, max_tuning_steps=64000):       
        from aprec.recommenders.rl_generative.generative_tuning_recommender import GenerativeTuningRecommender
        from aprec.recommenders.sequential.models.generative.gpt_rec_rl import RLGPT2RecConfig
        from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig
        model_config = RLGPT2RecConfig(transformer_blocks=3, embedding_size=256, tokenizer='id', tokens_per_item=1, values_per_dim=1500, attention_heads=4)
        recommender_config = SequentialRecommenderConfig(model_config, train_epochs=0) 
        recommender = GenerativeTuningRecommender(recommender_config,
                                                  pre_trained_checkpoint_dir=checkpoint_dir,
                                                  #pre_train_recommender_factory=lambda: RandomRecommender(),
                                                  max_tuning_steps=max_tuning_steps, 
                                                  tuning_batch_size=128, 
                                                  clip_eps=0.2, 
                                                  reward_metric=WeightedSumReward([NDCGReward(10), ILD_REWARD], [1, ild_lambda]),
                                                  tradeoff_monitoring_rewards=[(NDCGReward(10), ILD_REWARD)],
                                                  gae_gamma=gae_gamma, 
                                                  gae_lambda=gae_lambda,
                                                  validate_before_tuning=True,
                                                  sampling_processessess=8,
                                                  batch_recommendation_processess=1,
                                                  entropy_bonus=0.0,
                                                  ppo_lr=1e-5,
                                                  use_klpen=False,
                                                  #klpen_d_target=0.01,
                                                  supervised_guidance=False)
        return recommender
        

recommenders = {
} 

#initial pcount is approximately 15 times smaller than ndcg

lambdas = [0.3, 0.15, 0.075]
for l in lambdas:
    recommenders[f"generative_tuning_recommender_ild_{l}"] = lambda l=l: generative_tuning_recommender(l)


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
MAX_TEST_USERS=6040
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
RECOMMENDERS = get_recommenders(filter_seen=True)

if __name__ == "__main__":
    from aprec.tests.misc.test_configs import TestConfigs
    TestConfigs().validate_config(__file__)