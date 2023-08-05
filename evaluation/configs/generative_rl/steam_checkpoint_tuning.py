from aprec.evaluation.metrics.ild import ILD
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.hit import HIT
from aprec.evaluation.split_actions import LeaveOneOut
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.recommenders.sequential.models.generative.reward_metrics.ild_reward import ILDReward
from aprec.recommenders.sequential.models.generative.reward_metrics.ndcg_reward import NDCGReward
from aprec.recommenders.sequential.models.generative.reward_metrics.weighted_sum_reward import WeightedSumReward
from aprec.datasets.steam import get_genres_steam_deduped_1000items_warm_users



USERS_FRACTIONS = [1.0]
genre_func = get_genres_steam_deduped_1000items_warm_users


METRICS = [HIT(1), HIT(10), NDCG(10), ILD(genre_func()) ]
#TARGET_ITEMS_SAMPLER = PopTargetItemsWithReplacementSampler(101)

SEQUENCE_LENGTH=200

#checkpoints will be created in any case
SAVE_MODELS=False
RECOMMENDATIONS_LIMIT=100


CHECKPOINT="/home/aprec/Projects/aprec/evaluation/results/checkpoints_for_rl/steam_supervised_pre_trained_checkpoint"

def generative_tuning_recommender(ild_lambda=0.5, checkpoint_dir=CHECKPOINT, gae_gamma=0.1, gae_lambda=0.1):       
        from aprec.recommenders.rl_generative.generative_tuning_recommender import GenerativeTuningRecommender
        from aprec.recommenders.sequential.models.generative.gpt_rec_rl import RLGPT2RecConfig
        from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig


        model_config = RLGPT2RecConfig(transformer_blocks=3, embedding_size=256, tokenizer='id', tokens_per_item=1, values_per_dim=3500, attention_heads=4)
        recommender_config = SequentialRecommenderConfig(model_config) 
        recommender = GenerativeTuningRecommender(recommender_config,
                                                  pre_trained_checkpoint_dir=checkpoint_dir,
                                                  max_tuning_steps=32000, 
                                                  tuning_batch_size=128, 
                                                  clip_eps=0.1,
                                                  reward_metric=WeightedSumReward([NDCGReward(10), ILDReward(genre_func())], [1, ild_lambda]),
                                                  tradeoff_monitoring_rewards=[(NDCGReward(10), ILDReward(genre_func()))],
                                                  gae_gamma=gae_gamma, 
                                                  gae_lambda=gae_lambda,
                                                  validate_before_tuning=True,
                                                  sampling_processessess=8
                                                  )
        return recommender
        

recommenders = {
} 

recommenders[f"generative_tuning_recommender_ild_0.5"] = lambda: generative_tuning_recommender(0.5)
recommenders[f"generative_tuning_recommender_ild_0.2"] = lambda: generative_tuning_recommender(0.2)
recommenders[f"generative_tuning_recommender_ild_0.1"] = lambda: generative_tuning_recommender(0.1)
recommenders[f"generative_tuning_recommender_ild_0.0"] = lambda: generative_tuning_recommender(0.0)
recommenders[f"generative_tuning_recommender_ild_1.0"] = lambda: generative_tuning_recommender(1.0)

recommenders[f"generative_tuning_recommender_ild_0.5_gamma_0.1_lambda_0.1"] = lambda: generative_tuning_recommender(0.5, gae_gamma=0.1, gae_lambda=0.1)
recommenders[f"generative_tuning_recommender_ild_0.5_gamma_0.1_lambda_0.5"] = lambda: generative_tuning_recommender(0.5, gae_gamma=0.1, gae_lambda=0.5)
recommenders[f"generative_tuning_recommender_ild_0.5_gamma_0.1_lambda_0.9"] = lambda: generative_tuning_recommender(0.5, gae_gamma=0.1, gae_lambda=0.9)
recommenders[f"generative_tuning_recommender_ild_0.5_gamma_0.5_lambda_0.1"] = lambda: generative_tuning_recommender(0.5, gae_gamma=0.5, gae_lambda=0.1)
recommenders[f"generative_tuning_recommender_ild_0.5_gamma_0.5_lambda_0.5"] = lambda: generative_tuning_recommender(0.5, gae_gamma=0.5, gae_lambda=0.5)
recommenders[f"generative_tuning_recommender_ild_0.5_gamma_0.5_lambda_0.9"] = lambda: generative_tuning_recommender(0.5, gae_gamma=0.5, gae_lambda=0.9)
recommenders[f"generative_tuning_recommender_ild_0.5_gamma_0.9_lambda_0.1"] = lambda: generative_tuning_recommender(0.9, gae_gamma=0.5, gae_lambda=0.1)
recommenders[f"generative_tuning_recommender_ild_0.5_gamma_0.9_lambda_0.5"] = lambda: generative_tuning_recommender(0.9, gae_gamma=0.5, gae_lambda=0.5)
recommenders[f"generative_tuning_recommender_ild_0.5_gamma_0.9_lambda_0.9"] = lambda: generative_tuning_recommender(0.9, gae_gamma=0.5, gae_lambda=0.9)

    

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




DATASET = "steam_deduped_1000items_warm_users"
N_VAL_USERS=512
MAX_TEST_USERS=6040
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
RECOMMENDERS = get_recommenders(filter_seen=True)

if __name__ == "__main__":
    from aprec.tests.misc.test_configs import TestConfigs
    TestConfigs().validate_config(__file__)