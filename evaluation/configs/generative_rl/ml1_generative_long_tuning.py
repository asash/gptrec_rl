
from aprec.evaluation.metrics.ild import ILD
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.hit import HIT
from aprec.evaluation.split_actions import LeaveOneOut
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.recommenders.sequential.models.generative.reward_metrics.acc_reward import ACCReward
from aprec.recommenders.sequential.models.generative.reward_metrics.ild_reward import ILDReward
from aprec.recommenders.sequential.models.generative.reward_metrics.ndcg_reward import NDCGReward
from aprec.recommenders.sequential.models.generative.reward_metrics.weighted_sum_reward import WeightedSumReward
from aprec.datasets.bert4rec_datasets import get_movielens1m_genres


USERS_FRACTIONS = [1.0]

#checkpoints will be created in any case
SAVE_MODELS=False

genre_func = get_movielens1m_genres


METRICS = [HIT(1), HIT(10), NDCG(10), ILD(genre_func()) ]
#TARGET_ITEMS_SAMPLER = PopTargetItemsWithReplacementSampler(101)

SEQUENCE_LENGTH=200
CHECKPOINT="/home/aprec/Projects/aprec/evaluation/results/checkpoints_for_rl/ml1m_supervised_pre_trained_checkpoint"

#1.0  for gae_gamma and gae_lambda allows the model to see and plan for the whole sequence
def generative_tuning_recommender(ild_lambda=0.5, checkpoint_dir=CHECKPOINT, gae_gamma=1.0, gae_lambda=1.0, max_tuning_steps=32000):       
        from aprec.recommenders.rl_generative.generative_tuning_recommender import GenerativeTuningRecommender
        from aprec.recommenders.sequential.models.generative.gpt_rec_rl import RLGPT2RecConfig
        from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig


        model_config = RLGPT2RecConfig(transformer_blocks=3, embedding_size=256, tokenizer='id', tokens_per_item=1, values_per_dim=3500, attention_heads=4)
        recommender_config = SequentialRecommenderConfig(model_config, train_epochs=0) 
        recommender = GenerativeTuningRecommender(recommender_config,
                                                  pre_trained_checkpoint_dir=checkpoint_dir,
                                                  #pre_train_recommender_factory=lambda: RandomRecommender(),
                                                  max_tuning_steps=max_tuning_steps, 
                                                  tuning_batch_size=128, 
                                                  clip_eps=1.0, #allow large changes here, most divergence is controlled by klpen
                                                  reward_metric=WeightedSumReward([NDCGReward(10), ILDReward(genre_func())], [1, ild_lambda]),
                                                  tradeoff_monitoring_rewards=[(NDCGReward(10), ILDReward(genre_func()))],
                                                  gae_gamma=gae_gamma, 
                                                  gae_lambda=gae_lambda,
                                                  validate_before_tuning=True,
                                                  sampling_processessess=8,
                                                  batch_recommendation_processess=1,
                                                  entropy_bonus=0.0,
                                                  klpen_d_target=0.08,
                                                  ppo_lr=1e-4,
                                                  use_klpen=True
                                                  )
        return recommender
        

recommenders = {
} 

recommenders[f"generative_tuning_recommender_ild_0.0"] = lambda: generative_tuning_recommender(0.0, max_tuning_steps=2000000)
    

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
N_VAL_USERS=512
MAX_TEST_USERS=6040
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
RECOMMENDERS = get_recommenders(filter_seen=True)

if __name__ == "__main__":
    from aprec.tests.misc.test_configs import TestConfigs
    TestConfigs().validate_config(__file__)
