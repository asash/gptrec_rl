import json

from aprec.evaluation.metrics.ild import ILD
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.hit import HIT
from aprec.evaluation.metrics.pcount import PCOUNT
from aprec.evaluation.split_actions import LeaveOneOut
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender


from aprec.datasets.bert4rec_datasets import get_movielens1m_genres
from aprec.datasets.datasets_register import DatasetsRegister
from aprec.recommenders.sequential.models.generative.reward_metrics.pcount_reward import PCountReward
from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.lightfm import LightFMRecommender


USERS_FRACTIONS = [1.0]
DATASET = "BERT4rec.ml-1m"
RECOMMENDATIONS_LIMIT=900


genre_func = get_movielens1m_genres


METRICS = [HIT(1), HIT(10), NDCG(10), ILD(genre_func()), PCOUNT(10, DatasetsRegister()[DATASET]()) ]
#TARGET_ITEMS_SAMPLER = PopTargetItemsWithReplacementSampler(101)
PCOUNT_REWARD = PCountReward(10, DatasetsRegister()[DATASET]())

SEQUENCE_LENGTH=200

#CHECKPOINT="/home/alekspet/Projects/aprec/aprec/evaluation/results/ml1m_items_with5users/ml1_generative_long_tuning_2023_08_08T16_05_30/checkpoints/checkpoint_step_30080"
#CHECKPOINT="/home/alekspet/Projects/aprec/aprec/evaluation/results/ml1m_items_with5users/ml1_generative_long_tuning_2023_08_09T08_25_52/checkpoints/checkpoint_step_46360"
#CHECKPOINT="/home/alekspet/Projects/aprec/aprec/evaluation/results/ml1m_items_with5users/ml1_generative_long_tuning_2023_08_09T19_37_36/checkpoints/checkpoint_step_43920"
#CHECKPOINT="/home/alekspet/Projects/aprec/aprec/evaluation/results/BERT4rec.ml-1m/ml1_generative_long_tuning_2023_08_12T15_25_02/checkpoints/checkpoint_step_33700" -- a good checkpoint on desktop, achieves 0.14 NDCG
#CHECKPOINT="./results/BERT4rec.ml-1m/ml1_generative_long_tuning_2023_08_13T07_36_23/checkpoints/checkpoint_step_18820"
#CHECKPOINT="/home/aprec/Projects/aprec/evaluation/results/checkpoints_for_rl/ml1m_supervised_pre_trained_checkpoint"
CHECKPOINT="/home/aprec/Projects/aprec/evaluation/results/checkpoints_for_rl/ml_popularity_tuned/pcount_lambda_3.5"

#1.0  for gae_gamma and gae_lambda allows the model to see and plan for the whole sequence
def generative_tuning_recommender(pcount_lambda=0.5, checkpoint_dir=CHECKPOINT, gae_gamma=1.0, gae_lambda=1.0, max_tuning_steps=32000):       
        import os
        #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        from aprec.recommenders.rl_generative.generative_tuning_recommender import GenerativeTuningRecommender
        from aprec.recommenders.sequential.models.generative.gpt_rec_rl import RLGPT2RecConfig
        from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig
        from aprec.recommenders.sequential.models.generative.reward_metrics.ild_reward import ILDReward
        from aprec.recommenders.sequential.models.generative.reward_metrics.ndcg_reward import NDCGReward
        from aprec.recommenders.sequential.models.generative.reward_metrics.weighted_sum_reward import WeightedSumReward

        model_config = RLGPT2RecConfig(transformer_blocks=3, embedding_size=256, tokenizer='id', tokens_per_item=1, values_per_dim=3500, attention_heads=4)
        recommender_config = SequentialRecommenderConfig(model_config, train_epochs=0) 
        recommender = GenerativeTuningRecommender(recommender_config,
                                                  pre_trained_checkpoint_dir=checkpoint_dir,
                                                  #pre_train_recommender_factory=lambda: RandomRecommender(),
                                                  max_tuning_steps=max_tuning_steps, 
                                                  tuning_batch_size=128, 
                                                  clip_eps=0.2, 
                                                  reward_metric=WeightedSumReward([NDCGReward(10), PCOUNT_REWARD], [1, pcount_lambda]),
                                                  tradeoff_monitoring_rewards=[(NDCGReward(10), PCOUNT_REWARD)],
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

        with open(CHECKPOINT + "/data_stats.json") as f:
            data_stats = json.load(f)        
        recommender.tune = lambda: None
        return recommender

recommenders = {
} 


#recommenders["mf_bpr"] = mf_bpr
#recommenders["bert4rec"] = full_bert 
#recommenders["vanilla_sasrec"] = vanilla_sasrec
#recommenders["top"] = TopRecommender
#recommenders[f"gptrec_supervised_checkpoint"] = lambda: generative_tuning_recommender(checkpoint_dir = CHECKPOINT)
recommenders[f"generative_tuning_recommender_pcount_3.5"] = lambda: generative_tuning_recommender(checkpoint_dir = CHECKPOINT)

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
